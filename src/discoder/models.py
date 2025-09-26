import json
from typing import Optional, Dict, Union

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
import torch.nn.functional as F
import dac
from dac.model.dac import Decoder as DACDecoder
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from discoder.modules import ResBlock, Activation1d, Snake1d


class DDP(torch.nn.parallel.DistributedDataParallel):
   def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class DisCoderEncoder(nn.Module):
    def __init__(self, in_channels: int, initial_out_channels: int, predict_type: str, activation: str, n_resblocks: int, resblock_type: str, intermediate_dim: int,
                 n_codebooks: int, codebook_size: int, codebook_dim: int, resblock_kernel_sizes, resblock_dilations):
        super(DisCoderEncoder, self).__init__()

        self.init_conv = weight_norm(nn.Conv1d(in_channels=in_channels, out_channels=initial_out_channels, kernel_size=7, padding=3, stride=1))
        self.sec_conv = weight_norm(nn.Conv1d(in_channels=initial_out_channels, out_channels=intermediate_dim, kernel_size=7, padding=3, stride=1))
        self.up_conv = weight_norm(nn.Conv1d(in_channels=intermediate_dim, out_channels=intermediate_dim, kernel_size=7, padding=3, stride=1))

        if activation == "relu":
            self.acts = [F.relu for _ in range(5)]
        elif activation == "gelu":
            self.acts = [F.gelu for _ in range(5)]
        elif activation == "snake":
            self.acts = nn.ModuleList([
                Activation1d(activation=Snake1d(initial_out_channels)),
                Activation1d(activation=Snake1d(intermediate_dim)),
                Activation1d(activation=Snake1d(intermediate_dim)),
                Activation1d(activation=Snake1d(initial_out_channels)),
                Activation1d(activation=Snake1d(codebook_size)),
            ])
        else:
            raise ValueError(f"Activation {activation} not defined")

        self.resblocks = nn.ModuleList([])
        for i in range(n_resblocks):
            if resblock_type == "AMP":
                block = ResBlock(channels=intermediate_dim, kernel_size=resblock_kernel_sizes[i], dilation=resblock_dilations[i])
            else:
                raise ValueError(f"Residual block type {resblock_type} not defined")
            self.resblocks.append(block)

        self.strided_conv = weight_norm(nn.Conv1d(in_channels=intermediate_dim, out_channels=initial_out_channels, kernel_size=7, padding=3, stride=2))
        self.down_conv = weight_norm(nn.Conv1d(in_channels=initial_out_channels, out_channels=codebook_size, kernel_size=7, padding=3, stride=1))

        if predict_type == "quant_latents":
            final_dim = n_codebooks * codebook_dim
        elif predict_type == "z":
            final_dim = codebook_size
        else:
            raise ValueError(f"Output size not defined for predict type {predict_type}")

        self.output_conv = nn.Conv1d(in_channels=codebook_size, out_channels=final_dim, kernel_size=7, padding=3, stride=1)


    def forward(self, x):
        """
        Returns the encoded representation (quantized_latents/z depending on config) and the first
        embedding for the skip connection to decoder.
        """
        x1 = self.init_conv(x)
        x = self.acts[0](x1)
        x = self.acts[1](self.sec_conv(x))
        x = self.acts[2](self.up_conv(x))

        residual = x
        for block in self.resblocks:
            x = block(x)

        x = x + residual
        x = self.acts[3](self.strided_conv(x))
        x = self.acts[4](self.down_conv(x))

        return self.output_conv(x), F.avg_pool1d(x1, kernel_size=2, stride=2)



class DisCoderDecoder(nn.Module):
    def __init__(self, dac_decoder: DACDecoder):
        super(DisCoderDecoder, self).__init__()
        self.dac_decoder = dac_decoder

    def forward(self, z: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if skip is not None:
            return self.dac_decoder(z + skip)
        return self.dac_decoder(z)


class DisCoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: dict, dac_decoder: DACDecoder, dac_encoder_quantizer):
        super(DisCoder, self).__init__()

        self.config = config
        self.dac_encoder_quantizer = dac_encoder_quantizer
        self.n_codebooks = config["model"]["n_codebooks"]
        self.codebook_dim = config["model"]["codebook_dim"]

        self.encoder = DisCoderEncoder(
            in_channels=config["mel"]["n_mels"],
            initial_out_channels=config["model"]["initial_out_channels"],
            predict_type=config["model"]["predict_type"],
            activation=config["model"]["activation"],
            n_resblocks=config["model"]["n_resblocks"],
            resblock_type=config["model"]["resblock_type"],
            intermediate_dim=config["model"]["intermediate_dim"],
            n_codebooks=config["model"]["n_codebooks"],
            codebook_size=config["model"]["codebook_size"],
            codebook_dim=config["model"]["codebook_dim"],
            resblock_kernel_sizes=config["model"]["resblock_kernel_sizes"],
            resblock_dilations=config["model"]["resblock_dilations"]
        )
        self.decoder = DisCoderDecoder(dac_decoder=dac_decoder)

        self.frozen_decoder = True
        self.frozen_quantizer = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        custom_out, x1 = self.encode(x)
        return self.decode(custom_out, skip=x1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes the mel spectrogram to specified representation (quantized latents, z, ...)"""
        custom_out, x1 = self.encoder(x)
        return custom_out, x1

    def decode(self, z: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Decodes the continuous representation after quantization [B, 1024, frames] to real audio."""
        if self.frozen_decoder:
            return self.decoder(z, None)  # if frozen decoder, no sip connection as we use normal DAC
        return self.decoder(z,skip)

    def project_quant_latents(self, latents_quant: torch.Tensor) -> torch.Tensor:
        """Uses the DAC encoder quantizer to project the quantized latents to z."""
        bs = latents_quant.shape[0]
        latents_per_codebook = latents_quant.reshape((bs, self.n_codebooks, self.codebook_dim, -1))
        z_q = 0
        for i, quantizer in enumerate(self.dac_encoder_quantizer.quantizers):
            # [B, n_codebooks, codebook_dim, #frames] -> [B, latent_dim, #frames]
            z_q_i = quantizer.out_proj(latents_per_codebook[:, i, ...])
            z_q = z_q + z_q_i
        return z_q

    def unfreeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = True

        self.decoder.train()
        self.frozen_decoder = False

    def unfreeze_quantizer(self):
        for param in self.dac_encoder_quantizer.parameters():
            param.requires_grad = True

        self.decoder.train()
        self.frozen_quantizer = False

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str,
        cache_dir: str,
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""

        # Load config
        config_file = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
        )
        with open(config_file, "r") as config_file:
            config = json.load(config_file)

        # Initialize model
        dac_model = dac.DAC.load(str(dac.utils.download(model_type="44khz")))
        model = cls(
            config=config,
            dac_decoder=dac_model.decoder,
            dac_encoder_quantizer=None,
        )

        # Load model
        model_file = hf_hub_download(
            repo_id=model_id,
            filename="model.pt",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
        )
        state_dict = torch.load(model_file, map_location=map_location)
        model_state_dict = {k.replace("module.", ""): v for k, v in state_dict["model_state_dict"].items()}
        model.load_state_dict(model_state_dict, strict=False)
        model.frozen_decoder = False

        return model


class DACEncoder(nn.Module):
    def __init__(self, dac_model, n_codebooks: int, codebook_dim: int):
        super(DACEncoder, self).__init__()
        self.dac_model = dac_model
        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dac_model.encode(x)


    def from_latents(self, x: torch.Tensor) -> torch.Tensor:
        return self.dac_model.quantizer.from_latents(x)[0]


    def from_codes(self, x: torch.Tensor) -> torch.Tensor:
        return self.dac_model.quantizer.from_codes(x)[0]


    def freeze(self):
        for param in self.dac_model.encoder.parameters():
            param.requires_grad = False

        for param in self.dac_model.quantizer.parameters():
            param.requires_grad = False

    def quantize_latents(self, latents):
        """Quantize continuous latents to quantized latents of the same shape [B, n_codebooks, codebook_dim, #frames]"""
        latents_per_codebook = latents.reshape((latents.shape[0], self.n_codebooks, self.codebook_dim, -1))
        quantized_vectors = []
        for i, quantizer in enumerate(self.dac_model.quantizer.quantizers):
            codebook_latent = latents_per_codebook[:, i, ...]
            z_q, indices = quantizer.decode_latents(codebook_latent)
            quantized_vectors.append(z_q)

        return torch.cat(quantized_vectors, dim=1)  # quantized vectors of shape  [B, 9*8, #frames]

    def project_quant_latents(self, latents_quant):
        """
        Project quantized latents [B, n_codebooks*codebook_dim, #frames] to continuous representation of the
        input after quantization [B, latent_dim, #frames]
        """
        bs = latents_quant.shape[0]
        latents_per_codebook = latents_quant.reshape((bs, self.n_codebooks, self.codebook_dim, -1))
        z_q = 0
        for i, quantizer in enumerate(self.dac_model.quantizer.quantizers):
            # [B, n_codebooks, codebook_dim, #frames] -> [B, latent_dim, #frames]
            z_q_i = quantizer.out_proj(latents_per_codebook[:, i, ...])
            z_q = z_q + z_q_i
        return z_q

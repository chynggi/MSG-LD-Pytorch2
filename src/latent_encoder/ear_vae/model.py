import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import pytorch_lightning as pl
from huggingface_hub import HfHubHTTPError, hf_hub_download

from .autoencoders import OobleckDecoder, OobleckEncoder

# -----------------------------------------------------------------------------
# Configuration utilities
# -----------------------------------------------------------------------------


def _ensure_path(path_like: Optional[str]) -> Optional[Path]:
    if path_like is None:
        return None
    path = Path(path_like).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def _download_from_hub(repo_id: str, filename_candidates: Sequence[str]) -> Path:
    errors = []
    for candidate in filename_candidates:
        try:
            return Path(hf_hub_download(repo_id, filename=candidate, repo_type="model"))
        except HfHubHTTPError as exc:  # pragma: no cover - network path
            errors.append((candidate, exc))
        except Exception as exc:  # pragma: no cover - best effort diagnostics
            errors.append((candidate, exc))
    tried = "\n".join(f"  - {name}: {err}" for name, err in errors)
    raise RuntimeError(
        "Unable to download required εar-VAE asset from Hugging Face.\n"
        f"Repository: {repo_id}\nTried:\n{tried}\n"
        "Please ensure you have accepted any model licenses and authenticated via 'huggingface-cli login'."
    )


def _load_json_config(repo_id: Optional[str], config_path: Optional[str]) -> Dict[str, Any]:
    path = _ensure_path(config_path)
    if path is None:
        if repo_id is None:
            raise ValueError("Either 'pretrained_name' or 'config_path' must be provided for EARVAE.")
        path = _download_from_hub(
            repo_id,
            (
                "config/model_config.json",
                "model_config.json",
                "ear_vae_config.json",
                "config.json",
            ),
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_checkpoint_path(repo_id: Optional[str], checkpoint_path: Optional[str]) -> Path:
    path = _ensure_path(checkpoint_path)
    if path is None:
        if repo_id is None:
            raise ValueError("Either 'pretrained_name' or 'checkpoint_path' must be provided for EARVAE.")
        path = _download_from_hub(
            repo_id,
            (
                "ear_vae_44k.pth",
                "ear_vae_44k.pyt",
                "ear_vae_44k.ckpt",
                "model.safetensors",
                "model.ckpt",
                "pytorch_model.bin",
            ),
        )
    return path


@dataclass
class EarVAEConfig:
    sample_rate: int = 44100
    downsampling_ratio: int = 1024
    latent_dim: int = 64
    encoder: Dict[str, Any] = None
    decoder: Dict[str, Any] = None
    stems: int = 4
    expected_channels: int = 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EarVAEConfig":
        model_cfg = data.get("model", data)
        encoder_cfg = model_cfg.get("encoder", {})
        decoder_cfg = model_cfg.get("decoder", {})
        sample_rate = int(data.get("sample_rate", 44100))
        downsampling_ratio = int(model_cfg.get("downsampling_ratio", _infer_downsampling_ratio(encoder_cfg)))
        latent_dim = int(model_cfg.get("latent_dim", encoder_cfg.get("latent_dim", 64)))
        stems = int(model_cfg.get("stems", data.get("stems", 4)))
        expected_channels = int(encoder_cfg.get("in_channels", 1))
        return cls(
            sample_rate=sample_rate,
            downsampling_ratio=downsampling_ratio,
            latent_dim=latent_dim,
            encoder=encoder_cfg,
            decoder=decoder_cfg,
            stems=stems,
            expected_channels=expected_channels,
        )


def _infer_downsampling_ratio(encoder_cfg: Dict[str, Any]) -> int:
    strides = encoder_cfg.get("strides")
    if not strides:
        return 1024
    ratio = 1
    for stride in strides:
        ratio *= int(stride)
    return ratio


# -----------------------------------------------------------------------------
# Main εar-VAE wrapper
# -----------------------------------------------------------------------------


class EARVAE(pl.LightningModule):
    """Lightweight wrapper around Descript's εar-VAE for MSG-LD pipelines."""

    def __init__(
        self,
        pretrained_name: Optional[str] = None,
        *,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        stems: int = 4,
        reduce_to_mono: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["config_path", "checkpoint_path"])

        cfg_dict = _load_json_config(pretrained_name, config_path)
        self.config = EarVAEConfig.from_dict(cfg_dict)
        if stems is not None:
            self.config.stems = int(stems)

        ckpt_path = _load_checkpoint_path(pretrained_name, checkpoint_path)

        encoder_cfg = dict(self.config.encoder or {})
        decoder_cfg = dict(self.config.decoder or {})
        latent_dim = self.config.latent_dim
        encoder_cfg.setdefault("latent_dim", latent_dim)
        decoder_cfg.setdefault("latent_dim", latent_dim)
        decoder_cfg.setdefault("out_channels", encoder_cfg.get("in_channels", 1))

        self.encoder = OobleckEncoder(**encoder_cfg)
        self.decoder = OobleckDecoder(**decoder_cfg)
        self.latent_dim = latent_dim
        self.downsampling_ratio = int(self.config.downsampling_ratio)
        self.sample_rate = int(self.config.sample_rate)
        self.expected_stems = int(self.config.stems)
        self.io_channels = int(self.config.expected_channels)
        self.reduce_to_mono = bool(reduce_to_mono)
        self.latent_frequency_bins = 1
        self.latent_channels = self.latent_dim
        self.z_channels = self.latent_channels

        state = torch.load(ckpt_path, map_location="cpu")
        state = _unwrap_state_dict(state)
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            print(f"[EARVAE] Warning: missing keys when loading checkpoint: {missing}")
        if unexpected:
            print(f"[EARVAE] Warning: unexpected keys when loading checkpoint: {unexpected}")

        self.eval()
        self.requires_grad_(False)

        self.outputs_waveform = True
        self.first_stage_key = "waveform_stems"
        self.image_key = "waveform"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_device(self) -> torch.device:
        return next(self.parameters()).device

    def _ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        device = self.get_device()
        if tensor.device != device:
            tensor = tensor.to(device)
        return tensor

    def _ensure_channel_count(self, audio: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        # audio: (B, C, T) expected
        channels = audio.shape[1]
        if channels == self.io_channels:
            return audio, False
        if channels == 1 and self.io_channels == 2:
            return audio.repeat(1, 2, 1), True
        if channels == 2 and self.io_channels == 1:
            return audio.mean(dim=1, keepdim=True), True
        raise ValueError(
            f"Unsupported channel conversion: input={channels}, expected={self.io_channels}."
        )

    def _trim_channels(self, audio: torch.Tensor, duplicated: bool) -> torch.Tensor:
        if not duplicated or not self.reduce_to_mono:
            return audio
        return audio.mean(dim=1, keepdim=True)

    def _assert_stride_alignment(self, audio: torch.Tensor) -> None:
        stride = self.downsampling_ratio
        remainder = audio.shape[-1] % stride
        if remainder != 0:
            raise ValueError(
                "εar-VAE expects audio lengths divisible by the downsampling ratio "
                f"({stride}). Received length {audio.shape[-1]} with remainder {remainder}."
            )

    # ------------------------------------------------------------------
    # Core encode/decode API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError("Expected waveform tensor with shape (batch, samples) or (batch, stems, samples).")

        batch, stems, samples = x.shape
        if self.expected_stems and stems != self.expected_stems:
            raise ValueError(
                f"Configured for {self.expected_stems} stems but received {stems}. Update the config or disable the check."
            )

        x = x.view(batch * stems, 1, samples)
        x = self._ensure_device(x.to(torch.float32))
        x, _ = self._ensure_channel_count(x)
        self._assert_stride_alignment(x)

        latents = self.encoder(x)
        latent_t = latents.shape[-1]
        latents = latents.view(batch, stems, self.latent_channels, latent_t, 1)
        return latents.contiguous()

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 4:
            z = z.unsqueeze(-1)
        if z.dim() != 5:
            raise ValueError("Expected latent tensor with shape (batch, stems, channels, T, F).")
        batch, stems, channels, latent_t, latent_f = z.shape
        if latent_f != 1:
            raise ValueError("εar-VAE latent frequency dimension must be 1.")
        if channels != self.latent_channels:
            raise ValueError(
                f"Latent channel mismatch: received {channels}, expected {self.latent_channels}."
            )
        z = z.view(batch * stems, channels, latent_t)
        z = self._ensure_device(z)

        audio = self.decoder(z)
        audio = self._trim_channels(audio, duplicated=(self.io_channels == 2))
        audio = audio.view(batch, stems, -1)
        return audio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    @property
    def vocoder_sample_rate(self) -> int:
        return self.sample_rate

    def expected_latent_shape(self, segment_length: int) -> Tuple[int, int]:
        latent_time = math.ceil(segment_length / self.downsampling_ratio)
        return latent_time, self.latent_frequency_bins

    # Lightning hooks - not used -------------------------------------------------
    def training_step(self, *args, **kwargs):  # pragma: no cover - inference only
        raise NotImplementedError("EARVAE is intended for inference/feature extraction only.")

    def configure_optimizers(self):  # pragma: no cover - inference only
        return []


def _unwrap_state_dict(state: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            inner = state["model"]
            if "state_dict" in inner:
                state = inner["state_dict"]
            else:
                state = inner
    state = {k.replace("module.", ""): v for k, v in state.items()}
    return state

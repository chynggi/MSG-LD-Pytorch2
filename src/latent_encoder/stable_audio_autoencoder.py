import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from huggingface_hub import HfHubHTTPError, hf_hub_download

from stable_audio_tools.models.autoencoders import create_autoencoder_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict


def _ensure_path(path_like: Optional[str]) -> Optional[Path]:
    if path_like is None:
        return None
    path = Path(path_like)
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
    joined = "\n".join(f"  - {name}: {err}" for name, err in errors)
    raise RuntimeError(
        "Unable to download required Stable Audio asset from Hugging Face.\n"
        f"Repository: {repo_id}\nTried:\n{joined}\n"
        "Please ensure you have accepted the model license and are authenticated via 'huggingface-cli login'."
    )


def _load_config(repo_id: Optional[str], config_path: Optional[str]) -> Dict[str, Any]:
    path = _ensure_path(config_path)
    if path is None:
        if repo_id is None:
            raise ValueError("Either 'pretrained_name' or 'config_path' must be provided.")
        path = _download_from_hub(repo_id, ("vae/config.json", "vae_model_config.json", "model_config.json"))
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_checkpoint(repo_id: Optional[str], checkpoint_path: Optional[str]) -> Path:
    path = _ensure_path(checkpoint_path)
    if path is None:
        if repo_id is None:
            raise ValueError("Either 'pretrained_name' or 'checkpoint_path' must be provided.")
        path = _download_from_hub(
            repo_id,
            (
                "vae/diffusion_pytorch_model.safetensors",
                "vae_model/diffusion_pytorch_model.safetensors",
                "vae_model.ckpt",
                "model.safetensors",
                "model.ckpt",
            ),
        )
    return path


class StableAudioAutoencoder(pl.LightningModule):
    """Wrapper around Stability AI's Stable Audio autoencoder for MSG-LD."""

    def __init__(
        self,
        pretrained_name: Optional[str] = "stabilityai/stable-audio-open-1.0",
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    latent_channels: int = 4,
    latent_frequency_bins: int = 16,
        stems: int = 4,
        chunk_size: int = 128,
        overlap: int = 32,
        encode_chunked: bool = False,
        decode_chunked: bool = False,
        reduce_to_mono: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["config_path", "checkpoint_path"],
        )

        config = _load_config(pretrained_name, config_path)
        checkpoint = _load_checkpoint(pretrained_name, checkpoint_path)

        self.autoencoder = create_autoencoder_from_config(config)
        state_dict = load_ckpt_state_dict(str(checkpoint))
        copy_state_dict(self.autoencoder, state_dict)
        self.autoencoder.eval()
        self.autoencoder.requires_grad_(False)

        model_cfg = config.get("model", {})
        self.sample_rate: int = int(config.get("sample_rate", 44100))
        self.downsampling_ratio: int = int(model_cfg.get("downsampling_ratio", 1024))
        self.latent_dim: int = int(model_cfg.get("latent_dim", 64))
        self.io_channels: int = int(model_cfg.get("io_channels", 2))

        expected_latent = latent_channels * latent_frequency_bins
        if expected_latent != self.latent_dim:
            raise ValueError(
                "Latent dimension mismatch: Stable Audio config reports "
                f"{self.latent_dim}, but latent_channels × latent_frequency_bins == {expected_latent}. "
                "Update the configuration so these values agree."
            )
        self.latent_channels = int(latent_channels)
        self.latent_frequency_bins = int(latent_frequency_bins)
        self.expected_stems = stems
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.encode_chunked = bool(encode_chunked)
        self.decode_chunked = bool(decode_chunked)
        self.reduce_to_mono = bool(reduce_to_mono)

        self.z_channels = self.latent_channels

        # Inform the outer pipeline that this autoencoder works on waveforms directly
        self.outputs_waveform = True
        self.first_stage_key = "waveform_stems"
        self.image_key = "waveform"

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device != next(self.autoencoder.parameters()).device:
            self.autoencoder.to(tensor.device)
        return tensor

    def _prepare_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Match channel count expected by the Stable Audio autoencoder."""
        if audio.dim() != 3:
            raise ValueError(f"Audio tensor must be 3-D (batch, channels, samples); got shape {tuple(audio.shape)}")
        batch, channels, _ = audio.shape
        if channels == self.io_channels:
            return audio, False
        if channels == 1 and self.io_channels == 2:
            return audio.repeat(1, 2, 1), True
        if channels == 2 and self.io_channels == 1:
            return audio.mean(dim=1, keepdim=True), True
        raise ValueError(
            "Unsupported channel remapping: input has "
            f"{channels} channels but Stable Audio expects {self.io_channels}."
        )

    def _restore_channels(self, audio: torch.Tensor, duplicated: bool) -> torch.Tensor:
        if not duplicated or not self.reduce_to_mono:
            return audio
        return audio.mean(dim=1, keepdim=True)

    # ------------------------------------------------------------------
    # Core encode/decode API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode waveform stems into diffusion latents."""
        if x.dim() not in (2, 3):
            raise ValueError("Expected input shape (batch, samples) or (batch, stems, samples).")

        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch, stems, samples = x.shape
        if self.expected_stems and stems != self.expected_stems:
            raise ValueError(
                f"Configured for {self.expected_stems} stems but received {stems}. Update the config or disable the check."
            )

        device = x.device
        x = x.reshape(batch * stems, samples)
        x = x.unsqueeze(1).to(torch.float32)  # (B*S, 1, samples)
        x = self._ensure_device(x)
        x, _ = self._prepare_audio(x)

        latents = self.autoencoder.encode_audio(
            x,
            chunked=self.encode_chunked,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )
        latents = latents.to(device)

        latent_time = latents.shape[-1]
        latents = latents.view(batch, stems, self.latent_dim, latent_time)
        latents = latents.view(
            batch,
            stems,
            self.latent_channels,
            self.latent_frequency_bins,
            latent_time,
        )
        latents = latents.permute(0, 1, 2, 4, 3).contiguous()
        return latents

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode diffusion latents back into waveform stems."""
        if z.dim() not in (4, 5):
            raise ValueError("Expected latent shape (batch, channels, T, F) or (batch, stems, channels, T, F).")

        if z.dim() == 4:
            z = z.unsqueeze(1)

        batch, stems, channels, latent_t, latent_f = z.shape
        if channels != self.latent_channels or latent_f != self.latent_frequency_bins:
            raise ValueError(
                "Latent shape mismatch: received channels="
                f"{channels}, freq_bins={latent_f}, expected "
                f"{self.latent_channels}×{self.latent_frequency_bins}."
            )

        device = z.device
        z = z.permute(0, 1, 3, 2, 4).contiguous()  # (B, stems, T, channels, F)
        z = z.view(batch * stems, latent_t, channels * latent_f)
        z = z.permute(0, 2, 1).contiguous()  # (B*S, latent_dim, T)
        z = self._ensure_device(z)

        audio = self.autoencoder.decode_audio(
            z,
            chunked=self.decode_chunked,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )
        audio = audio.to(device)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        if audio.shape[1] != 1:
            audio = audio.mean(dim=1, keepdim=True) if self.reduce_to_mono else audio[:, :1]

        audio = audio.view(batch * stems, 1, -1)
        return audio

    @torch.no_grad()
    def decode_to_waveform(self, z: torch.Tensor) -> np.ndarray:
        wave = self.decode(z)
        wave_np = wave.detach().cpu().numpy()
        return wave_np

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def get_device(self) -> torch.device:
        return next(self.autoencoder.parameters()).device

    # ------------------------------------------------------------------
    # Lightning hooks (unused but kept for API parity)
    # ------------------------------------------------------------------
    def training_step(self, *args, **kwargs):  # pragma: no cover - not used
        raise NotImplementedError("StableAudioAutoencoder is intended for inference/feature extraction only.")

    def configure_optimizers(self):  # pragma: no cover - not used
        return []

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def vocoder_sample_rate(self) -> int:
        return self.sample_rate

    def expected_latent_shape(self, segment_length: int) -> Tuple[int, int]:
        """Return (latent_time, latent_freq) for a given audio segment length."""
        latent_time = math.ceil(segment_length / self.downsampling_ratio)
        return latent_time, self.latent_dim

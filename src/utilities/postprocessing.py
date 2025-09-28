from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np
import torch

from utilities.audio.stft import TacotronSTFT
from utilities.model import get_vocoder

DISCODER_TARGET_SAMPLE_RATE = 44100


class DisCoderPostProcessor:
    def __init__(
        self,
        config: dict,
        preproc_cfg: dict,
        mel_bins: int,
        device: Optional[torch.device] = None,
    ) -> None:
        self._cfg = deepcopy(config)
        self._preproc_cfg = deepcopy(preproc_cfg)
        self._mel_bins = int(mel_bins)
        self._device = torch.device(device) if device is not None else torch.device("cpu")
        self._vocoder: Optional[torch.nn.Module] = None
        self._stft: Optional[TacotronSTFT] = None
        self._target_sample_rate = DISCODER_TARGET_SAMPLE_RATE
        self._file_suffix = self._cfg.get("file_suffix", "discoder")
        self._batch_size = int(self._cfg.get("batch_size", 4))

    @property
    def sample_rate(self) -> int:
        return DISCODER_TARGET_SAMPLE_RATE

    @property
    def file_suffix(self) -> str:
        return self._file_suffix

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def to(self, device: torch.device) -> "DisCoderPostProcessor":
        self._device = torch.device(device)
        if self._vocoder is not None:
            self._vocoder = self._vocoder.to(self._device)
        if self._stft is not None:
            self._stft = self._stft.to(self._device)
        return self

    def _ensure_modules(self, device: torch.device) -> None:
        if self._vocoder is None:
            cfg = deepcopy(self._cfg)
            cfg.setdefault("type", "discoder")
            cfg["target_sample_rate"] = DISCODER_TARGET_SAMPLE_RATE
            self._vocoder = get_vocoder(
                config=None,
                device=str(device),
                mel_bins=self._mel_bins,
                vocoder_config=cfg,
            )
        else:
            self._vocoder = self._vocoder.to(device)
        if self._stft is None:
            audio_cfg = self._preproc_cfg.get("audio", {})
            mel_cfg = self._preproc_cfg.get("mel", {})
            stft_cfg = self._preproc_cfg.get("stft", {})
            self._stft = TacotronSTFT(
                int(stft_cfg.get("filter_length", 1024)),
                int(stft_cfg.get("hop_length", 256)),
                int(stft_cfg.get("win_length", 1024)),
                int(mel_cfg.get("n_mel_channels", self._mel_bins)),
                int(audio_cfg.get("sampling_rate", DISCODER_TARGET_SAMPLE_RATE)),
                float(mel_cfg.get("mel_fmin", 0.0)),
                float(mel_cfg.get("mel_fmax", 8000.0)),
            ).to(device)
        else:
            self._stft = self._stft.to(device)

    def process_tensor(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)
        if waveform.dim() != 2:
            raise ValueError("waveform tensor must have shape [batch, time] or [batch, 1, time]")

        device = waveform.device if waveform.is_cuda else self._device
        self._ensure_modules(device)

        waveform = waveform.to(device=device, dtype=torch.float32)
        outputs = []
        step = max(1, self._batch_size)
        for start in range(0, waveform.size(0), step):
            chunk = waveform[start : start + step]
            mels, _, _ = self._stft.mel_spectrogram(chunk)
            audio = self._vocoder(mels)
            if isinstance(audio, (list, tuple)):
                audio = audio[0]
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
            outputs.append(audio.detach())
        return torch.cat(outputs, dim=0)

    def process_numpy(self, waveform: np.ndarray, device: Optional[torch.device] = None) -> np.ndarray:
        tensor = torch.from_numpy(waveform).float()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)
        if device is None:
            device = self._device
        tensor = tensor.to(device)
        audio = self.process_tensor(tensor.squeeze(1))
        return audio.detach().cpu().numpy()


def build_discoder_postprocessor(
    config: dict,
    preproc_cfg: dict,
    mel_bins: int,
    device: Optional[torch.device] = None,
) -> DisCoderPostProcessor:
    return DisCoderPostProcessor(config=config, preproc_cfg=preproc_cfg, mel_bins=mel_bins, device=device)

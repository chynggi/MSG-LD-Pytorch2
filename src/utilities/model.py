import json
import os
import sys
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # torchaudio may be optional in some environments
    import torchaudio.functional as AF
except ImportError:  # pragma: no cover - torchaudio is expected in runtime envs
    AF = None

import hifigan


def _import_discoder_class() -> Type[nn.Module]:
    """Import the local or installed DISCoder class and ensure path availability."""

    try:
        from discoder.models import DisCoder  # type: ignore[import-not-found]
        return DisCoder
    except ImportError as original_exc:  # pragma: no cover - requires optional dependency
        project_src = Path(__file__).resolve().parents[1]
        local_discoder_pkg = project_src / "discoder"

        if (local_discoder_pkg / "__init__.py").exists():
            if str(project_src) not in sys.path:
                sys.path.insert(0, str(project_src))
            try:
                return import_module("discoder.models").DisCoder  # type: ignore[attr-defined]
            except ImportError as secondary_exc:  # pragma: no cover - sanity check
                raise ImportError(
                    "Failed to import DISCoder from the bundled package even after updating sys.path."
                ) from secondary_exc

        raise ImportError(
            "DISCoder integration requires either the external 'discoder' package or the bundled copy under 'src/discoder'."
        ) from original_exc


def get_available_checkpoint_keys(model, ckpt):
    print("==> Attemp to reload from %s" % ckpt)
    state_dict = torch.load(ckpt)["state_dict"]
    current_state_dict = model.state_dict()
    new_state_dict = {}
    for k in state_dict.keys():
        if (
            k in current_state_dict.keys()
            and current_state_dict[k].size() == state_dict[k].size()
        ):
            new_state_dict[k] = state_dict[k]
        else:
            print("==> WARNING: Skipping %s" % k)
    print(
        "%s out of %s keys are matched"
        % (len(new_state_dict.keys()), len(state_dict.keys()))
    )
    return new_state_dict


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


class DisCoderVocoder(nn.Module):
    """Wrapper around the DISCoder vocoder for seamless integration."""

    def __init__(
        self,
        device: str,
        mel_bins: int,
        vocoder_config: Optional[Dict[str, Any]] = None,
        model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        if vocoder_config is None:
            vocoder_config = {}

        self.device = torch.device(device)
        self._input_mel_bins = mel_bins
        self._cfg = vocoder_config

        if model is None:
            model = self._load_model(vocoder_config)
        self.model = model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Resolve metadata with sensible fallbacks
        model_config = getattr(self.model, "config", {})
        mel_cfg = model_config.get("mel", {}) if isinstance(model_config, dict) else {}
        model_subcfg = model_config.get("model", {}) if isinstance(model_config, dict) else {}

        self.sample_rate = vocoder_config.get(
            "sample_rate",
            model_config.get("sample_rate", 44100) if isinstance(model_config, dict) else 44100,
        )
        self.expected_mel_bins = vocoder_config.get(
            "mel_bins",
            mel_cfg.get("n_mels", mel_bins) if isinstance(mel_cfg, dict) else mel_bins,
        )
        self.predict_type = vocoder_config.get(
            "predict_type",
            model_subcfg.get("predict_type", "z") if isinstance(model_subcfg, dict) else "z",
        )
        self.target_sample_rate = vocoder_config.get("target_sample_rate", self.sample_rate)
        self.vocoder_type = "discoder"

    def forward(self, mel: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if mel.dim() == 4:
            mel = mel.squeeze(1)

        input_device = mel.device
        mel = mel.to(self.device, dtype=torch.float32)

        if mel.shape[1] != self.expected_mel_bins:
            mel = F.interpolate(
                mel.unsqueeze(1),
                size=(self.expected_mel_bins, mel.shape[-1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        with torch.no_grad():
            audio = self.model(mel)

        if isinstance(audio, (list, tuple)):
            audio = audio[0]

        if self.target_sample_rate != self.sample_rate:
            if AF is None:
                raise ImportError(
                    "torchaudio is required to resample DISCoder outputs. Install torchaudio to proceed."
                )
            audio = AF.resample(audio, self.sample_rate, self.target_sample_rate)

        return audio.to(input_device)

    def _load_model(self, vocoder_config: Dict[str, Any]) -> nn.Module:
        DisCoder = _import_discoder_class()

        repo_id = vocoder_config.get("repo_id")
        revision = vocoder_config.get("revision", "main")
        map_location = vocoder_config.get("map_location", "cpu")

        if repo_id:
            model = DisCoder.from_pretrained(repo_id, revision=revision, map_location=map_location)
            model.frozen_decoder = False
            return model

        checkpoint_path = vocoder_config.get("checkpoint_path")
        config_path = vocoder_config.get("config_path")

        if not (checkpoint_path and config_path):
            raise ValueError(
                "To load DISCoder locally, both 'checkpoint_path' and 'config_path' must be provided in vocoder_config."
            )

        try:
            from discoder import utils as discoder_utils  # type: ignore[import-not-found]
            import dac  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Local DISCoder loading requires the 'descript-audio-codec' dependency. "
                "Please install discoder's requirements."
            ) from exc

        config = discoder_utils.read_config(config_path)
        sample_rate = config.get("sample_rate", 44100)
        sr_key = discoder_utils.sample_rate_str(sample_rate)
        if sr_key is None:
            raise ValueError(f"Unsupported sample rate {sample_rate} for DISCoder.")

        dac_model = dac.DAC.load(str(dac.utils.download(model_type=sr_key)))
        model = DisCoder(config=config, dac_decoder=dac_model.decoder, dac_encoder_quantizer=None)

        state = torch.load(checkpoint_path, map_location=map_location)
        state_dict = state.get("model_state_dict", state)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.frozen_decoder = False
        return model


def _load_hifigan_vocoder(
    device: str,
    mel_bins: int,
    ckpt_path: Optional[str],
    vocoder_config: Dict[str, Any],
    root: str,
) -> nn.Module:
    name = vocoder_config.get("name", "HiFi-GAN")
    speaker = vocoder_config.get("speaker", "")

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
        vocoder.vocoder_type = "melgan"
        target_sr = vocoder_config.get("target_sample_rate", 16000)
        vocoder.target_sample_rate = target_sr
        vocoder.expected_mel_bins = mel_bins
        return vocoder

    if mel_bins == 64:
        with open(os.path.join(root, "config_16k_64.json"), "r") as f:
            cfg = json.load(f)
        cfg = hifigan.AttrDict(cfg)
        vocoder = hifigan.Generator(cfg)
        if ckpt_path is not None:
            print("Load:", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)
        vocoder.target_sample_rate = vocoder_config.get("target_sample_rate", 16000)
        vocoder.expected_mel_bins = mel_bins
        vocoder.vocoder_type = "hifigan"
        return vocoder

    if mel_bins == 128:
        config_path = vocoder_config.get("config_path")
        if config_path is None:
            raise ValueError("HiFi-GAN with 128 mel bins requires 'config_path' in vocoder_config.")
        with open(config_path, "r") as f:
            cfg = json.load(f)
        cfg = hifigan.AttrDict(cfg)
        vocoder = hifigan.Generator(cfg)
        ckpt_path = vocoder_config.get("checkpoint_path") or ckpt_path
        if ckpt_path is None:
            raise ValueError("Checkpoint path must be provided for 128-bin HiFi-GAN vocoder.")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)
        vocoder.target_sample_rate = vocoder_config.get("target_sample_rate", 16000)
        vocoder.expected_mel_bins = mel_bins
        vocoder.vocoder_type = "hifigan"
        return vocoder

    raise ValueError(f"Unsupported mel bin configuration {mel_bins} for HiFi-GAN vocoder.")


def _load_discoder_vocoder(device: str, mel_bins: int, vocoder_config: Dict[str, Any]) -> nn.Module:
    return DisCoderVocoder(device=device, mel_bins=mel_bins, vocoder_config=vocoder_config)


def get_vocoder(
    config: Optional[Dict[str, Any]],
    device: str,
    mel_bins: int,
    ckpt_path: Optional[str] = None,
    vocoder_config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    root = "src/hifigan"

    cfg = deepcopy(vocoder_config) if vocoder_config is not None else {}
    cfg.setdefault("type", "hifigan")
    cfg.setdefault("target_sample_rate", _resolve_target_sample_rate(config, cfg))
    if ckpt_path is not None and "checkpoint_path" not in cfg:
        cfg["checkpoint_path"] = ckpt_path

    vocoder_label = cfg["type"]
    vocoder_type = "".join(ch for ch in vocoder_label.lower() if ch.isalnum())
    device_to_use = cfg.get("device", device)

    if vocoder_type in {"hifigan", "melgan"}:
        return _load_hifigan_vocoder(device_to_use, mel_bins, ckpt_path, cfg, root)
    if vocoder_type == "discoder":
        return _load_discoder_vocoder(device_to_use, mel_bins, cfg)

    raise ValueError(f"Unsupported vocoder type '{vocoder_label}'.")


def vocoder_infer(mels, vocoder, lengths=None):
    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)

    wavs = (wavs.cpu().numpy() * 32768).astype("int16")

    if lengths is not None:
        wavs = wavs[:, :lengths]

    # wavs = [wav for wav in wavs]

    # for i in range(len(mels)):
    #     if lengths is not None:
    #         wavs[i] = wavs[i][: lengths[i]]

    return wavs


def _resolve_target_sample_rate(
    config: Optional[Dict[str, Any]], vocoder_config: Dict[str, Any]
) -> int:
    if "target_sample_rate" in vocoder_config:
        return int(vocoder_config["target_sample_rate"])

    if config and isinstance(config, dict):
        target = (
            config.get("preprocessing", {})
            .get("audio", {})
            .get("sampling_rate")
        )
        if target is not None:
            return int(target)

    return 16000

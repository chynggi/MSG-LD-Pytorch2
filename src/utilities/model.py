import os
import json

import torch
import numpy as np
import resampy

from typing import Any, Dict, Optional

import hifigan

try:
    from discoder import DisCoder
except ImportError:  # pragma: no cover - optional dependency
    DisCoder = None  # type: ignore


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


def _infer_sample_rate(vocoder_config: Optional[Dict[str, Any]], config: Optional[Dict[str, Any]]) -> Optional[int]:
    if vocoder_config is not None and "sample_rate" in vocoder_config:
        return vocoder_config["sample_rate"]
    if config is None:
        return None
    try:
        return config["preprocessing"]["audio"]["sampling_rate"]
    except (KeyError, TypeError):
        return None


def get_vocoder(
    config: Optional[Dict[str, Any]],
    device: str,
    mel_bins: int,
    ckpt_path: Optional[str] = None,
    vocoder_config: Optional[Dict[str, Any]] = None,
):
    vocoder_cfg = vocoder_config or {}
    vocoder_type = vocoder_cfg.get("type", "hifigan").lower()
    sample_rate = _infer_sample_rate(vocoder_cfg, config)
    target_sample_rate = vocoder_cfg.get("target_sample_rate")
    device_obj = torch.device(device)

    if vocoder_type == "discoder":
        if DisCoder is None:
            raise ImportError(
                "DISCoder is not installed. Please install dependencies listed in requirements.txt."
            )
        repo_id = vocoder_cfg.get("repo_id", "disco-eth/discoder")
        revision = vocoder_cfg.get("revision")
        map_location = device_obj if device_obj.type == "cpu" else "cpu"
        model = DisCoder.from_pretrained(
            repo_id,
            revision=revision,
            map_location=map_location,
        )
        if sample_rate is not None and sample_rate != model.config.get("sample_rate"):
            print(
                f"[DISCoder] Warning: config sample rate {sample_rate} does not match pretrained model "
                f"{model.config.get('sample_rate')}"
            )
        expected_mels = model.config.get("mel", {}).get("n_mels")
        if expected_mels is not None and expected_mels != mel_bins:
            print(
                f"[DISCoder] Warning: model expects {expected_mels} mel bins but pipeline provides {mel_bins}."
            )
        model = model.to(device_obj)
        model.eval()
        setattr(model, "_vocoder_type", "discoder")
        if sample_rate is None:
            sample_rate = model.config.get("sample_rate")
        setattr(model, "_sample_rate", sample_rate)
        if target_sample_rate is not None:
            setattr(model, "_target_sample_rate", target_sample_rate)
        return model

    # Default to HiFi-GAN
    ROOT = "src/hifigan"
    with open(os.path.join(ROOT, "config_16k_64.json"), "r") as f:
        default_config = json.load(f)

    if mel_bins == 64:
        config_json = default_config
    elif mel_bins == 128:
        custom_config_path = vocoder_cfg.get("config_path")
        if custom_config_path is None:
            raise ValueError(
                "HiFi-GAN with 128 mel bins requires 'config_path' in vocoder configuration."
            )
        with open(custom_config_path, "r") as f:
            config_json = json.load(f)
    else:
        raise ValueError(f"Unsupported mel bin size {mel_bins} for HiFi-GAN vocoder")

    config_attr = hifigan.AttrDict(config_json)
    vocoder = hifigan.Generator(config_attr)
    if ckpt_path is not None:
        print("Load:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device_obj)
    setattr(vocoder, "_vocoder_type", "hifigan")
    setattr(vocoder, "_sample_rate", sample_rate or default_config.get("sampling_rate", 16000))
    if target_sample_rate is not None:
        setattr(vocoder, "_target_sample_rate", target_sample_rate)
    return vocoder


def _ensure_tensor(mels: Any, device: torch.device) -> torch.Tensor:
    if isinstance(mels, torch.Tensor):
        return mels.to(device)
    return torch.tensor(mels, device=device, dtype=torch.float32)


def _infer_vocoder_type(vocoder: Any, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    inferred = getattr(vocoder, "_vocoder_type", None)
    if inferred:
        return str(inferred)
    return vocoder.__class__.__name__.lower()


def _resample_batch(wavs: np.ndarray, native_rate: int, target_rate: int) -> np.ndarray:
    if wavs.ndim == 1:
        wavs = wavs[None, :]
    dtype = wavs.dtype
    resampled = []
    for wav in wavs:
        if dtype == np.int16:
            wav_float = wav.astype(np.float32) / 32768.0
        else:
            wav_float = wav.astype(np.float32)
        res = resampy.resample(wav_float, native_rate, target_rate)
        if dtype == np.int16:
            res = np.clip(res * 32768.0, -32768, 32767).astype(np.int16)
        else:
            res = res.astype(np.float32)
        resampled.append(res)

    max_len = max(x.shape[-1] for x in resampled)
    if any(x.shape[-1] != max_len for x in resampled):
        pad_value = 0 if dtype == np.int16 else 0.0
        padded = []
        for arr in resampled:
            if arr.shape[-1] < max_len:
                arr = np.pad(arr, (0, max_len - arr.shape[-1]), mode="constant", constant_values=pad_value)
            padded.append(arr)
        resampled = padded

    return np.stack(resampled, axis=0)


def vocoder_infer(mels, vocoder, *_, lengths=None, vocoder_type=None, target_sample_rate=None):
    device = next(vocoder.parameters()).device if hasattr(vocoder, "parameters") else torch.device("cpu")
    mels_tensor = _ensure_tensor(mels, device)
    if mels_tensor.dim() == 2:
        mels_tensor = mels_tensor.unsqueeze(0)
    if mels_tensor.size(1) < mels_tensor.size(2):  # likely [B, frames, mel]
        mels_tensor = mels_tensor.transpose(1, 2)

    inferred_type = _infer_vocoder_type(vocoder, vocoder_type)
    native_rate = getattr(vocoder, "_sample_rate", None)
    target_sample_rate = target_sample_rate or getattr(vocoder, "_target_sample_rate", None)

    with torch.no_grad():
        wavs = vocoder(mels_tensor).squeeze(1)

    wavs = wavs.detach().cpu()

    if lengths is not None:
        wavs = torch.stack([w[..., :int(l)] for w, l in zip(wavs, lengths)])

    wavs_np = wavs.numpy()

    if inferred_type == "discoder":
        output = wavs_np.astype("float32")
    else:
        output = (wavs_np * 32768).astype("int16")

    if (
        target_sample_rate is not None
        and native_rate is not None
        and target_sample_rate != native_rate
    ):
        output = _resample_batch(output, native_rate, target_sample_rate)

    return output

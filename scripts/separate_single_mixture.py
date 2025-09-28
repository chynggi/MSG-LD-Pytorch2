#!/usr/bin/env python
"""Standalone inference script for separating a single mixture file with MusicLDM.

The script loads a trained checkpoint, chunks the input mixture into segments
matching the model's receptive field, performs diffusion-based inference for
all segments, and reconstructs full-length stem waveforms via simple overlap-add.

Example
-------
python scripts/separate_single_mixture.py \
    --config config/MSG-LD/multichannel_musicldm_musdb18_eval.yaml \
    --checkpoint lightning_logs/.../checkpoints/last.ckpt \
    --mixture path/to/song.wav \
    --output-dir outputs/song_separation
"""
from __future__ import annotations

import argparse
import copy
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from tqdm.auto import tqdm

# Ensure "src" is on the Python path so that project modules can be imported.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from latent_diffusion.util import instantiate_from_config  # type: ignore[import-error]  # noqa: E402
from utilities.audio.stft import TacotronSTFT  # type: ignore[import-error]  # noqa: E402
from utilities.postprocessing import build_discoder_postprocessor  # type: ignore[import-error]  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separate a single mixture file using MusicLDM")
    parser.add_argument("--config", type=Path, required=True, help="YAML config used to train the model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint containing trained weights")
    parser.add_argument("--mixture", type=Path, required=True, help="Path to the input mixture audio file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write separated stems")
    parser.add_argument("--device", type=str, default=None, help="Computation device override, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of segments to process per diffusion batch")
    parser.add_argument("--overlap", type=float, default=0.0, help="Fractional overlap between consecutive segments (0-0.95)")
    parser.add_argument("--ddim-steps", type=int, default=200, help="Number of DDIM steps to use during sampling")
    parser.add_argument("--ddim-eta", type=float, default=1.0, help="DDIM sampling eta parameter controlling stochasticity")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Classifier-free guidance scale (1.0 disables guidance)")
    parser.add_argument("--use-plms", action="store_true", help="Use PLMS sampler instead of DDIM")
    parser.add_argument("--fp16", action="store_true", help="Run the diffusion model in torch.float16 precision where possible")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_device(preferred: str | None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config: dict, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model_cfg = copy.deepcopy(config["model"])
    # We load weights explicitly below, so avoid double-loading via ckpt_path.
    model_cfg.setdefault("params", {})["ckpt_path"] = None
    model = instantiate_from_config(model_cfg)

    state = torch.load(checkpoint, map_location="cpu")
    state_dict = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys while loading checkpoint: {sorted(missing)[:10]}")
    if unexpected:
        print(f"Warning: unexpected keys while loading checkpoint: {sorted(unexpected)[:10]}")

    model = model.to(device)
    model.eval()

    # Ensure auxiliary modules reside on the same device.
    if hasattr(model, "first_stage_model"):
        model.first_stage_model = model.first_stage_model.to(device)
        if hasattr(model.first_stage_model, "vocoder") and model.first_stage_model.vocoder is not None:
            model.first_stage_model.vocoder = model.first_stage_model.vocoder.to(device)
    if getattr(model, "cond_stage_model", None) is not None:
        model.cond_stage_model = model.cond_stage_model.to(device)
        # Patch callable that returns the current device (used by Patch_Cond_Model).
        if hasattr(model.cond_stage_model, "device"):
            model.cond_stage_model.device = lambda: device
    return model


def prepare_stft(preproc_cfg: dict, device: torch.device) -> TacotronSTFT:
    stft = TacotronSTFT(
        preproc_cfg["stft"]["filter_length"],
        preproc_cfg["stft"]["hop_length"],
        preproc_cfg["stft"]["win_length"],
        preproc_cfg["mel"]["n_mel_channels"],
        preproc_cfg["audio"]["sampling_rate"],
        preproc_cfg["mel"]["mel_fmin"],
        preproc_cfg["mel"]["mel_fmax"],
    )
    return stft.to(device)


def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    waveform = waveform - waveform.mean()
    peak = waveform.abs().max().clamp_min(1e-8)
    waveform = 0.5 * waveform / peak
    return waveform


def load_waveform(path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(str(path))
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    waveform = normalize_waveform(waveform.squeeze(0)).unsqueeze(0)
    return waveform, sr


def create_segments(
    waveform: torch.Tensor,
    segment_length: int,
    overlap: float,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    if not 0.0 <= overlap < 0.95:
        raise ValueError("overlap must be between 0.0 and 0.95")

    step = int(segment_length * (1.0 - overlap))
    step = max(1, step)
    if step > segment_length:
        step = segment_length

    total_len = waveform.size(-1)
    segments: List[torch.Tensor] = []
    starts: List[int] = []
    valids: List[int] = []

    cursor = 0
    while cursor < total_len:
        end = min(cursor + segment_length, total_len)
        chunk = waveform[..., cursor:end]
        valid = chunk.size(-1)
        if valid < segment_length:
            padding = segment_length - valid
            chunk = torch.nn.functional.pad(chunk, (0, padding))
        segments.append(chunk.squeeze(0))
        starts.append(cursor)
        valids.append(valid)
        cursor += step
        if cursor >= total_len and valid == segment_length:
            break
    stacked = torch.stack(segments) if segments else waveform.new_zeros((1, segment_length))
    return stacked, starts, valids


def window_taper(segment_length: int, overlap: float) -> np.ndarray:
    if overlap <= 0:
        return np.ones(segment_length, dtype=np.float32)
    fade = max(1, int(segment_length * overlap))
    window = np.ones(segment_length, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
    window[:fade] = ramp
    window[-fade:] = ramp[::-1]
    return window


def compute_fbank(
    stft: TacotronSTFT,
    segments: torch.Tensor,
) -> torch.Tensor:
    mels, _, _ = stft.mel_spectrogram(segments)
    # Output shape: [batch, n_mels, frames] -> transpose to [batch, frames, n_mels]
    return mels.permute(0, 2, 1)


def render_segments(
    model: torch.nn.Module,
    stft: TacotronSTFT,
    segments: torch.Tensor,
    batch_size: int,
    segment_length: int,
    overlap: float,
    guidance_scale: float,
    ddim_steps: int,
    ddim_eta: float,
    use_plms: bool,
    progress: bool,
    use_fp16: bool,
) -> np.ndarray:
    device = next(model.parameters()).device
    segments = segments.to(device)
    batch_size = max(1, batch_size)
    num_segments = segments.size(0)
    if use_fp16 and device.type == "cuda":
        cm_factory = lambda: torch.autocast(device_type=device.type, dtype=torch.float16)  # noqa: E731
    else:
        cm_factory = lambda: nullcontext()  # noqa: E731

    outputs: List[np.ndarray] = []
    iterator = range(0, num_segments, batch_size)
    if progress:
        iterator = tqdm(iterator, desc="Diffusion", unit="batch")

    with torch.no_grad():
        for start in iterator:
            end = min(start + batch_size, num_segments)
            chunk = segments[start:end]
            with cm_factory():
                fbanks = compute_fbank(stft, chunk)
            cond_input = fbanks.unsqueeze(1)  # [batch, 1, frames, mel_bins]
            cond = model.get_learned_conditioning(cond_input)
            if guidance_scale != 1.0:
                uncond = model.cond_stage_model.get_unconditional_condition(cond.size(0))
                uncond = uncond.to(device)
            else:
                uncond = None

            ddim_flag = not use_plms
            samples, _ = model.sample_log(
                cond=cond,
                batch_size=cond.shape[0],
                ddim=ddim_flag,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
                unconditional_guidance_scale=guidance_scale,
                unconditional_conditioning=uncond,
                use_plms=use_plms,
            )
            samples = model.adapt_latent_for_VAE_decoder(samples)
            mel = model.decode_first_stage(samples)
            waveform = model.mel_spectrogram_to_waveform(mel, save=False)
            waveform = waveform.astype(np.float32)
            np.clip(waveform, -1.0, 1.0, out=waveform)
            segment_waveforms = waveform.reshape(cond.shape[0], model.num_stems, -1)

            # Align waveform length with expected segment length.
            current_len = segment_waveforms.shape[-1]
            if current_len > segment_length:
                segment_waveforms = segment_waveforms[..., :segment_length]
            elif current_len < segment_length:
                pad = segment_length - current_len
                segment_waveforms = np.pad(
                    segment_waveforms,
                    ((0, 0), (0, 0), (0, pad)),
                    mode="constant",
                )
            outputs.append(segment_waveforms)
    return np.concatenate(outputs, axis=0)


def overlap_add(
    stems_segments: np.ndarray,
    starts: Sequence[int],
    valids: Sequence[int],
    segment_length: int,
    total_length: int,
    overlap: float,
) -> np.ndarray:
    num_segments, num_stems, _ = stems_segments.shape
    window = window_taper(segment_length, overlap)
    output = np.zeros((num_stems, total_length + segment_length), dtype=np.float32)
    weight = np.zeros(total_length + segment_length, dtype=np.float32)

    for idx in range(num_segments):
        start = starts[idx]
        valid = valids[idx]
        segment = stems_segments[idx, :, :valid]
        window_slice = window[:valid]
        output[:, start : start + valid] += segment * window_slice
        weight[start : start + valid] += window_slice

    weight = np.maximum(weight, 1e-4)
    result = output[:, :total_length] / weight[:total_length]
    return np.clip(result, -1.0, 1.0)


def save_outputs(
    stems: np.ndarray,
    stem_names: Sequence[str],
    sample_rate: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(stem_names):
        stem_path = output_dir / f"{name}.wav"
        sf.write(stem_path, stems[idx], sample_rate)
    mix_path = output_dir / "mixture_estimate.wav"
    sf.write(mix_path, stems.sum(axis=0), sample_rate)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_cfg = config["data"]["params"]
    preproc_cfg = data_cfg["preprocessing"]
    sample_rate = preproc_cfg["audio"]["sampling_rate"]
    hop_length = preproc_cfg["stft"]["hop_length"]
    target_length = preproc_cfg["mel"]["target_length"]
    segment_length = hop_length * target_length

    device = select_device(args.device)
    print(f"Using device: {device}")

    model = build_model(config, args.checkpoint, device)
    stft = prepare_stft(preproc_cfg, device)

    post_cfg = (config.get("postprocessing") or {}).get("discoder")
    postprocessor = None
    if post_cfg and post_cfg.get("enabled", True):
        mel_bins = preproc_cfg.get("mel", {}).get("n_mel_channels")
        if mel_bins is None:
            raise ValueError("Missing mel bin configuration required for discoder post-processing.")
        postprocessor = build_discoder_postprocessor(
            config=post_cfg,
            preproc_cfg=preproc_cfg,
            mel_bins=int(mel_bins),
            device=device,
        )
        print(f"DISCoder post-processing enabled (target {postprocessor.sample_rate} Hz)")

    waveform, sr = load_waveform(args.mixture, sample_rate)
    if sr != sample_rate:
        raise RuntimeError(f"Failed to resample mixture to {sample_rate} Hz")

    segments, starts, valids = create_segments(waveform, segment_length, args.overlap)
    print(f"Prepared {len(starts)} segment(s) of {segment_length} samples each")

    stems_segments = render_segments(
        model,
        stft,
        segments,
        batch_size=args.batch_size,
        segment_length=segment_length,
        overlap=args.overlap,
        guidance_scale=args.guidance_scale,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        use_plms=args.use_plms,
        progress=not args.no_progress,
        use_fp16=args.fp16,
    )

    full_length = waveform.size(-1)
    stems = overlap_add(stems_segments, starts, valids, segment_length, full_length, args.overlap)

    if postprocessor is not None:
        upsampled = postprocessor.process_numpy(stems, device=device)
        stems = upsampled[:, 0, :]
        sample_rate = postprocessor.sample_rate

    stem_names = data_cfg.get("path", {}).get("stems")
    if not stem_names:
        stem_names = [f"stem_{idx}" for idx in range(stems.shape[0])]

    save_outputs(stems, stem_names, sample_rate, args.output_dir)
    print(f"Saved stems to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

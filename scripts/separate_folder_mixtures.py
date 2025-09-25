#!/usr/bin/env python
"""Batch inference script for separating all mixtures inside a folder with MusicLDM.

This utility reuses the single-file separation helpers but iterates over every
supported audio file in a directory (optionally recursively). Each mixture is
processed independently and the corresponding stems are written to a dedicated
sub-folder under the requested output directory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from tqdm.auto import tqdm

# Ensure repository sources and helper scripts are importable when executed directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.separate_single_mixture import (  # noqa: E402
    create_segments,
    load_config,
    load_waveform,
    overlap_add,
    prepare_stft,
    render_segments,
    save_outputs,
    select_device,
    build_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate every mixture file inside a folder using MusicLDM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config used to train the model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint containing trained weights")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing mixture audio files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where stem folders will be written")
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="*",
        default=[".wav", ".flac", ".mp3", ".ogg"],
        help="File extensions (lowercase) treated as audio mixtures",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for audio files")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files whose output folder already exists")
    parser.add_argument("--device", type=str, default=None, help="Computation device override, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of segments to process per diffusion batch")
    parser.add_argument("--overlap", type=float, default=0.0, help="Fractional overlap between consecutive segments (0-0.95)")
    parser.add_argument("--ddim-steps", type=int, default=200, help="Number of DDIM steps to use during sampling")
    parser.add_argument("--ddim-eta", type=float, default=1.0, help="DDIM sampling eta parameter controlling stochasticity")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Classifier-free guidance scale (1.0 disables guidance)")
    parser.add_argument("--use-plms", action="store_true", help="Use PLMS sampler instead of DDIM")
    parser.add_argument("--fp16", action="store_true", help="Run diffusion in torch.float16 precision when supported")
    parser.add_argument(
        "--per-file-progress",
        action="store_true",
        help="Show an inner progress bar for diffusion batches of each file",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit the number of mixtures processed (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume by skipping files lexicographically before the provided stem name",
    )
    return parser.parse_args()


def discover_audio_files(root: Path, extensions: Sequence[str], recursive: bool) -> List[Path]:
    extensions = tuple(ext.lower() for ext in extensions)
    if recursive:
        candidates: Iterable[Path] = root.rglob("*")
    else:
        candidates = root.glob("*")
    return sorted(
        [p for p in candidates if p.is_file() and p.suffix.lower() in extensions]
    )


def separate_one_file(
    mixture_path: Path,
    output_root: Path,
    model: torch.nn.Module,
    stft,
    sample_rate: int,
    segment_length: int,
    stem_names: Sequence[str],
    args: argparse.Namespace,
) -> None:
    waveform, sr = load_waveform(mixture_path, sample_rate)
    if sr != sample_rate:
        raise RuntimeError(f"Failed to resample mixture to {sample_rate} Hz")

    segments, starts, valids = create_segments(waveform, segment_length, args.overlap)
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
        progress=args.per_file_progress,
        use_fp16=args.fp16,
    )

    full_length = waveform.size(-1)
    stems = overlap_add(stems_segments, starts, valids, segment_length, full_length, args.overlap)

    target_dir = output_root / mixture_path.stem
    save_outputs(stems, stem_names, sample_rate, target_dir)


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    audio_files = discover_audio_files(args.input_dir, args.extensions, args.recursive)
    if args.resume_from is not None:
        audio_files = [p for p in audio_files if p.stem >= args.resume_from]
    if not audio_files:
        print("No audio files found with the requested extensions.")
        return
    if args.max_files is not None:
        audio_files = audio_files[: args.max_files]

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

    stem_names = data_cfg.get("path", {}).get("stems")
    if not stem_names:
        num_stems = getattr(model, "num_stems", None)
        if num_stems is None:
            raise AttributeError("Model does not expose 'num_stems'; please specify stems list in the config")
        stem_names = [f"stem_{idx}" for idx in range(num_stems)]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures: List[Tuple[Path, Exception]] = []
    iterator = tqdm(audio_files, desc="Mixtures", unit="track")
    for mixture_path in iterator:
        target_dir = args.output_dir / mixture_path.stem
        if args.skip_existing and target_dir.exists():
            iterator.set_postfix_str("skipped")
            continue
        try:
            separate_one_file(
                mixture_path,
                args.output_dir,
                model,
                stft,
                sample_rate,
                segment_length,
                stem_names,
                args,
            )
            iterator.set_postfix_str("done")
        except Exception as exc:  # noqa: BLE001
            failures.append((mixture_path, exc))
            iterator.set_postfix_str("failed")
            print(f"Failed to process {mixture_path}: {exc}")

    if failures:
        print("\nThe following files failed to process:")
        for path, exc in failures:
            print(f" - {path}: {exc}")
    else:
        print("\nAll mixtures processed successfully.")


if __name__ == "__main__":
    main()

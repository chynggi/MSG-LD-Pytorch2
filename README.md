# Simultaneous Music Separation and Generation Using Multi-Track Latent Diffusion Models

This is the official repository for: [Simultaneous Music Separation and Generation Using Multi-Track Latent Diffusion Models](https://arxiv.org/pdf/2409.12346).

The paper is published at **ICASSP 2025**.

Diffusion models have recently shown strong potential in both music generation and music source separation tasks. Although in early stages, a trend is emerging towards integrating these tasks into a single framework, as both involve generating musically aligned parts and can be seen as facets of the same generative process. In this work, we introduce a latent diffusion-based multi-track generation model capable of both source separation and multi-track music synthesis by learning the joint probability distribution of tracks sharing a musical context. Our model also enables arrangement generation by creating any subset of tracks given the others. We trained our model on the Slakh2100 dataset, compared it with an existing simultaneous generation and separation model, and observed significant improvements across objective metrics for source separation, music, and arrangement generation tasks. 

Sound examples are available at https://msg-ld.github.io/

# Installation

To install MSG-LD, follow these steps:

Clone the repository to your local machine
```bash
$ git clone https://github.com/chynggi/MSG-LD-Pytorch2
```

To run the code in this repository, you will need python 3.9+ 

Navigate to the project directory and install the required dependencies




# Data

In this project, the Slakh2100 data is used by default.

Please follow the instructions for data download and set up given here:

https://github.com/gladia-research-group/multi-source-diffusion-models/blob/main/data/README.md

### MUSDB18-HQ support

MSG-LD now ships with a dedicated multi-track dataloader for [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html).

1. Download and extract the official MUSDB18-HQ package so that you have `train/` and `test/` sub-directories containing stem files (`bass.wav`, `drums.wav`, `other.wav`, `vocals.wav`).
2. Update the new configuration templates to point at those folders:
	- `config/MSG-LD/multichannel_musicldm_musdb18_train.yaml`
	- `config/MSG-LD/multichannel_musicldm_musdb18_eval.yaml`
3. Optionally adjust the `stems` list if you prepared custom subsets (defaults match the canonical four-stem layout).
4. The datamodule re-samples the 44.1 kHz stems to the internal 16 kHz rate automatically, so no additional preprocessing is required.

## Vocoder options

The first-stage autoencoder now supports both HiFi-GAN and the DISCoder vocoder. HiFi-GAN remains the default, but you can opt into DISCoder by extending the `ddconfig` block:

```yaml
first_stage_config:
	params:
		ddconfig:
			# existing settings …
			hifigan_ckpt: lightning_logs/musicldm_checkpoints/hifigan-ckpt.ckpt
			vocoder:
				type: discoder
				repo_id: disco-eth/discoder        # use checkpoint/config paths for offline runs
				revision: main
				target_sample_rate: 16000          # resample DISCoder output to match the training rate
```

> **Note:** DISCoder expects 128-bin mel spectrograms and natively operates at 44.1 kHz. The wrapper bundled with MSG-LD automatically interpolates 64-bin mels and rescales the decoded audio back to `target_sample_rate`, so you can reuse existing checkpoints without retraining.

If you prefer working offline, replace `repo_id`/`revision` with local `checkpoint_path` and `config_path` entries. Make sure you install the additional dependencies listed in the updated `requirements.txt` (`huggingface_hub`, `descript-audio-codec`, and the DISCoder repository).

# Training MSG-LD

After data and conda evn are intalled properlly, you will need to dowload components of MusicLDM that are used for MSG-LD too. For this please 

```
# Download hifigan-ckpt.ckpt
wget https://zenodo.org/record/10643148/files/hifigan-ckpt.ckpt

# Download vae-ckpt.ckpt
wget https://zenodo.org/record/10643148/files/vae-ckpt.ckpt

```

After placing this in some directory and changing corresponding links in the config file, train MSG-LD with one of the provided configurations. For example:

```bash
# Slakh2100 (default)
python train_musicldm.py --config config/MSG-LD/multichannel_musicldm_slakh_3d_train.yaml

# MUSDB18-HQ (new)
python train_musicldm.py --config config/MSG-LD/multichannel_musicldm_musdb18_train.yaml
```
<!-- 
# Checkpoints

Plase download checkpoints from:

```
# For un-conditional:
wget https://zenodo.org/records/13947715/files/2024-03-24T19-51-37_3_D_4_stems_slakh_uncond_ch%3D192_3e-05_.tar.gz?download=1

# For conditional:
wget https://zenodo.org/records/13947715/files/2024-03-25T00-55-31_3_D_4_stems_slakh_with_CALP_ch%3D192_3e-05_.tar.gz?download=1
``` -->

# Inference

For **separation** and **total generation**, use the following command. Adjust the `unconditional_guidance_scale` parameter as follows:
- Set `unconditional_guidance_scale` to `0` for total generation in unconditional mode.
- Set `unconditional_guidance_scale` to `1` or `2` for conditional generation, which performs separation.

```bash
# Separation and Total Generation (Slakh):
python train_musicldm.py --config config/MSG-LD/multichannel_musicldm_slakh_3d_eval.yaml

# Separation and Total Generation (MUSDB18-HQ):
python train_musicldm.py --config config/MSG-LD/multichannel_musicldm_musdb18_eval.yaml
```

For **arrangement generation**, run the command below and specify the instrument(s) you want to generate in the `stems_to_inpaint` parameter.

```bash
# Arrangement Generation:
python train_musicldm.py --config config/MSG-LD/multichannel_musicldm_slakh_3d_eval_inpaint.yaml
```

## 🔧 Quick single-file separation

When you only need to demix a *single* mixture file (instead of running the full
evaluation loop), use the lightweight helper in `scripts/separate_single_mixture.py`:

```bash
python scripts/separate_single_mixture.py \
	--config config/MSG-LD/multichannel_musicldm_musdb18_eval.yaml \
	--checkpoint /path/to/lightning_logs/.../checkpoints/last.ckpt \
	--mixture path/to/song.wav \
	--output-dir outputs/song
```

Key flags:

- `--overlap`: cross-fade ratio between consecutive segments (defaults to `0`).
- `--guidance-scale`: classifier-free guidance strength (same semantics as the
	Lightning evaluation entry point).
- `--use-plms`: switch from DDIM to PLMS sampling.
- `--batch-size`: number of chunks processed in parallel (useful when you have
	multiple GPUs).

The script writes one WAV file per stem plus a reconstructed mixture to the
requested output directory.

## 🚀 Batch folder separation

To demix an entire directory of songs in one go, run the batch helper. It
reuses the same weights and parameters as the single-track script but iterates
over every supported audio file in the folder (recursively if requested):

```bash
python scripts/separate_folder_mixtures.py \
	--config config/MSG-LD/multichannel_musicldm_musdb18_eval.yaml \
	--checkpoint /path/to/lightning_logs/.../checkpoints/last.ckpt \
	--input-dir path/to/mixtures \
	--output-dir outputs/batch
```

Useful optional flags:

- `--recursive`: descend into subfolders when searching for audio files.
- `--skip-existing`: continue from where you left off by skipping already
	rendered stem folders.
- `--extensions .wav .flac`: control which file endings count as mixtures.
- `--per-file-progress`: enables the inner diffusion progress bar if you want
	detailed feedback per song.


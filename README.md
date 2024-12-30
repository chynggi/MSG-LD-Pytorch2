# Simultaneous Music Separation and Generation Using Multi-Track Latent Diffusion Models

This is the official repository for: [Simultaneous Music Separation and Generation Using Multi-Track Latent Diffusion Models](https://arxiv.org/pdf/2409.12346).

The paper is published at **ICASSP 2025**.

Diffusion models have recently shown strong potential in both music generation and music source separation tasks. Although in early stages, a trend is emerging towards integrating these tasks into a single framework, as both involve generating musically aligned parts and can be seen as facets of the same generative process. In this work, we introduce a latent diffusion-based multi-track generation model capable of both source separation and multi-track music synthesis by learning the joint probability distribution of tracks sharing a musical context. Our model also enables arrangement generation by creating any subset of tracks given the others. We trained our model on the Slakh2100 dataset, compared it with an existing simultaneous generation and separation model, and observed significant improvements across objective metrics for source separation, music, and arrangement generation tasks. 

Sound examples are available at https://msg-ld.github.io/

# Installation

To install MSG-LD, follow these steps:

Clone the repository to your local machine
```bash
$ git clone https://github.com/karchkha/MSG-LD
```

To run the code in this repository, you will need python 3.9 

Navigate to the project directory and install the required dependencies

If you already installed conda before, skip this step, otherwise:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Then, after installing you should make sure your conda environment is running on your bash


```
conda env create -f musicldm_env.yml
``` 


then 
```
conda activate musicldm_env
```


# Data

In this project, the Slakh2100 data is used.

Please follow the instructions for data download and set up given here:

https://github.com/gladia-research-group/multi-source-diffusion-models/blob/main/data/README.md

# Training MSG-LD

After data and conda evn are intalled properlly, you will need to dowload components of MusicLDM that are used for MSG-LD too. For this please 

```
# Download hifigan-ckpt.ckpt
wget https://zenodo.org/record/10643148/files/hifigan-ckpt.ckpt

# Download vae-ckpt.ckpt
wget https://zenodo.org/record/10643148/files/vae-ckpt.ckpt

```

After placing this in some directory and changing corresponding links in the config file, for the trainion of MSG-LD please run:

```
python train_musicldm.py --config config/MSG-LD/multichannel_musicldm_slakh_3d_train.yaml
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
# Separation and Total Generation:
python train_musicldm.py --config config/MSG-LD/multichannel_musicldm_slakh_3d_eval.yaml
```

For **arrangement generation**, run the command below and specify the instrument(s) you want to generate in the `stems_to_inpaint` parameter.

```bash
# Arrangement Generation:
python train_musicldm.py --config config/MSG-LD/multichannel_musicldm_slakh_3d_eval_inpaint.yaml
```


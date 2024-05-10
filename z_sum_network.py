import sys
import wandb

sys.path.append("src")

import os
import numpy as np

import argparse
import yaml
import torch
import time
import datetime
from pathlib import Path

from pytorch_lightning.strategies.ddp import DDPStrategy
from latent_diffusion.models.musicldm import MusicLDM

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from utilities.tools import listdir_nohidden, get_restore_step, copy_test_subset_data
from utilities.data.dataset import AudiostockDataset

from latent_diffusion.util import instantiate_from_config
    

config = yaml.load(open("config/z_sum_net/multichannel_musicldm_slakh_z_sum_net.yaml", 'r'), Loader=yaml.FullLoader)

# seed_everything(0)
batch_size = config["data"]["params"]["batch_size"]
log_path = config["log_directory"]


print(f'Batch Size {batch_size} | Log Folder {log_path}')

data = instantiate_from_config(config["data"])
# data.prepare_data()
# data.setup()

latent_diffusion = MusicLDM(**config["model"]["params"]) #to("cuda:0")





#=============================================================================================================================



import matplotlib.pyplot as plt

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from latent_diffusion.models.ddim import DDIMSampler


from tqdm import tqdm
from latent_diffusion.modules.diffusionmodules.util import noise_like
import IPython.display as ipd
from torch.utils.checkpoint import checkpoint
import soundfile as sf
from pytorch_lightning.loggers import WandbLogger
import torchaudio



log_project = os.path.join(log_path, "z_sum_net")
os.makedirs(log_project, exist_ok=True)


# Initialize wandb logging
wandb_logger = WandbLogger(project="z_sum_net", log_model=False, save_dir=log_project)




class z_sum_net(pl.LightningModule):
    def __init__(self, model, learning_rate=0.001, dataset=None):
        super().__init__()

        # Define the initial convolutional layer
        self.conv1 = nn.Conv3d(in_channels=4, out_channels=4, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # Define the final convolutional layers to adjust dimensions
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=(1, 1))


        self.learning_rate = learning_rate

        # Example of a frozen sub-network (let's use a simple linear layer)
        self.frozen_network = model

    def get_input(self, batch):
        z, _ = latent_diffusion.get_input(batch, latent_diffusion.first_stage_key)
        z_mix, _ = latent_diffusion.get_input(batch, "fbank")
        return z, z_mix

    def forward(self, z):

        # Apply the first convolution
        z = self.conv1(z)  # shape becomes [bs, 4, 8, 256, 16]

        # Sum along the dimension 1 (size 4), maintaining the other dimensions
        z = torch.sum(z, dim=1)  # shape becomes [bs, 8, 256, 16]

        # Apply the second convolution
        z = self.conv2(z)  # adjusting channel dimensions to match the output
        
        return z
            

    def configure_optimizers(self):
        # Explicitly optimize only self.vector
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


    def latent_to_waveform(self, z):

        mel = self.frozen_network.decode_first_stage(z)

        waveform = self.frozen_network.mel_spectrogram_to_waveform(mel, save=False)
        waveform = np.nan_to_num(waveform)
        waveform = np.clip(waveform, -1, 1)
        return mel.numpy(), waveform

    def training_step(self, batch, batch_idx):

        z, z_mix = self.get_input(batch)
        
        z_mix_hat = self(z)


        # Compute MSE loss on waveform
        loss = nn.functional.mse_loss(z_mix_hat, z_mix)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):
        z, z_mix = self.get_input(batch)
        z_mix_hat = self(z)
        val_loss = nn.functional.mse_loss(z_mix_hat, z_mix)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        
        if batch_idx == 0:  # Optionally log only for the first batch of each epoch
            self.log_audio(z_mix, z_mix_hat, 'val')

        return val_loss

    def log_audio(self, z_mix, z_mix_hat, stage):

        # Log audio and mel spectrogram
        mel_mix, waveform_mix = self.latent_to_waveform(z_mix)
        mel_mix_hat, waveform_mix_hat = self.latent_to_waveform(z_mix_hat)


        sample_rate = 16000  # Assuming the sample rate is 16000Hz
        
        for i in range(waveform_mix.shape[0]):
            log_dict = {}
            audio_clip = waveform_mix[i][0]
            log_dict[f"{stage} Orig Audio"] = wandb.Audio(audio_clip, sample_rate=sample_rate, caption=f"{stage} Sample {i}")
            log_dict[f"{stage} Orig Mel"] = wandb.Image(mel_mix[i].squeeze().T, caption=f"{stage} Mel {i}")

            audio_clip = waveform_mix_hat[i][0]
            log_dict[f"{stage} Reconst Audio"] = wandb.Audio(audio_clip, sample_rate=sample_rate, caption=f"{stage} Sample {i}")
            log_dict[f"{stage} Reconst Mel"] = wandb.Image(mel_mix_hat[i].squeeze().T, caption=f"{stage} Mel {i}")

            # wandb.log(log_dict)
            self.logger.experiment.log(log_dict)

# Model
model = z_sum_net(model = latent_diffusion) #.to("cuda:0")


# Specify the path to your checkpoint
checkpoint_path = None #"/home/karchkhadze/MusicLDM-Ext/lightning_logs/z_sum_net/z_sum_net/emur7c9k/checkpoints/epoch=53-step=540.ckpt"

# # Train
trainer = Trainer(max_epochs=10000, 
                logger=wandb_logger,
                num_sanity_val_steps=0,
                accelerator="cpu", 
                limit_train_batches= 2,
                limit_val_batches=2,
                # accelerator="gpu", 
                # devices = [0],
                    # default_root_dir=default_root_dir,
                log_every_n_steps=1,
                resume_from_checkpoint=checkpoint_path,
                )

trainer.fit(model, data)


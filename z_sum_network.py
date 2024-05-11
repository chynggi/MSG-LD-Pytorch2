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
import datetime

# Current date and time for unique folder names
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Base directory for logs and checkpoints
base_log_dir = Path(config["log_directory"]) / "z_sum_net"
run_log_dir = base_log_dir / f"run_{current_datetime}"
os.makedirs(run_log_dir, exist_ok=True)

# Save the configuration file in the run directory
config_file_path = run_log_dir / "config.yaml"
with open(config_file_path, 'w') as file:
    yaml.dump(config, file)

# Setting up the checkpoint callback to save checkpoints in a subdirectory named after the current date and time
checkpoint_callback = ModelCheckpoint(
    dirpath=run_log_dir / "checkpoints",
    filename="{epoch}-{step}",
    save_top_k=1,  # Save only the best checkpoint
    monitor="val_loss",  # Assuming you want to monitor validation loss
    mode="min",
    save_last=True
)

# Initialize wandb logging
wandb_logger = WandbLogger(project="z_sum_net", log_model=False, save_dir=run_log_dir)
wandb_logger.experiment.name = f"z_sum_net_{current_datetime}"  # Naming the WandB run



class ZSum_conv_simple_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(4, 4, (1, 3, 3), padding=(0, 1, 1))
        self.conv2 = nn.Conv2d(8, 8, (3, 3), padding=1)

    def forward(self, x):
        # Apply the first convolution
        x = self.conv1(x)
        # Sum along the dimension 1 (size 4), maintaining the other dimensions
        x = torch.sum(x, dim=1)
        # Apply the second convolution
        x = self.conv2(x)
        return x


class ZSum_conv_bigger_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(4, 8, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(8)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv3d(8, 8, (2, 1, 1), stride=(2, 1, 1))  # This will reduce the second dimension from 8 to 4
        self.bn2 = nn.BatchNorm3d(8)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv3d(8, 8, (2, 1, 1), stride=(2, 1, 1))  # Further reduce the second dimension from 4 to 2
        self.bn3 = nn.BatchNorm3d(8)
        self.relu3 = nn.ReLU()
        
        # Additional layers can be added here
        
        self.conv_final = nn.Conv2d(16, 8, (1, 1))  # Adjust this to match the dimensions you need to output

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Adjust shape for 2D convolution
        x = x.view(x.size(0), 16, 256, 16)  # Adjust the view depending on the exact output size needed
        
        x = self.conv_final(x)
        return x


class Basic3DResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(out_channels)
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

class ZSum_conv_Advanced_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Basic3DResBlock(4, 8)
        self.conv1 = nn.Conv3d(8, 8, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.layer2 = Basic3DResBlock(8, 8)
        self.conv2 = nn.Conv3d(8, 8, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.layer3 = Basic3DResBlock(8, 8)
        self.conv3 = nn.Conv3d(8, 8, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.final_conv = nn.Conv3d(8, 8, kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.layer3(x)
        x = self.conv3(x)
        # x = self.final_conv(x)  # This conv layer is designed to reduce the second dimension from 8 to 4
        x = x.squeeze(2)  # This removes the singleton dimension, adapting from [bs, 8, 1, 256, 16] to [bs, 8, 256, 16]
        return x


class ZSum_conv_big_Advanced_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Basic3DResBlock(4, 32)
        self.conv1 = nn.Conv3d(32, 32, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.layer2 = Basic3DResBlock(32, 16)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.layer3 = Basic3DResBlock(16, 8)
        self.conv3 = nn.Conv3d(8, 8, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.final_conv = nn.Conv3d(8, 8, kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.layer3(x)
        x = self.conv3(x)
        # x = self.final_conv(x)  # This conv layer is designed to reduce the second dimension from 8 to 4
        x = x.squeeze(2)  # This removes the singleton dimension, adapting from [bs, 8, 1, 256, 16] to [bs, 8, 256, 16]
        return x



class ZSum_conv_huge_Advanced_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_4 = Basic3DResBlock(4, 1024)
        self.layer_3 = Basic3DResBlock(1024, 512)
        self.layer_2 = Basic3DResBlock(512, 256)
        self.layer_1 = Basic3DResBlock(256, 128)
        self.conv1 = nn.Conv3d(128, 128, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        
        self.layer0 = Basic3DResBlock(128, 64)
        self.layer1 = Basic3DResBlock(64, 32)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        
        self.layer2 = Basic3DResBlock(32, 16)
        self.layer3 = Basic3DResBlock(16, 8)
        self.conv3 = nn.Conv3d(8, 8, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        # self.final_conv = nn.Conv3d(8, 8, kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def forward(self, x):
        x = self.layer_4(x)
        x = self.layer_3(x)
        x = self.layer_2(x)
        x = self.layer_1(x)
        x = self.conv1(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.conv2(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv3(x)
        # x = self.final_conv(x)  # This conv layer is designed to reduce the second dimension from 8 to 4
        x = x.squeeze(2)  # This removes the singleton dimension, adapting from [bs, 8, 1, 256, 16] to [bs, 8, 256, 16]
        return x


# Model registry
model_registry = {
    "ZSum_conv_simple_Model": ZSum_conv_simple_Model,
    "ZSum_conv_bigger_Model": ZSum_conv_bigger_Model,
    "ZSum_conv_Advanced_Model": ZSum_conv_Advanced_Model,
    "ZSum_conv_big_Advanced_Model": ZSum_conv_big_Advanced_Model,
    "ZSum_conv_huge_Advanced_Model": ZSum_conv_huge_Advanced_Model,
}

class z_sum_net(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.learning_rate = config['z_sum_net_model']['base_learning_rate']
        model_type = config['z_sum_net_model']['model_type']
        self.z_sum_model = self.choose_model(model_type)

    def choose_model(self, model_type):
        try:
            model_class = model_registry[model_type]
            return model_class()
        except KeyError:
            raise ValueError(f"Unknown model type: {model_type}")


    def get_input(self, batch):
        z, _ = latent_diffusion.get_input(batch, latent_diffusion.first_stage_key)
        z_mix, _ = latent_diffusion.get_input(batch, "fbank")
        return z, z_mix

    def forward(self, z):
        return self.z_sum_model(z)        

    def configure_optimizers(self):
        # Explicitly optimize only self.vector
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


    def latent_to_waveform(self, z):
        mel = latent_diffusion.decode_first_stage(z)
        waveform = latent_diffusion.mel_spectrogram_to_waveform(mel, save=False)
        waveform = np.nan_to_num(waveform)
        waveform = np.clip(waveform, -1, 1)
        return mel.cpu().numpy(), waveform

    def training_step(self, batch, batch_idx):

        z, z_mix = self.get_input(batch)
        
        z_mix_hat = self(z)

        # compute mse loss
        loss = nn.functional.mse_loss(z_mix_hat, z_mix)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx>0 and batch_idx % 100 == 0:  # Log every 100 batches
            self.log_audio(z_mix, z_mix_hat, 'train')

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

            log_dict[f"{stage} /Mel"] = [wandb.Image(mel_mix[i].squeeze().T, caption=f"{stage} Orig {i}"),
                                         wandb.Image(mel_mix_hat[i].squeeze().T, caption=f"{stage} Reconst {i}")]

            audio_clip = waveform_mix[i][0]
            log_dict[f"{stage} /Orig Audio"] = wandb.Audio(audio_clip, sample_rate=sample_rate, caption=f"{stage} Sample {i}")

            audio_clip = waveform_mix_hat[i][0]
            log_dict[f"{stage} /Reconst Audio"] = wandb.Audio(audio_clip, sample_rate=sample_rate, caption=f"{stage} Sample {i}")

            self.logger.experiment.log(log_dict)

# Model
model = z_sum_net(config)


# Specify the path to your checkpoint
checkpoint_path = config['trainer']['resume_from_checkpoint']

latent_diffusion = latent_diffusion.to(f"cuda:{config['trainer']['devices'][0]}")

# # Train
trainer = Trainer(max_epochs=config['trainer']['max_epochs'], 
                logger=wandb_logger,
                num_sanity_val_steps=0,
                callbacks=[checkpoint_callback],  # Add other callbacks if needed
                # accelerator="cpu", 
                limit_train_batches= config['trainer']['limit_train_batches'],
                limit_val_batches=config['trainer']['limit_val_batches'],
                accelerator=config['trainer']['accelerator'], 
                devices = config['trainer']['devices'],
                log_every_n_steps=1,
                default_root_dir=run_log_dir 
                )

if config['mode'] in ["test", "validate"]:
    # Evaluation / Validation
    trainer.validate(model, data, ckpt_path=checkpoint_path)
if config['mode'] == "validate_and_train":
    # Training
    trainer.validate(model, data, ckpt_path=checkpoint_path)
    trainer.fit(model, data, ckpt_path=checkpoint_path)
elif config['mode'] == "train":
    trainer.fit(model, data, ckpt_path=checkpoint_path)
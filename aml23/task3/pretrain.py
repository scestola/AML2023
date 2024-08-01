import os
del os.environ["SLURM_NTASKS"]

import fire
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
import albumentations as A

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.environments import SLURMEnvironment
from functools import partial

from transformers import get_cosine_schedule_with_warmup
from model import MaskedAutoencoderViT
from util import *

class MAEDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__()
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        x = np.squeeze(self.data[i])
        transformed = self.transform(image=x)
        img = torch.tensor(transformed["image"] / 255, dtype=torch.float32)
        img = img.unsqueeze(0)
        return img


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model,
        lr,
        weight_decay,
        warmup_steps,
        batch_size,
        num_epochs,
        num_gpus,
        num_samples,
        debug=True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_gpus = num_gpus
        self.num_samples = num_samples
        self.debug = debug
        self.eps = 1e-7

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, _, _ = self(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self(batch)
        self.log("val/loss", loss)

    def training_step_end(self, training_step_outputs):
        (lr,) = self.scheduler.get_last_lr()
        self.log("lr", lr)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        total_steps = (
            self.num_samples * self.num_epochs // self.batch_size // self.num_gpus
        )
        print(f"Number of warm-up steps: {self.warmup_steps}/{total_steps}")
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )
        lr_scheduler_config = {
            "scheduler": self.scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


def main(
    debug=False,
    base_lr=1.5e-4,
    weight_decay=0.05,
    num_epochs=400,
    batch_size=64,
    num_gpus=1,
    img_size=224,
    patch_size=8,
    embed_dim=512,
    depth=6,
    num_heads=8,
    decoder_embed_dim=256,
    decoder_depth=4,
    decoder_num_heads=8,
    mlp_ratio=2,
):
    train_data = load_zipped_pickle("data/train.pkl")
    test_data = load_zipped_pickle("data/test.pkl")

    train_test_frames = []
    for x in train_data:
        video = np.rollaxis(x["video"], 2) # shape (H, W, L) -> (L, H, W)
        L, _, _ = video.shape
        frames = np.split(video, L, 0)
        train_test_frames.extend(frames)

    for x in test_data:
        video = np.rollaxis(x["video"], 2)
        L, _, _ = video.shape
        frames = np.split(video, L, 0)
        train_test_frames.extend(frames)
        
    print("Number of frames:", len(train_test_frames))

    rng = np.random.default_rng(42)
    rng.shuffle(train_test_frames)
    num_frames = len(train_test_frames)
    train_val_ratio = 1.0
    num_train = int(train_val_ratio * num_frames)
    train_set = train_test_frames[:num_train]
    val_set = train_test_frames[num_train:]
    print(f"Train set has {len(train_set)} frames")
    print(f"Validation set has {len(val_set)} frames")

    train_ds = MAEDataset(
        train_set,
        transform=A.Compose([
            A.RandomResizedCrop(
                height=img_size,
                width=img_size,
                scale=(0.8, 1.0),
                ratio=(1.0, 1.0),
                interpolation=cv2.INTER_LINEAR,
            ),
            A.HorizontalFlip(p=0.5),
        ])
    )

    num_samples = len(train_ds)
    lr = base_lr * batch_size / 256
    warmup_steps = int(num_samples // batch_size * num_epochs * 0.05)

    model = MaskedAutoencoderViT(
        img_size=img_size,
        in_chans=1,
        patch_size=patch_size, 
        embed_dim=embed_dim, 
        depth=depth, 
        num_heads=num_heads,
        decoder_embed_dim=decoder_embed_dim, 
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=mlp_ratio,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    print("Pretrain model has", count_parameters(model)/1e6, "M parameters")

    base_dir = "output/pretrain/"
    save_name = f"Enc{embed_dim}d{depth}h{num_heads}-Dec{decoder_embed_dim}d{decoder_depth}h{decoder_num_heads}-Mlp{mlp_ratio}"
    log_dir = os.path.join(base_dir, save_name)
    os.makedirs(log_dir, exist_ok=True)
    
    wandb_logger = WandbLogger(
        project="aml22-ethz",
        entity="kvu207",
        name=save_name,
        save_dir=log_dir,
        offline=debug,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="train/loss",
        mode="min",
        dirpath=log_dir,
        filename=save_name,
    )
    callback_list = [checkpoint_callback] if not debug else []

    pl_model = LightningModel(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        batch_size=batch_size, 
        num_gpus=num_gpus,
        num_epochs=num_epochs,
        num_samples=num_samples,
        debug=debug
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        pin_memory=True, 
        shuffle=True, 
        num_workers=4
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        deterministic=False, # True doesn't work
        precision=32, # Use fp32 for now
        detect_anomaly=debug,
        log_every_n_steps=1 if debug else 50,
        fast_dev_run=5 if debug else False,
        default_root_dir=log_dir,
        max_epochs=num_epochs,
        track_grad_norm=2,
        enable_progress_bar=True,
        logger=False if debug else wandb_logger,
        enable_checkpointing=not debug,
        callbacks=callback_list,
    )

    trainer.fit(pl_model, train_loader)

    wandb.finish()

if __name__ == "__main__":
    fire.Fire(main)

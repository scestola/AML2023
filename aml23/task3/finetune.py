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
from torchvision.transforms import InterpolationMode

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.environments import SLURMEnvironment
from functools import partial

from sklearn.metrics import jaccard_score
from torchvision.ops import sigmoid_focal_loss
import transformers
from transformers import get_cosine_schedule_with_warmup
from model import ViTBackbone, SegmentationModel, interpolate_pos_embed
from util import *


class MAEDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__()
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        frame, mask, bbox = self.data[i]
        frame = frame.astype(np.uint8)
        mask = mask.astype(np.uint8)
        bbox = bbox.astype(np.uint8)
        mask_bbox = np.stack([mask, bbox], axis=2)
        transformed = self.transform(image=frame, mask=mask_bbox)
        lbl = transformed["mask"]
        mask = lbl[:, :, 0]
        bbox = lbl[:, :, 1]
        H, W = bbox.shape
        pos = np.nonzero(bbox)
        xmin = np.min(pos[1]) / W
        xmax = np.max(pos[1]) / W
        ymin = np.min(pos[0]) / H
        ymax = np.max(pos[0]) / H
        img = torch.tensor(transformed["image"] / 255, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        bbox_pos = torch.tensor([xmin, xmax, ymin, ymax], dtype=torch.float32)
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        return img, mask, bbox_pos


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
        self.save_hyperparameters(ignore=['model'])
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
        
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, gt, bbox = batch
        mask_logit, bbox_logit = self(x)
        
        # BCE loss
        bce_loss = self.bce_loss_fn(mask_logit, gt)
        
        # Dice loss
        mask_out = torch.sigmoid(mask_logit)
        mask_out = mask_out.view(-1)
        gt = gt.view(-1)
        intersection = (mask_out * gt).sum()
        smooth = 1
        dice_loss = 1 - (2.0 * intersection + smooth) / (mask_out.sum() + gt.sum() + smooth) 
        
        # Bounding box loss
        bbox_out = torch.sigmoid(bbox_logit)
        bbox_loss = self.mse_loss_fn(bbox_out, bbox)
        
        loss = bce_loss + dice_loss + bbox_loss
        
        self.log("train/bce_loss", bce_loss)
        self.log("train/dice_loss", dice_loss)
        self.log("train/bbox_loss", bbox_loss)
        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, gt, bbox = batch
        B, _, _, _ = x.shape
        mask_logit, bbox_logit = self(x)
        
        # BCE loss
        bce_loss = self.bce_loss_fn(mask_logit, gt)
        
        # Dice loss
        mask_out = torch.sigmoid(mask_logit)
        mask_out = mask_out.view(-1)
        gt = gt.view(-1)
        intersection = (mask_out * gt).sum()
        smooth = 1
        dice_loss = 1 - (2.0 * intersection + smooth) / (mask_out.sum() + gt.sum() + smooth) 
        
        # Bounding box loss
        bbox_out = torch.sigmoid(bbox_logit)
        bbox_loss = self.mse_loss_fn(bbox_out, bbox)
        
        loss = bce_loss + dice_loss + bbox_loss
        
        # Median IoU
        batch_out = torch.sigmoid(mask_logit).view(B, -1)
        mask = batch_out > 0.5
        batch_out[mask] = 1.0
        batch_out[~mask] = 0.0
        batch_out = batch_out.cpu().numpy()
        batch_gt = gt.view(B, -1).cpu().numpy()
        iou_scores = [jaccard_score(batch_gt[i], batch_out[i]) for i in range(B)]
        median_iou = np.median(iou_scores)
        
        self.log("val/bce_loss", bce_loss)
        self.log("val/dice_loss", dice_loss)
        self.log("val/bbox_loss", bbox_loss)
        self.log("val/loss", loss)
        self.log("val/median_iou", median_iou)

    def training_step_end(self, training_step_outputs):
        lrs = self.scheduler.get_last_lr()
        self.log("train/lr", max(lrs))

    def configure_optimizers(self):
        total_steps = (
            self.num_samples
            * self.num_epochs
            // self.batch_size
            // self.num_gpus
        )
        print(f"Number of warm-up steps: {self.warmup_steps}/{total_steps}")
        
        params = list(self.named_parameters())
        is_backbone = lambda n: "vit_backbone" in n
        
        # Layer-wise learning rate decay
        backbone_param_names = [n for n, p in params if is_backbone(n)]
        backbone_param_names.reverse()
        backbone_lr = self.lr
        lr_decay = 0.75
        parameters = []
        for idx, name in enumerate(backbone_param_names):
            # print(f'{idx}: lr = {backbone_lr:.6f}, {name}')
            parameters += [
                {'params': [p for n, p in params if n == name and p.requires_grad],
                 'lr': backbone_lr,
                 "weight_decay": 0.05,
                 "betas": (0.9, 0.999)
                }
            ]
            backbone_lr *= lr_decay
        
        parameters += [
            {
                "params": [p for n, p in params if not is_backbone(n)]
            }
        ]
        
        optimizer = transformers.AdamW(
            parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
            correct_bias=False,
        )
        
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=total_steps
        )
        
        lr_scheduler_config = {
            "scheduler": self.scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


def main(
    debug=False,
    base_lr=3e-4,
    weight_decay=1e-3,
    num_epochs=100,
    batch_size=16,
    warm_up_ratio=0.05,
    num_gpus=1,
    img_size=512,
    patch_size=16,
    # embed_dim=768,
    # depth=12,
    # num_heads=12,
    # mlp_ratio=4,
    embed_dim=512,
    depth=12,
    num_heads=8,
    mlp_ratio=2,
    train_full=False,
):
    train_data = load_zipped_pickle("data/train.pkl")
    labeled_data = []

    rng = np.random.default_rng(42)
    rng.shuffle(train_data)
    for x in train_data:
        video = x["video"]
        labels = x["label"]
        for i in x["frames"]:
            labeled_data.append((video[:, :, i], labels[:, :, i], x["box"]))
            
    num_frames = len(labeled_data)
    if train_full:
        num_train = num_frames
    else:
        num_train = int(num_frames * 0.9)
        
    train_set = labeled_data[:num_train]
    val_set = labeled_data[num_train:]
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
            A.ColorJitter(
                brightness=0.2, 
                contrast=0.5, 
                saturation=0.2, 
                hue=0.2, 
                p=0.5
            ),
        ])
    )

    val_ds = MAEDataset(
        val_set,
        transform=A.Compose([
            A.Resize(
                height=img_size,
                width=img_size,
                interpolation=cv2.INTER_LINEAR,
            )
        ])
    )

    pretrain_ckpt = torch.load("output/pretrain/Enc512d12h8-Dec256d8h8-Mlp2.ckpt")
    backbone_ckpt = {}
    for k, v in pretrain_ckpt["state_dict"].items():
        new_k = k.replace("model.", "")
        
        if "mask" in new_k or "decoder" in new_k:
            continue
            
        backbone_ckpt[new_k] = v
    
    vit_backbone = ViTBackbone(
        img_size=img_size,
        in_chans=1,
        patch_size=patch_size, 
        embed_dim=embed_dim, 
        depth=depth, 
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_path=0.1,
    )

    interpolate_pos_embed(vit_backbone, backbone_ckpt)
    vit_backbone.load_state_dict(backbone_ckpt)
    model = SegmentationModel(vit_backbone)
    print("Model has", count_parameters(model)/1e6, "M parameters")

    num_samples = len(train_set)
    # lr = base_lr * batch_size / 256
    lr = base_lr
    warmup_steps = int(num_samples // batch_size * num_epochs * warm_up_ratio)

    base_dir = "output/finetune/"
    save_name = f"FT-lr{lr}-ep{num_epochs}-wd{weight_decay}"
    if train_full: save_name += "-full"
    log_dir = os.path.join(base_dir, save_name)
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        # monitor="val/loss",
        # mode="min",
        dirpath=log_dir,
        filename=save_name,
    )

    wandb_logger = WandbLogger(
        project="aml22-ethz",
        entity="kvu207",
        name=save_name,
        save_dir=log_dir,
        offline=debug,
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

    if not train_full:
        val_loader = DataLoader(
            val_ds, 
            batch_size=16, 
            pin_memory=True, 
            shuffle=False, 
            num_workers=4
        )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        deterministic=False, # True doesn't work
        precision=32, # Use fp32 for now
        detect_anomaly=debug,
        log_every_n_steps=1 if debug else 5,
        fast_dev_run=5 if debug else False,
        default_root_dir=log_dir,
        max_epochs=num_epochs,
        track_grad_norm=2,
        enable_progress_bar=True,
        logger=False if debug else wandb_logger,
        enable_checkpointing=not debug,
        callbacks=callback_list,
        # gradient_clip_val=0.05,
    )

    if train_full:
        trainer.fit(pl_model, train_loader)
    else:
        trainer.fit(pl_model, train_loader, val_loader)

    wandb.finish()

if __name__ == "__main__":
    fire.Fire(main)

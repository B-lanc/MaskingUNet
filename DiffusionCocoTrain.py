import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch

from dataset import collate, CocoDataset, CocoDatasetHdf
from models.Diffusion import CocoDiffusion
import settings

import os


def main():
    MASKING = False
    NUM_CLASSES = 80
    model = CocoDiffusion(
        timesteps=1000,
        class_rate=0.8,
        Masking=MASKING,
        unet_channels=settings.unet_channels,
        emb_channels=settings.emb_channels,
        num_classes=NUM_CLASSES,
        attention=False,
    )

    save_dir = os.path.join(settings.save_dir, settings.tag)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ds = CocoDataset(
    ds = CocoDatasetHdf(
        settings.coco_dataset_dir, "train", settings.image_size, augment=True
    )
    dataloader = DataLoader(
        ds,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=12,
        collate_fn=collate,
    )

    checkpoint_callback_last = ModelCheckpoint(
        save_top_k=3,
        monitor="step",
        mode="max",
        filename="last_check_{epoch:02d}",
    )
    checkpoint_callback_best = ModelCheckpoint(
        save_top_k=3,
        monitor="train_loss_epoch",
        mode="min",
        filename="best_check_{epoch:02d}",
    )

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=30,
        min_epochs=3,
        default_root_dir=save_dir,
        callbacks=[checkpoint_callback_last, checkpoint_callback_best],
    )

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()

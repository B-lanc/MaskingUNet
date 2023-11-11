import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch

from dataset import collate, CocoDataset, CocoDatasetHdf
from models.Segmentation import Segmentation
import settings

import os


def main():
    MASKING = False
    out_channels = 80
    model = Segmentation(
        3,
        out_channels,
        settings.unet_channels,
        depth=2,
        Masking=MASKING,
        Attention=False,
    )

    save_dir = os.path.join(settings.save_dir, settings.tag)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ds = CocoDataset(
    train_ds = CocoDatasetHdf(
        settings.coco_dataset_dir, "train", settings.image_size, augment=True
    )
    val_ds = CocoDatasetHdf(
        settings.coco_dataset_dir, "val", settings.image_size, augment=False
    )
    train_dataloader = DataLoader(
        train_ds,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=12,
        collate_fn=collate,
    )
    val_dataloader = DataLoader(
        val_ds,
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
        monitor="val_loss",
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

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    main()

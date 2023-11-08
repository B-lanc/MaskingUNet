import lightning as L
from torch.utils.data import DataLoader
import torch

from dataset import flowers_dataset, FlowersDataset, collate
from models.Diffusion import Diffusion
import settings

import os


def main():
    TAG = "Testing"
    MASKING = False
    NUM_CLASSES = 102
    model = Diffusion(
        timesteps=1000,
        class_rate=0.9,
        Masking=MASKING,
        unet_channels=settings.unet_channels,
        emb_channels=settings.emb_channels,
        num_classes=NUM_CLASSES,
    )

    save_dir = os.path.join(settings.save_dir, TAG)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ds = flowers_dataset(settings.flowers_dataset_dir, False)
    ds = FlowersDataset(ds)
    dataloader = DataLoader(
        ds,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=12,
        collate_fn=collate,
    )

    trainer = L.Trainer(
        accelerator="gpu", max_epochs=100, min_epochs=30, default_root_dir=save_dir
    )

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()

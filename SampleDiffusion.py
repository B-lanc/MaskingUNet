import torch

from models.Diffusion import Diffusion
import settings
from utils import save_image

import os


def main():
    labels = [77]
    TAG = "Testing"
    checkpoint_dir = os.path.join(
        settings.save_dir,
        TAG,
        "lightning_logs",
        "version_19",
        "checkpoints",
        "epoch=99-step=12800.ckpt",
    )

    model = (
        Diffusion.load_from_checkpoint(
            checkpoint_dir,
            timesteps=1000,
            class_rate=0.9,
            Masking=False,
            unet_channels=settings.unet_channels,
            emb_channels=settings.emb_channels,
            num_classes=102,
        )
        .to(settings.device)
        .eval()
    )

    save_dir = os.path.join("generated", TAG)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results = model.sample(10, None, 2, 1, 2).detach().cpu().numpy()
    results = results.clip(0, 1)
    for idx, res in enumerate(results):
        save_image(res, os.path.join(save_dir, f"{idx}.png"))


if __name__ == "__main__":
    main()

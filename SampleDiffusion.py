import torch

from models.Diffusion import Diffusion
import settings
from utils import save_image

import os


def main():
    labels = [i for i in range(64)]
    checkpoint_dir = os.path.join(
        settings.save_dir,
        settings.tag,
        "lightning_logs",
        "version_0",
        "checkpoints",
        "last_check_epoch=499.ckpt",
    )

    model = (
        Diffusion.load_from_checkpoint(
            checkpoint_dir,
        )
        .to(settings.device)
        .eval()
    )

    save_dir = os.path.join("generated", settings.tag)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results = model.sample(1000, labels, 64, 3, 2).detach().cpu().numpy()
    results = results.clip(0, 1)
    for idx, res in enumerate(results):
        save_image(res, os.path.join(save_dir, f"{idx}.png"))


if __name__ == "__main__":
    main()

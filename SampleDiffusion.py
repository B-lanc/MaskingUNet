import torch

from models.Diffusion import Diffusion, CocoDiffusion
import settings
from utils import save_image

import os


def main():
    DATASET = "coco"
    MODEL = CocoDiffusion if DATASET == "coco" else Diffusion
    if DATASET == "coco":
        labels = [[0 for _ in range(80)] for _ in range(64)]
        for idx, label in enumerate(labels):
            label[idx] = 1
    else:
        labels = [i for i in range(64)]
    checkpoint_dir = os.path.join(
        settings.save_dir,
        settings.tag,
        "lightning_logs",
        "version_0",
        "checkpoints",
        "last_check_epoch=29.ckpt",
    )

    model = (
        MODEL.load_from_checkpoint(
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

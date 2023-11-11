from PIL import Image
import numpy as np


def load_image(path):
    im = np.array(Image.open(path))
    if len(im.shape) == 2:
        im = np.stack([im, im, im], axis=2)
    return im.transpose(2, 0, 1) / 255


def save_image(image, path):
    image = image.transpose(1, 2, 0) * 255
    image = image.astype("uint8")
    Image.fromarray(image).save(path)

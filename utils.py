from PIL import Image
import numpy as np


def load_image(path):
    return np.array(Image.open(path)).transpose(2, 0, 1) / 255


def save_image(image, path):
    image = image.transpose(1, 2, 0) * 255
    image = image.astype("uint8")
    Image.fromarray(image).save(path)

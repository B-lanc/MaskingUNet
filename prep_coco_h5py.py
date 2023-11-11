import h5py
from dataset import CocoDataset
import settings

import os

train_ds = CocoDataset(
    settings.coco_dataset_dir, "train", settings.image_size, augment=False
)
val_ds = CocoDataset(
    settings.coco_dataset_dir, "val", settings.image_size, augment=False
)

coco_hdf = os.path.join(settings.coco_dataset_dir, "hdf")
if not os.path.exists(coco_hdf):
    os.makedirs(coco_hdf)

train_name = f"coco_train_{settings.image_size}.hdf5"
val_name = f"coco_val_{settings.image_size}.hdf5"

train_path = os.path.join(coco_hdf, train_name)
val_path = os.path.join(coco_hdf, val_name)
with h5py.File(train_path, "w") as f:
    lab = f.create_group("labels")
    mask = f.create_group("masks")
    img = f.create_group("images")

    for idx, (la, se, im) in enumerate(train_ds):
        print(idx)
        lab.create_dataset(f"{idx}", data=la.numpy())
        mask.create_dataset(f"{idx}", data=se.numpy())
        img.create_dataset(f"{idx}", data=im.numpy())

with h5py.File(val_path, "w") as f:
    lab = f.create_group("labels")
    mask = f.create_group("masks")
    img = f.create_group("images")

    for idx, (la, se, im) in enumerate(val_ds):
        print(idx)
        lab.create_dataset(f"{idx}", data=la.numpy())
        mask.create_dataset(f"{idx}", data=se.numpy())
        img.create_dataset(f"{idx}", data=im.numpy())

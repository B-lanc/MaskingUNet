import h5py
from dataset import FlowersDataset, flowers_dataset
import settings

import os

ds = flowers_dataset(settings.flowers_dataset_dir, False)
ds2 = flowers_dataset(settings.flowers_dataset_dir, True)
train_ds = ds2["train"]
val_ds = ds2["val"]

ds = FlowersDataset(ds, image_size=settings.image_size, augment=False)
train_ds = FlowersDataset(train_ds, image_size=settings.image_size, augment=False)
val_ds = FlowersDataset(val_ds, image_size=settings.image_size, augment=False)

flowers_hdf = os.path.join(settings.flowers_dataset_dir, "hdf")
if not os.path.exists(flowers_dataset):
    os.makedirs(flowers_dataset)

all_name = f"all_{settings.image_size}.hdf5"
train_name = f"train_{settings.image_size}.hdf5"
val_name = f"val_{settings.image_size}.hdf5"

all_path = os.path.join(flowers_hdf, all_name)
train_path = os.path.join(flowers_hdf, train_name)
val_path = os.path.join(flowers_hdf, val_name)

with h5py.File(all_path, "w") as f:
    lab = f.create_group("labels")
    mask = f.create_group("masks")
    img = f.create_group("images")

    for idx, (la, se, im) in enumerate(ds):
        print(idx)
        lab.create_dataset(f"{idx}", data=la.numpy())
        mask.create_dataset(f"{idx}", data=se.numpy())
        img.create_dataset(f"{idx}", data=im.numpy())

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

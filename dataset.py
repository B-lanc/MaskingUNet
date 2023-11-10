import torch
from torch.utils.data import Dataset
from utils import load_image
import torchvision.transforms as transforms
from pycocotools.coco import COCO

import numpy as np
import scipy

import os
import glob


def prep_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
        ]
    )


def augment_transform():
    return transforms.Compose(
        [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)]
    )


def flowers_dataset(path, partition=False):
    """
    path is path to dataset
    partition=False will just have the list, while True will separate into train, val, test sets
    """
    labels = os.path.join(path, "imagelabels.mat")
    segmim = os.path.join(path, "segmim")
    images = os.path.join(path, "jpg")

    labels = scipy.io.loadmat(labels)["labels"][0]
    segmim = glob.glob(os.path.join(segmim, "*"))
    segmim.sort()
    images = glob.glob(os.path.join(images, "*"))
    images.sort()

    data = [(label - 1, seg, img) for label, seg, img in zip(labels, segmim, images)]

    if partition:
        ids = os.path.join(path, "setid.mat")
        ids = scipy.io.loadmat(ids)
        train = ids["trnid"][0]
        val = ids["valid"][0]
        test = ids["tstid"][0]
        train = [data[i - 1] for i in train]
        val = [data[i - 1] for i in val]
        test = [data[i - 1] for i in test]

        data = {"train": train, "val": val, "test": test}

    return data


class FlowersDataset(Dataset):
    def __init__(self, data, image_size=64, augment=False):
        """
        data is in the form of [(label, segmentation, image), (label, segmentation, image)]
        basically the output (or one of) of flowers_dataset function
        just remember to specify if train, val, test is specified
        """
        self.data = data
        self.augment = augment
        self.prep_transforms = prep_transform(image_size)
        self.augment_transform = augment_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, seg, img = self.data[idx]
        label = torch.Tensor([label]).long()
        seg = torch.Tensor(load_image(seg))
        img = torch.Tensor(load_image(img))
        combined = torch.cat((seg, img), dim=0)
        combined = self.prep_transforms(combined)
        seg, img = combined[:3, :, :], combined[3:, :, :]

        if self.augment:
            img = self.augment_transform(img)
        return label, seg, img


class CocoDataset(Dataset):
    def __init__(self, path, partition="train", image_size=64, augment=False):
        """
        partition is either train or val
        """
        self.augment = augment
        self.image_dir = os.path.join(path, f"{partition}2017")
        annotation_file = os.path.join(
            path, "annotations", f"instances_{partition}2017.json"
        )

        self.coco = COCO(annotation_file)

        cats = self.coco.loadCats(self.coco.getCatIds())
        self.id_to_index = {dicc["id"]: idx for idx, dicc in enumerate(cats)}
        self.cats = {idx: dicc for idx, dicc in enumerate(cats)}
        self.img_ids = self.coco.getImgIds()

        self.prep_transforms = prep_transform(image_size)
        self.augment_transform = augment_transform()

    def __len__(self):
        return len(self.img_ids)

    def load_everything(self, img_idx):
        img_meta = self.coco.loadImgs(img_idx)[0]
        img = os.path.join(self.image_dir, img_meta["file_name"])
        img = load_image(img)

        n_classes = len(self.cats)
        mask = np.zeros((n_classes, img.shape[1], img.shape[2]))
        labels = [0] * n_classes

        ann_id = self.coco.getAnnIds(imgIds=img_idx)
        ann = self.coco.loadAnns(ann_id)

        for i in ann:
            idx = self.id_to_index[i["category_id"]]
            curr_mask = self.coco.annToMask(i)
            mask[idx] = np.maximum(mask[idx], curr_mask)
            labels[idx] = 1

        return labels, mask, img

    def __getitem__(self, idx):
        img_idx = self.img_ids[idx]
        labels, seg, img = self.load_everything(img_idx)

        labels = torch.Tensor(labels).long()
        seg = torch.Tensor(seg)
        img = torch.Tensor(img)
        combined = torch.cat((img, seg), dim=0)
        combined = self.prep_transforms(combined)
        img, seg = combined[:3, :, :], combined[3:, :, :]

        if self.augment:
            img = self.augment_transform(img)
        return labels, seg, img


def collate(batch):
    labels = torch.stack([b[0] for b in batch])
    segs = torch.stack([b[1] for b in batch])
    imgs = torch.stack([b[2] for b in batch])

    return labels, segs, imgs

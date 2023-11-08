import torch
from torch.utils.data import Dataset
from utils import load_image
import torchvision.transforms as transforms

import scipy

import os
import glob


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

    data = [(label-1, seg, img) for label, seg, img in zip(labels, segmim, images)]

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
        self.prep_transforms = transforms.Compose(
            [
                transforms.Resize(image_size, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
        self.augment_transform = transforms.Compose(
            [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)]
        )

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

def collate(batch):
    labels = torch.stack([b[0] for b in batch])
    segs = torch.stack([b[1] for b in batch])
    imgs = torch.stack([b[2] for b in batch])

    return labels, segs, imgs
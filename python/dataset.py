"""
dataset.py
==========
MNIST loaders. We normalize to mean=0.1307, std=0.3081 (standard MNIST stats).
This zero-centers inputs, which is friendlier to symmetric INT8 quantization
than leaving them in [0,1].
"""
from __future__ import annotations
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)


def build_transforms(train: bool):
    if train:
        # Mild augmentation — MNIST is small, heavy augmentation hurts.
        return transforms.Compose([
            transforms.RandomAffine(degrees=8, translate=(0.08, 0.08)),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])


def get_loaders(data_root: str = "./data",
                batch_size: int = 128,
                num_workers: int = 2):
    os.makedirs(data_root, exist_ok=True)
    train_ds = datasets.MNIST(data_root, train=True,  download=True,
                              transform=build_transforms(True))
    test_ds  = datasets.MNIST(data_root, train=False, download=True,
                              transform=build_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512,        shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

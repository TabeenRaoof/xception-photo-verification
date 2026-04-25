import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROCESSED_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS,
    IMAGENET_MEAN, IMAGENET_STD
)

def get_transforms():
    transform = transforms.Compose([
        transforms.Resize([IMG_SIZE, IMG_SIZE]),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, str = IMAGENET_STD)
    ])
    return transform

def get_dataset(split="train"):
    split_dir = os.path.join(PROCESSED_DIR, split)

    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}\n"
            f"Did you run step1_preparedataset.py first?"
        )
    
    dataset = datasets.ImageFolder(
        root = split_dir,
        transform = get_transforms()
    )

    return dataset


"""
dataset_loader.py — PyTorch Dataset + DataLoader for TruPhoto.

This module provides the ImageFolder-based dataset and dataloaders
that Step 2 (feature extraction) and Step 5 (Gradio demo) will use.


USAGE (imported by other scripts):
    from dataset_loader import get_dataloaders, get_transforms

"""

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import (
    PROCESSED_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS,
    IMAGENET_MEAN, IMAGENET_STD
)

def get_transforms():
    transform = transforms.Compose([
        transforms.Resize([IMG_SIZE, IMG_SIZE]),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
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

def get_imagefolder_label_mapping(dataset):
    """
    Return the mapping from ImageFolder's auto-assigned labels
    to our project's label convention.

    ImageFolder assigns labels alphabetically:
      {'AI_Generated': 0, 'Forged': 1, 'Real': 2}

    Our convention (from config.py):
      {0: 'Real', 1: 'Forged', 2: 'AI_Generated'}

    This function returns a dict to remap:
      {imagefolder_label → our_label (tf_mapping)}
    """

    class_to_idx = dataset.class_to_idx

    tf_mapping = {
        "Real": 0,
        "Forged": 1,
        "AI_Generated": 2,
    }

    remap = {}

    for class_name, if_labeled in class_to_idx.items():
        if class_name in tf_mapping:
            remap[if_labeled] = tf_mapping[class_name]
        else:
            print(f" [WARN] Unknown class folder: {class_name}")
            remap[if_labeled] = if_labeled

    return remap


def get_dataloader(split="train", shuffle=False):
    """
    Get a DataLoader for the specified split.

    We set shuffle=False by default because for feature extraction
    we just need one deterministic pass through the data. The
    training shuffle happened when we created the splits.

    Args:
        split:   "train", "val", or "test"
        shuffle: Whether to shuffle (usually False for extraction)

    Returns:
        (DataLoader, Dataset) tuple so callers have access to both
    """

    dataset = get_dataset(split)

    loader = DataLoader(
        dataset,
        batch_size = BATCH_SIZE,
        shuffle = shuffle,
        num_workers = NUM_WORKERS,
        pin_memory = torch.cuda.is_available() # to speed up transfer to GPU if available
    )

    return loader, dataset

def get_all_dataloaders():
    """
    Convenience function: return all three split loaders at once.

    Returns:
        dict with keys "train", "val", "test", each mapping to
        a (DataLoader, Dataset) tuple
    """
    loaders = {}
    for split in ["train", "val", "test"]:
        loader, dataset = get_dataloader(split, shuffle=False)
        loaders[split] = (loader, dataset)
        print(f" {split:>5}: {len(dataset):,} images, "
                f"{len(loader)} batches of {BATCH_SIZE}")
        
    return loaders
    
if __name__ == "__main__":
    # Quick test: load all splits and print stats
    print("\n" + "="*60)
    print("  TruPhoto — DataLoader Test")
    print("="*60 + "\n")

    loaders = get_all_dataloaders()

    # Show the ImageFolder label mapping vs our convention
    train_loader, train_dataset = loaders["train"]
    print(f"\n  ImageFolder class_to_idx: {train_dataset.class_to_idx}")


    # Show the ImageFolder label mapping vs our convention
    remap = get_imagefolder_label_mapping(train_dataset)
    print(f"\n  Remap to our convention: {remap}")

    # grab one batch to verfy its shape and labels
    images, labels = next(iter(train_loader))
    print(f"\n  Sample batch shape: {images.shape}")
    print(f"  Sample labels:        {labels[:8].tolist()}")
    print(f"  Devices:              {'CUDA' if torch.cuda.is_available() else 'CPU'}")


          




    



        


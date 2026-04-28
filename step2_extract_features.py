import os
import sys
import time

import numpy as np
import torch
import timm
from tqdm import tqdm

# ---------------------------------------------------------------------------
# project imports
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FEATURES_DIR, BATCH_SIZE, NUM_WORKERS,
    PRIMARY_MODEL, ABLATION_MODEL, RANDOM_SEED
)
from dataset_loader import (
    get_dataloader, get_imagefolder_label_mapping
)

# ---------------------------------------------------------------------------
# Which models to extract features from
# Comment out a line if you only want one
# ---------------------------------------------------------------------------
MODELS_TO_EXTRACT = [
    PRIMARY_MODEL,    # "xception"        → 2048-dim features
    ABLATION_MODEL,   # "mobilenetv2_100" → 1280-dim features
]


# ---------------------------------------------------------------------------
# Load a pretrained model as a feature extractor
# ---------------------------------------------------------------------------
def load_feature_extractor(model_name, device):
    """
    Load a pretrained CNN from the `timm` library with the
    classification head removed (num_classes=0).

    When num_classes=0, timm returns the output of the global
    average pooling layer instead of class logits. This gives us:
      - XceptionNet:   2048-dimensional feature vector per image
      - MobileNetV2:   1280-dimensional feature vector per image

    The model is set to eval mode and all parameters are frozen
    because we're NOT training — just extracting features.

    Args:
        model_name: timm model identifier (e.g., "xception")
        device:     torch device (cuda or cpu)

    Returns:
        model on the specified device, ready for inference
    """
    print(f"\n   Loading {model_name} from timm...")

    # pretrained=True loads ImageNet weights; num_classes=0 removes the head → feature extractor
    model = timm.create_model(model_name, pretrained=True, num_classes=0)

    # Freeze every parameter — no gradients needed
    for param in model.parameters():
        param.requires_grad = False


    # Set to evaluation mode (disables dropout, fixes batchnorm)
    model.eval()

    # move to GPU if available
    model = model.to(device)

    # Print the feature dimension so we know what we're getting
    # Do a dummy forward pass to check output shape
    dummy = torch.randn(1, 3, 299, 299).to(device)
    with torch.no_grad():
        dummy_out = model(dummy)
    feat_dim = dummy_out.shape[1]
    print(f"  Feature dimension: {feat_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,} (all frozen)")

    return model, feat_dim

# ---------------------------------------------------------------------------
# Extract features from one split
# ---------------------------------------------------------------------------
def extract_features(model, dataloader, dataset, device, label_remap):
    """
    Run every image in the dataloader through the frozen model
    and collect the feature vectors + remapped labels.

    This is the core of the script — a single forward pass through
    the entire dataset with no gradient computation.

    Args:
        model:       frozen CNN feature extractor
        dataloader:  PyTorch DataLoader for one split
        dataset:     the underlying Dataset (for label info)
        device:      torch device
        label_remap: dict mapping ImageFolder labels → our labels

    Returns:
        features:  numpy array of shape (N, feat_dim)
        labels:    numpy array of shape (N,) with our label convention
    """

    all_features = []
    all_labels   = []

    with torch.no_grad():  # No gradient computation — saves memory + speed
        for batch_images, batch_labels in tqdm(
            dataloader,
            desc="    Extracting",
            unit="batch",
            leave=False
        ):
            # Move batch to GPU if available
            batch_images = batch_images.to(device)

            # Forward pass: images → feature vectors
            # Shape: (batch_size, feat_dim) e.g., (32, 2048)
            features = model(batch_images)

            # Move back to CPU and convert to numpy
            features_np = features.cpu().numpy()
            all_features.append(features_np)

            # Remap labels from ImageFolder convention to ours
            remapped = [label_remap[lbl.item()] for lbl in batch_labels]
            all_labels.extend(remapped)

    # Stack all batches into single arrays
    features_array = np.concatenate(all_features, axis=0)
    labels_array   = np.array(all_labels, dtype=np.int64)

    return features_array, labels_array

# ---------------------------------------------------------------------------
# Save features to disk as .npy files
# ---------------------------------------------------------------------------
def save_features(features, labels, model_name, split_name, output_dir):
    """
    Save extracted features and labels as numpy .npy files.

    File naming convention:
      {model_name}_X_{split}.npy  → feature matrix  (N, feat_dim)
      {model_name}_y_{split}.npy  → label vector     (N,)

    Example:
      xception_X_train.npy   (3500, 2048)
      xception_y_train.npy   (3500,)
    """
    os.makedirs(output_dir, exist_ok=True)

    x_path = os.path.join(output_dir, f"{model_name}_X_{split_name}.npy")
    y_path = os.path.join(output_dir, f"{model_name}_y_{split_name}.npy")

    np.save(x_path, features)
    np.save(y_path, labels)

    print(f"    Saved: {x_path}  shape={features.shape}")
    print(f"    Saved: {y_path}  shape={labels.shape}")

# --------------------------------------------------------
# MAIN: Run extraction for all models × all splits
# --------------------------------------------------------
def main():
    print("\n" + "="*60)
    print("  TruPhoto — Step 2: Feature Extraction")
    print("="*60)

    # -----------------------------------------------------------------
    # Device selection
    # -----------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n  Using GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        print(f"\n  Using CPU (this will be slower — ~5-15 min per model)")
        print(f"  Tip: If you have a GPU, install the CUDA version of PyTorch")

    # -----------------------------------------------------------------
    # Set random seed for reproducibility
    # -----------------------------------------------------------------
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # -----------------------------------------------------------------
    # Load dataloaders for all three splits
    # -----------------------------------------------------------------
    print(f"\n  Loading datasets...")
    splits = ["train", "val", "test"]
    loaders = {}
    datasets_dict = {}

    for split in splits:
        loader, dataset = get_dataloader(split, shuffle=False)
        loaders[split] = loader
        datasets_dict[split] = dataset
        print(f"    {split:>5}: {len(dataset):,} images, {len(loader)} batches")

    # Get the label remapping (ImageFolder alphabetical → our convention)
    label_remap = get_imagefolder_label_mapping(datasets_dict["train"])
    print(f"\n  Label remapping: {label_remap}")
    print(f"  (ImageFolder label → Our label)")

    # -----------------------------------------------------------------
    # Extract features for each model
    # -----------------------------------------------------------------
    for model_name in MODELS_TO_EXTRACT:
        print(f"\n{'='*60}")
        print(f"  Extracting features with: {model_name}")
        print(f"{'='*60}")

        # Load the model
        model, feat_dim = load_feature_extractor(model_name, device)

        total_start = time.time()

        # Extract from each split
        for split in splits:
            print(f"\n  --- {split} split ---")
            split_start = time.time()

            features, labels = extract_features(
                model=model,
                dataloader=loaders[split],
                dataset=datasets_dict[split],
                device=device,
                label_remap=label_remap
            )

            # Quick sanity check: print class distribution
            unique, counts = np.unique(labels, return_counts=True)
            dist = {int(u): int(c) for u, c in zip(unique, counts)}
            elapsed = time.time() - split_start
            print(f"    Class distribution: {dist}")
            print(f"    Time: {elapsed:.1f}s")

            # Save to disk
            save_features(features, labels, model_name, split, FEATURES_DIR)

        total_elapsed = time.time() - total_start
        print(f"\n  {model_name} total extraction time: {total_elapsed:.1f}s")

        # Free GPU memory before loading the next model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
    # Final summary: list all saved feature files
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  FEATURE EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Saved feature files:")

    total_size = 0
    for fname in sorted(os.listdir(FEATURES_DIR)):
        fpath = os.path.join(FEATURES_DIR, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        total_size += size_mb

        # Load and print shape for verification
        arr = np.load(fpath)
        print(f"    {fname:40s}  shape={str(arr.shape):20s}  {size_mb:.1f} MB")

    print(f"\n  Total disk usage: {total_size:.1f} MB")
    print(f"  Output directory: {FEATURES_DIR}")
    print(f"\n  Next step: python src/step3_train_classifiers.py\n")


if __name__ == "__main__":
    main()

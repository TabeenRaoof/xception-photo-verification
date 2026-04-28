"""
step1_prepare_dataset.py — Dataset Preparation for TruPhoto

This script takes the raw Kaggle downloads (ArtiFact + CASIA 2.0),
discovers the directory layout, samples a balanced subset, resizes
every image to 299x299, and splits into train/val/test folders.

IMPORTANT: The ArtiFact dataset organizes images by generator/source,
with each subfolder containing a metadata.csv file that has a "target"
column (0 = real, 1 = fake/AI-generated). We read these CSV files to
correctly classify images rather than guessing from folder names.

BEFORE RUNNING:
  - Download + unzip both Kaggle datasets into data/raw/
  - Verify paths in config.py match your directory names
  - Install deps: pip install Pillow tqdm scikit-learn pandas

USAGE:
  python step1_prepare_dataset.py

Author: Tabeen
Course: CS483 Deep Learning, SFBU
"""

import os
import sys
import csv
import random
import shutil
from collections import defaultdict

import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from config import (
    ARTIFACT_DIR, CASIA2_DIR, PROCESSED_DIR, RESULTS_DIR,
    CLASS_NAMES, SAMPLES_PER_CLASS, IMG_SIZE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def find_images(directory):
    """Recursively find all image files under a directory, sorted."""
    image_paths = []
    for root, _dirs, files in os.walk(directory):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in VALID_EXTENSIONS:
                image_paths.append(os.path.join(root, file_name))

    image_paths.sort()
    return image_paths


def scan_and_report(base_dir, dataset_name):
    """Print a tree-like summary of a raw dataset folder."""
    print(f"\n{'='*60}")
    print(f"  Scanning: {dataset_name}")
    print(f"  Path:     {base_dir}")
    print(f"{'='*60}")

    if not os.path.isdir(base_dir):
        print(f"  *** DIRECTORY IS NOT FOUND - check your path in config.py ***")
        return
    
    for entry in sorted(os.listdir(base_dir)):
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        imgs = find_images(entry_path)
        meta = "   (has metadata.csv)" if os.path.isfile(
            os.path.join(entry_path, "metadata.csv")) else ""
        print(f"  └── {entry:30s}  →  {len(imgs):,} images{meta}")

def discover_artifact_paths(artifact_base):
    """
    Read metadata.csv files in ArtiFact subfolders to separate
    real images (target=0) from AI-generated images (target=1).

    Falls back to folder name matching if no metadata.csv found.

    Returns: (real_paths, ai_generated_paths)
    """
    real_paths = []
    ai_paths = []
    metadata_found = False

    if not os.path.isdir(artifact_base):
        print(f" [ERROR] ArtiFact directory not found: {artifact_base}")
        return real_paths, ai_paths

    # Strategy 1: Read metadata.csv files
    for root, dirs, files in os.walk(artifact_base):
        if "metadata.csv" not in files:
            continue
        meta_path = os.path.join(root, "metadata.csv")
        try:
            df = pd.read_csv(meta_path)

            if "target" not in df.columns:
                print(f" [warn] No 'target' column in {meta_path}")
                continue

            metadata_found = True
            folder_name = os.path.relpath(root, artifact_base)

            # Find the column that contains image file paths
            path_col = None
            for candidate in ["image_path", "filename", "path", "file", "name"]:
                if candidate in df.columns:
                    path_col = candidate
                    break

            if path_col is None:
                print(f"  [WARN] No path column in {meta_path}, skipping")
                continue

            real_count = 0
            ai_count = 0
            for _, row in df.iterrows():
                target = int(row["target"])
                img_rel_path = str(row[path_col])

                # Build full path — try relative to metadata.csv location first
                img_full = os.path.join(root, img_rel_path)
                if not os.path.isfile(img_full):
                    # Try relative to artifact_base
                    img_full = os.path.join(artifact_base, img_rel_path)
                if not os.path.isfile(img_full):
                    continue

                if target == 0:
                    real_paths.append(img_full)
                    real_count += 1
                elif target == 1:
                    ai_paths.append(img_full)
                    ai_count += 1

            print(f"  [META] {folder_name:30s} → "
                  f"{real_count:,} real, {ai_count:,} AI-generated")
            
        except Exception as e:
            print(f"  [WARN] Error reading {meta_path}: {e}")


    # Strategy 2: Fallback to folder name matching
    if not metadata_found:
        print(f"  [INFO] no metadata.csv files found. Falling back to folder name matching.")

        real_keywords = ["real", "nature", "authentic", "0_real", "real_images"]
        ai_keywords   = [
            "fake", "ai", "generated", "synthetic", "gan",
            "diffusion", "dalle", "midjourney", "stable",
            "glide", "biggan", "stylegan", "1_fake",
            "progan", "stargan", "crn", "imle", "deepfake",
            "sdv14", "sdv15", "sdv2", "vqdm", "wukong"
        ]

        entries = os.listdir(artifact_base)
        subdirs = [e for e in entries if os.path.isdir(os.path.join(artifact_base, e))]
        split_dirs = [d for d in subdirs if d.lower() in ("train", "val", "test", "validation")]

        search_dirs = ([os.path.join(artifact_base, sd) for sd in split_dirs]
                       if split_dirs else [artifact_base])
        
        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                continue

            for folder_name in sorted(os.listdir(search_dir)):
                folder_path = os.path.join(search_dir, folder_name)
                if not os.path.isdir(folder_path):
                    continue

                folder_lower = folder_name.lower()

                if any(kw in folder_lower for kw in real_keywords):
                    imgs = find_images(folder_path)
                    real_paths.extend(imgs)
                    print(f"  [REAL] {folder_name:30s} → {len(imgs):,} images")

                elif any(kw in folder_lower for kw in ai_keywords):
                    imgs = find_images(folder_path)
                    ai_paths.extend(imgs)
                    print(f"  [AI] {folder_name:30s} → {len(imgs):,} AI-generated (by name)")
                else:
                    imgs = find_images(folder_path)
                    if imgs:
                        print(f"  [???]   {folder_name:30s} → {len(imgs):,} images (UNMAPPED)")

    return real_paths, ai_paths


def discover_casia2_forged_paths(casia2_base):
    """
    Find tampered (forged) images in CASIA 2.0.

    CASIA 2.0 layout:
      <casia2_base>/Au/   → authentic images (skipped — ArtiFact supplies "real")
      <casia2_base>/Tp/   → tampered/spliced images (our "Forged" class)

    Some redistributions use 'CASIA2' or different casing; we match
    case-insensitively and fall back to keyword search if the canonical
    folders aren't present.

    Returns: list of paths to forged images
    """
    forged_paths = []

    if not os.path.isdir(casia2_base):
        print(f"  [ERROR] CASIA2 directory not found: {casia2_base}")
        return forged_paths

    tampered_keywords = ["tp", "tampered", "tamper", "forged", "fake", "spliced"]

    matched_dirs = []
    for root, dirs, _files in os.walk(casia2_base):
        for d in dirs:
            if d.lower() in tampered_keywords or any(
                kw in d.lower() for kw in ("tamper", "forg", "splic")
            ):
                matched_dirs.append(os.path.join(root, d))

    if not matched_dirs:
        print(f"  [WARN] No tampered/forged subfolder found under {casia2_base}")
        return forged_paths

    seen = set()
    for tdir in matched_dirs:
        imgs = find_images(tdir)
        new_imgs = [p for p in imgs if p not in seen]
        seen.update(new_imgs)
        forged_paths.extend(new_imgs)
        rel = os.path.relpath(tdir, casia2_base)
        print(f"  [FORGED] {rel:30s} → {len(new_imgs):,} images")

    return forged_paths


def discover_all_paths(artifact_base, casia2_base):
    """
    Single entry point that returns paths for all 3 project classes.

    Combines ArtiFact (Real, AI_Generated) and CASIA 2.0 (Forged) into
    one dict keyed by the class names defined in config.CLASS_NAMES.

    Returns:
        dict: {"Real": [...], "Forged": [...], "AI_Generated": [...]}
    """
    print("--- ArtiFact: Real + AI_Generated ---")
    real_paths, ai_paths = discover_artifact_paths(artifact_base)

    print("\n--- CASIA 2.0: Forged ---")
    forged_paths = discover_casia2_forged_paths(casia2_base)

    return {
        CLASS_NAMES[0]: real_paths,
        CLASS_NAMES[1]: forged_paths,
        CLASS_NAMES[2]: ai_paths,
    }


def balanced_sample(paths, n_samples, label_name, seed):
    """Randomly sample n_samples images, or take all if fewer available."""
    random.seed(seed)
    available = len(paths)
    if available == 0:
        print(f"  [ERROR] No images found for class '{label_name}'!")
        return []
    if available < n_samples:
        print(f"  [WARN] {label_name}: requested {n_samples:,} but only {available:,} available.")
        return paths.copy()

    sampled = random.sample(paths, n_samples)
    print(f"  [OK]   {label_name}: sampled {len(sampled):,} / {available:,}")
    return sampled


def resize_and_save(src_path, dst_path, target_size):
    """Resize image to target_size x target_size RGB JPEG."""
    try:
        img = Image.open(src_path).convert("RGB")
        img = img.resize((target_size, target_size), Image.LANCZOS)
        img.save(dst_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"  [SKIP] Could not process {src_path}: {e}")
        return False


def split_and_save(image_paths, label, class_name, metadata_rows):
    """Split into train/val/test, resize, save, and track metadata."""
    val_test_ratio = VAL_RATIO + TEST_RATIO
    test_fraction = TEST_RATIO / val_test_ratio

    train_paths, valtest_paths = train_test_split(
        image_paths, test_size=val_test_ratio, random_state=RANDOM_SEED)
    val_paths, test_paths = train_test_split(
        valtest_paths, test_size=test_fraction, random_state=RANDOM_SEED)

    print(f"\n  {class_name} split: Train={len(train_paths):,}  Val={len(val_paths):,}  Test={len(test_paths):,}")

    total_saved = total_failed = 0

    for split_name, paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
        out_dir = os.path.join(PROCESSED_DIR, split_name, class_name)
        os.makedirs(out_dir, exist_ok=True)

        for idx, src_path in enumerate(tqdm(paths, desc=f"    {split_name}/{class_name}", unit="img", leave=False)):
            dst_path = os.path.join(out_dir, f"img_{idx:05d}.jpg")
            if resize_and_save(src_path, dst_path, IMG_SIZE):
                total_saved += 1
                metadata_rows.append({
                    "original_path": src_path, "processed_path": dst_path,
                    "class_label": label, "class_name": class_name, "split": split_name
                })
            else:
                total_failed += 1

    print(f"  {class_name}: {total_saved:,} saved, {total_failed:,} failed")
    return total_saved, total_failed


def main():
    print("\n" + "="*60)
    print("  TruPhoto — Step 1: Dataset Preparation")
    print("="*60)

    print("\n[PHASE 1] Scanning raw dataset directories...\n")
    scan_and_report(ARTIFACT_DIR, "ArtiFact Dataset")
    scan_and_report(CASIA2_DIR,   "CASIA 2.0 Dataset")

    print("\n[PHASE 2] Discovering image paths by class...\n")
    paths_by_class = discover_all_paths(ARTIFACT_DIR, CASIA2_DIR)

    print(f"\n  Discovery totals:")
    for label, name in CLASS_NAMES.items():
        print(f"    {name} (Class {label}): {len(paths_by_class[name]):,}")

    if any(len(paths_by_class[name]) == 0 for name in CLASS_NAMES.values()):
        print("\n  [FATAL] One or more classes have zero images.")
        print(f"    ARTIFACT_DIR = {ARTIFACT_DIR}")
        print(f"    CASIA2_DIR   = {CASIA2_DIR}")
        sys.exit(1)

    print(f"\n[PHASE 3] Sampling {SAMPLES_PER_CLASS:,} images per class...\n")
    sampled_by_class = {
        name: balanced_sample(paths_by_class[name], SAMPLES_PER_CLASS, name, RANDOM_SEED)
        for name in CLASS_NAMES.values()
    }

    total = sum(len(v) for v in sampled_by_class.values())
    print(f"\n  Total dataset size: {total:,} images")

    print(f"\n[PHASE 4] Resizing to {IMG_SIZE}x{IMG_SIZE} and splitting...\n")
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)

    metadata_rows = []
    grand_saved = grand_failed = 0
    for label, name in CLASS_NAMES.items():
        saved, failed = split_and_save(sampled_by_class[name], label, name, metadata_rows)
        grand_saved += saved
        grand_failed += failed

    print(f"\n[PHASE 5] Saving metadata...\n")
    metadata_path = os.path.join(RESULTS_DIR, "dataset_metadata.csv")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["original_path", "processed_path", "class_label", "class_name", "split"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"  Metadata CSV saved: {metadata_path}  ({len(metadata_rows):,} rows)")

    # Final summary table
    print("\n" + "="*60)
    print("  DATASET PREPARATION COMPLETE")
    print("="*60)
    split_counts = defaultdict(lambda: defaultdict(int))
    for row in metadata_rows:
        split_counts[row["split"]][row["class_name"]] += 1

    header = f"  {'Split':<10}" + "".join(f" {name:>14}" for name in CLASS_NAMES.values()) + f" {'Total':>8}"
    print(f"\n{header}")
    print(f"  {'-' * (len(header) - 2)}")
    for split in ["train", "val", "test"]:
        c = split_counts[split]
        t = sum(c.values())
        cells = "".join(f" {c.get(name, 0):>14,}" for name in CLASS_NAMES.values())
        print(f"  {split:<10}{cells} {t:>8,}")

    print(f"\n  Next step: python src/step2_extract_features.py\n")


if __name__ == "__main__":
    main()

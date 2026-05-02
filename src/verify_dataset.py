"""
verify_dataset.py — Quick sanity check after dataset preparation.

Run this after step1_prepare_dataset.py to confirm everything
looks right before moving to feature extraction.

Checks:
  1. All expected directories exist
  2. Image counts per split per class
  3. Sample images can be opened and are 299x299
  4. No overlap between train/val/test (via filename hashing)
  5. Class balance ratios

USAGE (run from repo root):
  python -m src.verify_dataset
"""

import os
import hashlib
from collections import defaultdict
from PIL import Image

from src.config import PROCESSED_DIR, CLASS_NAMES, IMG_SIZE


def hash_file(filepath):
    """Return MD5 hash of a file to detect duplicates across splits."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("\n" + "="*60)
    print("  TruPhoto — Dataset Verification")
    print("="*60)

    splits = ["train", "val", "test"]
    classes = list(CLASS_NAMES.values())  # ["Real", "Forged", "AI-Generated"]

    all_ok = True
    counts = defaultdict(lambda: defaultdict(int))
    all_hashes = defaultdict(set)  # split → set of file hashes

    # -----------------------------------------------------------
    # Check 1: Directories exist and count images
    # -----------------------------------------------------------
    print("\n[CHECK 1] Directory structure and image counts:\n")
    for split in splits:
        for cls in classes:
            dir_path = os.path.join(PROCESSED_DIR, split, cls)
            if not os.path.isdir(dir_path):
                print(f"  MISSING: {dir_path}")
                all_ok = False
                continue

            imgs = [f for f in os.listdir(dir_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            counts[split][cls] = len(imgs)

    # Print summary table
    print(f"  {'Split':<10} {'Real':>8} {'Forged':>8} {'AI-Gen':>8} {'Total':>8}")
    print(f"  {'-'*42}")
    for split in splits:
        total = sum(counts[split].values())
        print(f"  {split:<10} {counts[split].get('Real',0):>8,} "
              f"{counts[split].get('Forged',0):>8,} "
              f"{counts[split].get('AI_Generated',0):>8,} "
              f"{total:>8,}")

    # -----------------------------------------------------------
    # Check 2: Sample images are correct size
    # -----------------------------------------------------------
    print("\n[CHECK 2] Image dimensions (sampling 3 per split/class):\n")
    for split in splits:
        for cls in classes:
            dir_path = os.path.join(PROCESSED_DIR, split, cls)
            if not os.path.isdir(dir_path):
                continue
            imgs = sorted(os.listdir(dir_path))[:3]
            for fname in imgs:
                fpath = os.path.join(dir_path, fname)
                try:
                    img = Image.open(fpath)
                    w, h = img.size
                    status = "OK" if (w == IMG_SIZE and h == IMG_SIZE) else f"BAD ({w}x{h})"
                    if status != "OK":
                        all_ok = False
                    print(f"  {split}/{cls}/{fname}: {w}x{h} [{status}]")
                except Exception as e:
                    print(f"  {split}/{cls}/{fname}: CORRUPT — {e}")
                    all_ok = False

    # -----------------------------------------------------------
    # Check 3: No data leakage between splits (hash check)
    # -----------------------------------------------------------
    print("\n[CHECK 3] Checking for data leakage between splits...")
    print("  (hashing files — this may take a minute)\n")

    for split in splits:
        for cls in classes:
            dir_path = os.path.join(PROCESSED_DIR, split, cls)
            if not os.path.isdir(dir_path):
                continue
            for fname in os.listdir(dir_path):
                fpath = os.path.join(dir_path, fname)
                h = hash_file(fpath)
                all_hashes[split].add(h)

    # Check pairwise overlap
    for i, s1 in enumerate(splits):
        for s2 in splits[i+1:]:
            overlap = all_hashes[s1] & all_hashes[s2]
            if overlap:
                print(f"  WARNING: {len(overlap)} duplicate images in {s1} ∩ {s2}!")
                all_ok = False
            else:
                print(f"  {s1} ∩ {s2}: No duplicates ✓")

    # -----------------------------------------------------------
    # Check 4: Class balance
    # -----------------------------------------------------------
    print("\n[CHECK 4] Class balance (train set):\n")
    train_total = sum(counts["train"].values())
    if train_total > 0:
        for cls in classes:
            pct = counts["train"][cls] / train_total * 100
            bar = "█" * int(pct / 2)
            print(f"  {cls:15s} {counts['train'][cls]:>6,}  ({pct:5.1f}%)  {bar}")

    # -----------------------------------------------------------
    # Final verdict
    # -----------------------------------------------------------
    print("\n" + "="*60)
    if all_ok:
        print("  ALL CHECKS PASSED ✓")
    else:
        print("  SOME CHECKS FAILED — review warnings above")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

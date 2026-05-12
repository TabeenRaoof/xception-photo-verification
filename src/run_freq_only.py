"""
run_freq_only.py — Frequency-only feature extraction

Skips CNN re-extraction (mobilenetv2_100 and xception files already on disk)
and runs only the FFT + DCT frequency pass from step2.

Saves:
  data/features/freq_x_{split}.npy           (N, 228)
  data/features/freq_y_{split}.npy           (N,)
  data/features/mobilenetv2_100_freq_x_{split}.npy  (N, 1508)  combined (1280 CNN + 228 freq)
  data/features/mobilenetv2_100_freq_y_{split}.npy  (N,)

USAGE (run from repo root):
  python -m src.run_freq_only
"""

import os
import time

import numpy as np
from tqdm import tqdm

from src.config import (
    FEATURES_DIR, ABLATION_MODEL, RANDOM_SEED,
    FREQ_MODEL_NAME, FREQ_N_FFT_BINS, FREQ_DCT_BLOCK_SIZE,
)
from src.dataset_loader import get_dataloader, get_imagefolder_label_mapping
from src.frequency_features import extract_frequency_features


def save_features(features, labels, model_name, split_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    x_path = os.path.join(output_dir, f"{model_name}_x_{split_name}.npy")
    y_path = os.path.join(output_dir, f"{model_name}_y_{split_name}.npy")
    np.save(x_path, features)
    np.save(y_path, labels)
    print(f"    Saved: {x_path}  shape={features.shape}")
    print(f"    Saved: {y_path}  shape={labels.shape}")


def main():
    print("\n" + "="*60)
    print("  TruPhoto — Frequency Feature Extraction (freq only)")
    print("="*60)

    np.random.seed(RANDOM_SEED)
    splits = ["train", "val", "test"]

    # Load datasets (needed for .samples and label remapping)
    print("\n  Loading datasets...")
    datasets_dict = {}
    for split in splits:
        _, dataset = get_dataloader(split, shuffle=False)
        datasets_dict[split] = dataset
        print(f"    {split:>5}: {len(dataset):,} images")

    label_remap = get_imagefolder_label_mapping(datasets_dict["train"])
    print(f"\n  Label remapping: {label_remap}")

    # Load existing MobileNetV2 CNN features for the combined arrays
    cnn_cache = {}
    for split in splits:
        x_path = os.path.join(FEATURES_DIR, f"{ABLATION_MODEL}_x_{split}.npy")
        y_path  = os.path.join(FEATURES_DIR, f"{ABLATION_MODEL}_y_{split}.npy")
        if os.path.isfile(x_path):
            cnn_cache[split] = (np.load(x_path), np.load(y_path))
            print(f"  Loaded CNN cache [{split}]: {cnn_cache[split][0].shape}")
        else:
            print(f"  [WARN] CNN features missing for {split} — combined file will be skipped")

    print(f"\n{'='*60}")
    print(f"  Extracting frequency features  (FFT + DCT)")
    print(f"  Output dims: {FREQ_N_FFT_BINS} FFT bins + 2 decay + "
          f"{FREQ_DCT_BLOCK_SIZE**2 - 1}*2 DCT = "
          f"{FREQ_N_FFT_BINS + 2 + (FREQ_DCT_BLOCK_SIZE**2 - 1)*2} total")
    print(f"{'='*60}")

    for split in splits:
        print(f"\n  --- {split} split ---")
        split_start = time.time()

        samples = datasets_dict[split].samples   # list of (path, imagefolder_label)

        freq_feats  = []
        freq_labels = []

        for img_path, raw_label in tqdm(
            samples,
            desc=f"    freq {split}",
            unit="img",
            leave=False,
        ):
            feat = extract_frequency_features(
                img_path,
                n_fft_bins=FREQ_N_FFT_BINS,
                dct_block_size=FREQ_DCT_BLOCK_SIZE,
            )
            freq_feats.append(feat)
            freq_labels.append(label_remap[raw_label])

        x_freq = np.stack(freq_feats, axis=0)
        y_freq = np.array(freq_labels, dtype=np.int64)

        elapsed = time.time() - split_start
        unique, counts = np.unique(y_freq, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        print(f"    Class distribution: {dist}")
        print(f"    Time: {elapsed:.1f}s  ({elapsed/len(samples)*1000:.1f} ms/img)")

        save_features(x_freq, y_freq, FREQ_MODEL_NAME, split, FEATURES_DIR)

        # Build combined CNN + frequency array
        if split in cnn_cache:
            x_cnn, y_cnn = cnn_cache[split]
            x_combined = np.concatenate([x_cnn, x_freq], axis=1)
            combined_name = f"{ABLATION_MODEL}_freq"
            save_features(x_combined, y_cnn, combined_name, split, FEATURES_DIR)
        else:
            print(f"    [SKIP] No CNN cache for {split}; combined file not written")

    print(f"\n{'='*60}")
    print(f"  FREQUENCY EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"\n  All saved files in {FEATURES_DIR}:")
    total_mb = 0
    for fname in sorted(os.listdir(FEATURES_DIR)):
        fpath = os.path.join(FEATURES_DIR, fname)
        arr   = np.load(fpath)
        mb    = os.path.getsize(fpath) / 1e6
        total_mb += mb
        print(f"    {fname:<45}  shape={str(arr.shape):<22}  {mb:.1f} MB")
    print(f"\n  Total: {total_mb:.1f} MB")
    print(f"\n  Next step: python -m src.step3_train_classifiers\n")


if __name__ == "__main__":
    main()

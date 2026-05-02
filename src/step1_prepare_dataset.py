"""
step1_prepare_dataset.py — Dataset Preparation for TruPhoto

This script takes the raw Kaggle downloads (ArtiFact + CASIA 2.0),
discovers the directory layout, samples a balanced subset, resizes
every image to 299x299, and splits into train/val/test folders.

Note: The ArtiFact dataset organizes images by generator/source,
with each subfolder containing a metadata.csv file that has a "target"
column (0 = real, 1 = fake/AI-generated). We read these CSV files to
correctly classify images rather than guessing from folder names.


USAGE (run from repo root):
  python -m src.step1_prepare_dataset

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

from src.config import (
    ARTIFACT_DIR, CASIA2_DIR, PROCESSED_DIR, RESULTS_DIR,
    CLASS_NAMES, SAMPLES_PER_CLASS, IMG_SIZE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ArtiFact folders known to contain AI-generated images. Some folders use the
# canonical target=0/1 schema (those are handled directly), but others use a
# class-index schema (e.g. big_gan/metadata.csv has target=<imagenet_class>).
# When we hit a non-binary target, we fall back to this list to decide whether
# the rows are AI. Folders explicitly using target=0 (cycle_gan, pro_gan, etc.
# in our current data) are NOT in this list — their explicit label is honored.
AI_GENERATOR_FOLDERS = {
    "big_gan", "cips", "cycle_gan", "ddpm", "denoising_diffusion_gan",
    "diffusion_gan", "face_synthetics", "gansformer", "gau_gan",
    "generative_inpainting", "glide", "lama", "latent_diffusion", "mat",
    "palette", "pro_gan", "projected_gan", "sfhq", "stable_diffusion",
    "star_gan", "stylegan1", "stylegan2", "stylegan3", "taming_transformer",
    "vq_diffusion",
}

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

    Returns:
        (real_by_source, ai_by_source) — both are dicts mapping
        the ArtiFact subfolder name (e.g. "stylegan2") to a list of
        absolute image paths from that subfolder. Grouping by source
        is what enables stratified sampling in Phase 3.
    """
    real_by_source = defaultdict(list)
    ai_by_source = defaultdict(list)
    metadata_found = False

    if not os.path.isdir(artifact_base):
        print(f" [ERROR] ArtiFact directory not found: {artifact_base}")
        return dict(real_by_source), dict(ai_by_source)

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
            # Use the top-level subfolder name as the source key,
            # so nested layouts (e.g. stylegan2/train) collapse into "stylegan2".
            source_key = folder_name.split(os.sep)[0]

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
            unmapped_count = 0
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
                    real_by_source[source_key].append(img_full)
                    real_count += 1
                elif target == 1:
                    ai_by_source[source_key].append(img_full)
                    ai_count += 1
                else:
                    # Non-binary target schema (e.g. big_gan uses target=imagenet_class).
                    # Fall back to folder name: if it's a known generator, treat as AI.
                    if source_key in AI_GENERATOR_FOLDERS:
                        ai_by_source[source_key].append(img_full)
                        ai_count += 1
                    else:
                        unmapped_count += 1

            tail = f", {unmapped_count:,} unmapped" if unmapped_count else ""
            print(f"  [META] {folder_name:30s} → "
                  f"{real_count:,} real, {ai_count:,} AI-generated{tail}")

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
                    real_by_source[folder_name].extend(imgs)
                    print(f"  [REAL] {folder_name:30s} → {len(imgs):,} images")

                elif any(kw in folder_lower for kw in ai_keywords):
                    imgs = find_images(folder_path)
                    ai_by_source[folder_name].extend(imgs)
                    print(f"  [AI] {folder_name:30s} → {len(imgs):,} AI-generated (by name)")
                else:
                    imgs = find_images(folder_path)
                    if imgs:
                        print(f"  [???]   {folder_name:30s} → {len(imgs):,} images (UNMAPPED)")

    # Defensive dedup: if a path ended up in both Real and AI (e.g. mixed
    # metadata schema), trust the explicit target=0 label and remove from AI.
    real_paths_set = set()
    for paths in real_by_source.values():
        real_paths_set.update(paths)
    removed = 0
    for src in list(ai_by_source.keys()):
        kept = [p for p in ai_by_source[src] if p not in real_paths_set]
        removed += len(ai_by_source[src]) - len(kept)
        ai_by_source[src] = kept
        if not kept:
            del ai_by_source[src]
    if removed:
        print(f"  [DEDUP] removed {removed:,} AI rows that also appear as Real (kept Real label).")

    return dict(real_by_source), dict(ai_by_source)


def discover_casia2_authentic_paths(casia2_base):
    """
    Find authentic (untouched) images in CASIA 2.0.

    These are added to the Real class to break the "scene → Forged" shortcut
    that emerges when Real is sourced exclusively from ArtiFact (mostly faces
    and curated photos) and Forged is sourced exclusively from CASIA Tp/
    (exclusively non-face scene photos). Including CASIA Au/ in Real means
    the model sees CASIA-style scene photos with both Real and Forged labels,
    forcing it to use actual tampering cues rather than dataset-source priors.

    Returns:
        dict mapping CASIA subfolder name (typically just "Au") to a list of
        absolute image paths.
    """
    authentic_by_source = defaultdict(list)

    if not os.path.isdir(casia2_base):
        print(f"  [ERROR] CASIA2 directory not found: {casia2_base}")
        return dict(authentic_by_source)

    authentic_keywords = {"au", "authentic", "original"}
    tampered_markers = ("tp", "tamper", "forg", "splic", "fake")

    matched_dirs = []
    for root, dirs, _files in os.walk(casia2_base):
        for d in dirs:
            d_lower = d.lower()
            if any(t in d_lower for t in tampered_markers):
                continue
            if d_lower in authentic_keywords or "authentic" in d_lower:
                matched_dirs.append(os.path.join(root, d))

    if not matched_dirs:
        print(f"  [WARN] No authentic subfolder found under {casia2_base}")
        return dict(authentic_by_source)

    seen = set()
    for adir in matched_dirs:
        imgs = find_images(adir)
        new_imgs = [p for p in imgs if p not in seen]
        seen.update(new_imgs)
        rel = os.path.relpath(adir, casia2_base)
        source_key = rel.split(os.sep)[0]
        authentic_by_source[source_key].extend(new_imgs)
        print(f"  [AUTH] {rel:30s} → {len(new_imgs):,} images")

    return dict(authentic_by_source)


def discover_casia2_forged_paths(casia2_base):
    """
    Find tampered (forged) images in CASIA 2.0.

    CASIA 2.0 layout:
      <casia2_base>/Au/   → authentic images (now mixed into Real, see above)
      <casia2_base>/Tp/   → tampered/spliced images (our "Forged" class)

    Some redistributions use 'CASIA2' or different casing; we match
    case-insensitively and fall back to keyword search if the canonical
    folders aren't present.

    Returns:
        dict mapping CASIA subfolder name (typically just "Tp") to a list
        of forged image paths. Same shape as discover_artifact_paths so
        stratified_sample can treat both classes uniformly.
    """
    forged_by_source = defaultdict(list)

    if not os.path.isdir(casia2_base):
        print(f"  [ERROR] CASIA2 directory not found: {casia2_base}")
        return dict(forged_by_source)

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
        return dict(forged_by_source)

    seen = set()
    for tdir in matched_dirs:
        imgs = find_images(tdir)
        new_imgs = [p for p in imgs if p not in seen]
        seen.update(new_imgs)
        rel = os.path.relpath(tdir, casia2_base)
        # Use the top-level subfolder as the source key.
        source_key = rel.split(os.sep)[0]
        forged_by_source[source_key].extend(new_imgs)
        print(f"  [FORGED] {rel:30s} → {len(new_imgs):,} images")

    return dict(forged_by_source)


def discover_all_paths(artifact_base, casia2_base):
    """
    Single entry point that returns paths for all 3 project classes,
    grouped by source folder so the sampler can stratify across them.

    Combines ArtiFact (Real + AI_Generated) and CASIA 2.0 (Real + Forged).
    CASIA Au/ is merged into Real to break the "scene -> Forged" shortcut
    caused by Forged being CASIA-only and Real being ArtiFact-only.

    Returns:
        dict: {
            "Real":         {source_folder: [paths]},  # ArtiFact + CASIA Au/
            "Forged":       {source_folder: [paths]},  # CASIA Tp/
            "AI_Generated": {source_folder: [paths]},  # ArtiFact generators
        }
    """
    print("--- ArtiFact: Real + AI_Generated ---")
    real_by_src_artifact, ai_by_src = discover_artifact_paths(artifact_base)

    print("\n--- CASIA 2.0: Forged (Tp/) ---")
    forged_by_src = discover_casia2_forged_paths(casia2_base)

    print("\n--- CASIA 2.0: Authentic (Au/) -> mixed into Real class ---")
    casia_real_by_src = discover_casia2_authentic_paths(casia2_base)

    # Merge CASIA Au sources into Real with a "casia2_" prefix so the source
    # keys remain unambiguous (and step4b can attribute per-source results).
    real_by_src = dict(real_by_src_artifact)
    for src, paths in casia_real_by_src.items():
        real_by_src[f"casia2_{src}"] = paths

    return {
        CLASS_NAMES[0]: real_by_src,
        CLASS_NAMES[1]: forged_by_src,
        CLASS_NAMES[2]: ai_by_src,
    }


def stratified_sample(paths_by_source, n_samples, label_name, seed):
    """
    Sample n_samples paths, distributing equally across source folders.

    This replaces a flat random.sample() — that approach silently
    over-represents the largest source (e.g. a single ArtiFact generator
    with 10x more images than the others), turning the "AI_Generated"
    class into a "stylegan2 only" class. Stratifying forces every source
    to contribute, which is what makes the class label semantically valid.

    Algorithm:
      1. Equal allocation: each source gets n_samples // num_sources slots.
      2. If any source has fewer images than its slot, take all it has.
      3. Redistribute the leftover slots randomly across sources that
         still have unused images.

    Args:
        paths_by_source: dict {source_folder_name: [image_paths]}
        n_samples:       target total number of samples
        label_name:      class name (for logging)
        seed:            RNG seed for determinism

    Returns:
        flat list of sampled paths (shuffled).
    """
    rng = random.Random(seed)

    sources = sorted(paths_by_source.keys())
    if not sources:
        print(f"  [ERROR] No images found for class '{label_name}'!")
        return []

    base_quota = n_samples // len(sources)

    # Phase 1: shuffle each source and split into (sampled, unused).
    sampled = {}
    unused = {}
    for src in sources:
        avail = list(paths_by_source[src])
        rng.shuffle(avail)
        take = min(base_quota, len(avail))
        sampled[src] = avail[:take]
        unused[src] = avail[take:]

    # Phase 2: redistribute the deficit across whatever's still available.
    deficit = n_samples - sum(len(v) for v in sampled.values())
    if deficit > 0:
        leftover_pool = [(src, p) for src in sources for p in unused[src]]
        rng.shuffle(leftover_pool)
        for src, p in leftover_pool[:deficit]:
            sampled[src].append(p)

    flat = [p for src in sources for p in sampled[src]]
    rng.shuffle(flat)

    # Reporting
    total_avail = sum(len(paths_by_source[s]) for s in sources)
    print(f"  [OK]   {label_name}: sampled {len(flat):,} / {total_avail:,} "
          f"(target {n_samples:,}) across {len(sources)} sources")
    for src in sources:
        avail = len(paths_by_source[src])
        taken = len(sampled[src])
        marker = "  (exhausted)" if taken == avail and avail < base_quota else ""
        print(f"           {src:<40s}  {taken:>5,} / {avail:>6,}{marker}")

    if len(flat) < n_samples:
        print(f"  [WARN] {label_name}: only {len(flat):,} available "
              f"(requested {n_samples:,}). Class will be undersized.")
    return flat


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
        sources = paths_by_class[name]
        total = sum(len(v) for v in sources.values())
        print(f"    {name} (Class {label}): {total:,} images across {len(sources)} sources")

    if any(sum(len(v) for v in paths_by_class[name].values()) == 0
           for name in CLASS_NAMES.values()):
        print("\n  [FATAL] One or more classes have zero images.")
        print(f"    ARTIFACT_DIR = {ARTIFACT_DIR}")
        print(f"    CASIA2_DIR   = {CASIA2_DIR}")
        sys.exit(1)

    print(f"\n[PHASE 3] Stratified sampling: {SAMPLES_PER_CLASS:,} images per class,")
    print(f"          equally distributed across each class's source folders.\n")
    sampled_by_class = {
        name: stratified_sample(paths_by_class[name], SAMPLES_PER_CLASS, name, RANDOM_SEED)
        for name in CLASS_NAMES.values()
    }

    total = sum(len(v) for v in sampled_by_class.values())
    print(f"\n  Total dataset size: {total:,} images")

    print(f"\n[PHASE 4] Resizing to {IMG_SIZE}x{IMG_SIZE} and splitting...\n")
    if os.path.exists(PROCESSED_DIR):
        confirm = input(
            f"  PROCESSED_DIR already exists: {PROCESSED_DIR}\n"
            f"  This will DELETE its entire contents. Continue? [y/N] "
        ).strip().lower()
        if confirm not in ("y", "yes"):
            print("  Aborted.")
            sys.exit(0)
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

    print(f"\n  Next step: python -m src.step2_extract_features\n")


if __name__ == "__main__":
    main()

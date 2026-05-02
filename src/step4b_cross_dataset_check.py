"""
step4b_cross_dataset_check.py — Cross-source generalization sanity check.

Splits the test set by SOURCE (which generator / sub-dataset each image
came from) and reports per-source accuracy. If accuracy varies wildly
across sources, the model is learning dataset signatures (JPEG tables,
sensor noise, generator fingerprints) rather than authenticity cues.

USAGE (run from repo root):
  # default: xception + random forest
  python -m src.step4b_cross_dataset_check

  # specify any combination
  python -m src.step4b_cross_dataset_check --backbone mobilenetv2_100 --classifier svm
  python -m src.step4b_cross_dataset_check --backbone xception        --classifier rf

  # evaluate every saved combination in one go
  python -m src.step4b_cross_dataset_check --all
"""

import argparse
import csv
import os
from collections import defaultdict

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

from src.config import (
    ABLATION_MODEL, ARTIFACT_DIR, CASIA2_DIR, CLASS_NAMES,
    FEATURES_DIR, MODELS_DIR, PRIMARY_MODEL, RESULTS_DIR,
)


SUPPORTED_BACKBONES = [PRIMARY_MODEL, ABLATION_MODEL]
SUPPORTED_CLASSIFIERS = ["rf", "svm"]


def infer_source_group(original_path):
    """Map an image path back to (dataset, source_subfolder)."""
    path = os.path.abspath(original_path)
    artifact = os.path.abspath(ARTIFACT_DIR)
    casia = os.path.abspath(CASIA2_DIR)

    if path.startswith(artifact):
        rel = os.path.relpath(path, artifact)
        return ("artifact", rel.split(os.sep)[0])
    if path.startswith(casia):
        rel = os.path.relpath(path, casia)
        return ("casia2", rel.split(os.sep)[0])
    return ("unknown", os.path.dirname(original_path))


def load_test_metadata_aligned():
    """Return test-split metadata rows in the exact order ImageFolder yields them."""
    meta_path = os.path.join(RESULTS_DIR, "dataset_metadata.csv")
    rows = [r for r in csv.DictReader(open(meta_path, newline="")) if r["split"] == "test"]
    # ImageFolder iterates classes alphabetically, then filenames alphabetically.
    rows.sort(key=lambda r: (r["class_name"], r["processed_path"]))
    return rows


def load_classifier(backbone, classifier):
    """
    Return (clf, scaler_or_None). SVM combinations include a fitted StandardScaler.
    Raises FileNotFoundError if the requested combination has not been trained.
    """
    if classifier == "rf":
        clf_path = os.path.join(MODELS_DIR, f"rf_{backbone}.joblib")
        if not os.path.isfile(clf_path):
            raise FileNotFoundError(f"Trained RF not found: {clf_path}")
        return joblib.load(clf_path), None

    if classifier == "svm":
        clf_path = os.path.join(MODELS_DIR, f"svm_{backbone}.joblib")
        scl_path = os.path.join(MODELS_DIR, f"scaler_{backbone}.joblib")
        if not os.path.isfile(clf_path):
            raise FileNotFoundError(f"Trained SVM not found: {clf_path}")
        if not os.path.isfile(scl_path):
            raise FileNotFoundError(f"SVM scaler not found: {scl_path}")
        return joblib.load(clf_path), joblib.load(scl_path)

    raise ValueError(f"Unknown classifier: {classifier!r}")


def per_source_accuracy(y_true, y_pred, sources):
    """Print per-source accuracy table and return per-class spread summary."""
    groups = defaultdict(list)
    for src, t, p in zip(sources, y_true, y_pred):
        groups[src].append((t, p))

    print(f"\n  {'Source':<45} {'N':>6} {'Acc':>8}  {'Dominant class':>16}")
    print(f"  {'-' * 80}")

    # Track per-class accuracies for the spread summary at the end.
    per_class_accs = defaultdict(list)

    for src, items in sorted(groups.items()):
        truths = np.array([t for t, _ in items])
        preds = np.array([p for _, p in items])
        acc = accuracy_score(truths, preds)
        labels, counts = np.unique(truths, return_counts=True)
        dominant = CLASS_NAMES[int(labels[counts.argmax()])]
        label = f"{src[0]}/{src[1]}"
        print(f"  {label:<45} {len(items):>6,} {acc:>8.4f}  {dominant:>16}")
        per_class_accs[dominant].append(acc)

    return per_class_accs


def print_spread_summary(per_class_accs):
    """Within each class, print min/max/spread across that class's sources."""
    print(f"\n  Within-class spread (decision input for fine-tuning):")
    print(f"  {'Class':<16} {'Sources':>8} {'Min':>8} {'Max':>8} {'Spread':>8}")
    print(f"  {'-' * 52}")
    for class_name, accs in sorted(per_class_accs.items()):
        if len(accs) < 2:
            print(f"  {class_name:<16} {len(accs):>8} {min(accs):>8.4f} {max(accs):>8.4f} {'n/a':>8}")
            continue
        spread_pp = (max(accs) - min(accs)) * 100
        print(f"  {class_name:<16} {len(accs):>8} {min(accs):>8.4f} {max(accs):>8.4f} {spread_pp:>7.1f}pp")


def evaluate(backbone, classifier, sources, y_test):
    """Run one (backbone, classifier) combination and print results."""
    print(f"\n{'=' * 60}")
    print(f"  Combination: {backbone} + {classifier.upper()}")
    print(f"{'=' * 60}")

    X_test = np.load(os.path.join(FEATURES_DIR, f"{backbone}_X_test.npy"))
    if len(X_test) != len(y_test):
        raise RuntimeError(
            f"Feature rows ({len(X_test)}) != y_test rows ({len(y_test)}). "
            f"Re-run step2 for backbone={backbone}."
        )

    clf, scaler = load_classifier(backbone, classifier)
    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = clf.predict(X_eval)

    per_class_accs = per_source_accuracy(y_test, y_pred, sources)
    overall = accuracy_score(y_test, y_pred)
    print(f"\n  Overall accuracy: {overall:.4f}")
    print_spread_summary(per_class_accs)


def parse_args():
    description = (__doc__ or "").split("\n")[1] if __doc__ else "Cross-source sanity check."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--backbone", choices=SUPPORTED_BACKBONES, default=PRIMARY_MODEL,
        help=f"Feature extractor (default: {PRIMARY_MODEL}).",
    )
    parser.add_argument(
        "--classifier", choices=SUPPORTED_CLASSIFIERS, default="rf",
        help="Classifier head (default: rf).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run every saved (backbone, classifier) combination and skip --backbone/--classifier.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  TruPhoto — Cross-Source Sanity Check")
    print("=" * 60)

    # The metadata + label vector are shared across all combinations.
    meta = load_test_metadata_aligned()
    # We need y_test to figure out per-source dominant class. Any backbone's y_test works
    # since labels are identical regardless of feature extractor.
    y_test_path = os.path.join(FEATURES_DIR, f"{PRIMARY_MODEL}_y_test.npy")
    if not os.path.isfile(y_test_path):
        # Fall back to the ablation model if primary's labels are missing.
        y_test_path = os.path.join(FEATURES_DIR, f"{ABLATION_MODEL}_y_test.npy")
    y_test = np.load(y_test_path)

    if len(meta) != len(y_test):
        raise RuntimeError(
            f"Metadata rows ({len(meta)}) != feature rows ({len(y_test)}).\n"
            f"Metadata is out of sync with features — re-run step1 + step2."
        )

    sources = [infer_source_group(r["original_path"]) for r in meta]

    if args.all:
        combos = [(b, c) for b in SUPPORTED_BACKBONES for c in SUPPORTED_CLASSIFIERS]
    else:
        combos = [(args.backbone, args.classifier)]

    failed = []
    for backbone, classifier in combos:
        try:
            evaluate(backbone, classifier, sources, y_test)
        except FileNotFoundError as e:
            print(f"\n  [SKIP] {backbone} + {classifier}: {e}")
            failed.append((backbone, classifier))

    print(f"\n{'=' * 60}")
    print(f"  HOW TO READ:")
    print(f"   - Within-class spread <= 5pp  -> uniform; fine-tuning is safe.")
    print(f"   - Within-class spread 5-15pp  -> borderline; fix data first.")
    print(f"   - Within-class spread > 15pp  -> shortcut learning; do NOT fine-tune.")
    print(f"   - Any single source < 50% acc -> red flag regardless of spread.")

    if failed:
        print(f"\n  Skipped (artifacts not found): {failed}")


if __name__ == "__main__":
    main()

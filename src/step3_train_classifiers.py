"""
step3_train_classifiers.py — Classifier Training for TruPhoto

Trains Random Forest and SVM classifiers on the feature vectors
extracted in Step 2. Runs the full 2x2 ablation matrix:
  - XceptionNet + Random Forest  (primary)
  - XceptionNet + SVM
  - MobileNetV2 + Random Forest
  - MobileNetV2 + SVM

Saves trained models and scalers as .joblib files.

BEFORE RUNNING:
  - Complete step2_extract_features.py first
  - Install: pip install scikit-learn numpy joblib

USAGE (run from repo root):
  python -m src.step3_train_classifiers
"""

import json
import os
import subprocess
import time

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def train_logreg(x_train, y_train, x_val, y_val):
    print("\n    Scaling + training Logistic Regression...")
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)

    clf = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    clf.fit(x_train_s, y_train)
    val_acc = accuracy_score(y_val, clf.predict(x_val_s))
    return clf, scaler, val_acc

from src.config import (
    FEATURES_DIR, MODELS_DIR, CLASS_NAMES,
    PRIMARY_MODEL, ABLATION_MODEL, RANDOM_SEED,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT, RF_CLASS_WEIGHT,
    SVM_KERNEL, SVM_C, SVM_GAMMA,
    FREQ_MODEL_NAME,
)

def _git_sha():
    """Return short git SHA, or 'unknown' if not in a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def save_meta(model_path, **fields):
    """Write a sibling .meta.json next to a saved .joblib model."""
    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": _git_sha(),
        **fields,
    }
    meta_path = model_path.replace(".joblib", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return meta_path


def load_features(model_name, split):
    """Load feature array and label array for a given model and split."""
    x_path = os.path.join(FEATURES_DIR, f"{model_name}_x_{split}.npy")
    y_path = os.path.join(FEATURES_DIR, f"{model_name}_y_{split}.npy")

    if not os.path.isfile(x_path):
        raise FileNotFoundError(
            f"Feature file was not found: {x_path}\n"
            f"Did you run step2_extract_features.py first?"
        )

    x = np.load(x_path)
    y = np.load(y_path)

    return x, y


def load_feature_set(descriptor, split):
    """
    Load (and optionally concatenate) features for one entry in FEATURE_SETS.

    Each descriptor is a dict with:
      "cnn"         — CNN model name string (e.g. "mobilenetv2_100"), or None
      "freq"        — True to include frequency features, False to skip
      "save_prefix" — used for model file naming (matches the feature-set label)
      "label"       — human-readable name for printing

    Returns (x, y) where x is (N, features) and y is (N,).
    """
    arrays = []
    y = None

    # CNN features (may be None for frequency-only sets)
    if descriptor.get("cnn"):
        x_cnn, y = load_features(descriptor["cnn"], split)
        arrays.append(x_cnn)

    # Frequency features
    if descriptor.get("freq"):
        freq_x_path = os.path.join(FEATURES_DIR, f"{FREQ_MODEL_NAME}_x_{split}.npy")
        freq_y_path = os.path.join(FEATURES_DIR, f"{FREQ_MODEL_NAME}_y_{split}.npy")

        if not os.path.isfile(freq_x_path):
            raise FileNotFoundError(
                f"Frequency feature file not found: {freq_x_path}\n"
                f"Did you run step2_extract_features.py (it now also extracts freq features)?"
            )

        x_freq = np.load(freq_x_path)

        # y comes from frequency file if there is no CNN component
        if y is None:
            y = np.load(freq_y_path)

        arrays.append(x_freq)

    # Stack all feature blocks side-by-side into one matrix
    x = np.concatenate(arrays, axis=1)

    return x, y


# ---------------------------------------------------------------------------
# Feature sets for the ablation study
# ---------------------------------------------------------------------------
# Each entry defines one row in the final ablation table.
# "save_prefix" drives the output model file names:
#   rf_{save_prefix}.joblib, svm_{save_prefix}.joblib, etc.
# ---------------------------------------------------------------------------
FEATURE_SETS = [
    {
        "label":       f"{ABLATION_MODEL} — CNN only (baseline)",
        "cnn":         ABLATION_MODEL,
        "freq":        False,
        "save_prefix": ABLATION_MODEL,
    },
    {
        "label":       "freq — frequency only (FFT + DCT)",
        "cnn":         None,
        "freq":        True,
        "save_prefix": FREQ_MODEL_NAME,
    },
    {
        "label":       f"{ABLATION_MODEL} + freq — combined",
        "cnn":         ABLATION_MODEL,
        "freq":        True,
        "save_prefix": f"{ABLATION_MODEL}_freq",
    },
]

def train_random_forest(x_train, y_train, x_val, y_val):
    """Train a Random Forest classifier and return validation accuracy."""

    print(f"\n    Training Random Forest (n_estimators={RF_N_ESTIMATORS})...")

    start = time.time()

    rf = RandomForestClassifier(
        n_estimators = RF_N_ESTIMATORS,
        max_depth = RF_MAX_DEPTH,
        min_samples_split = RF_MIN_SAMPLES_SPLIT,
        class_weight = RF_CLASS_WEIGHT,
        random_state = RANDOM_SEED,
        n_jobs = -1 # use all CPU cores
    )

    rf.fit(x_train, y_train)

    train_acc = accuracy_score(y_train, rf.predict(x_train))
    val_acc = accuracy_score(y_val, rf.predict(x_val))
    elapsed = time.time() - start

    print(f"    Train accuracy: {train_acc:.4f}")
    print(f"    Val accuracy:   {val_acc:.4f}")
    print(f"    Time: {elapsed:.1f}s")

    return rf, val_acc

def train_svm(x_train, y_train, x_val, y_val):
    """
    Train an SVM classifier with feature scaling.
    Returns the trained SVM and the fitted scaler (both needed for inference).
    """
    print(f"\n    Scaling features for SVM...")
    # Fit scaler on training data ONLY — then transform both train and val

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    print(f"    Training SVM (kernel={SVM_KERNEL}, C={SVM_C})...")

    start = time.time()

    svm = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        random_state=RANDOM_SEED,
        probability=True # to enable predict_proba for Gradio demo
    )

    svm.fit(x_train_scaled, y_train)

    train_acc = accuracy_score(y_train, svm.predict(x_train_scaled))
    val_acc = accuracy_score(y_val, svm.predict(x_val_scaled))
    elapsed = time.time() - start

    print(f"    Train accuracy: {train_acc:.4f}")
    print(f"    Val accuracy:   {val_acc:.4f}")
    print(f"    Time: {elapsed:.1f}s")

    return svm, scaler, val_acc



def main():
    print("\n" + "="*60)
    print("  TruPhoto — Step 3: Train Classifiers")
    print("="*60)

    os.makedirs(MODELS_DIR, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    # Track results for the ablation summary table
    results = []

    # Run the ablation across all three feature sets:
    #   1. CNN only (MobileNetV2 baseline)
    #   2. Frequency only (FFT + DCT, content-invariant)
    #   3. Combined (CNN + frequency — the primary hypothesis)
    # Each feature set trains both RF and SVM classifiers.
    # Note: SVM on the combined 1508-d features may take ~45-60 min on CPU.
    for fset in FEATURE_SETS:
        print(f"\n{'='*60}")
        print(f"  Feature set: {fset['label']}")
        print(f"{'='*60}")

        # Load (and optionally concatenate) training and validation features
        try:
            x_train, y_train = load_feature_set(fset, "train")
            x_val,   y_val   = load_feature_set(fset, "val")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        save_prefix = fset["save_prefix"]

        print(f"  Train: {x_train.shape[0]:,} samples, {x_train.shape[1]} features")
        print(f"  Val:   {x_val.shape[0]:,} samples")

        # --- Random Forest ---
        print(f"\n  --- Random Forest ---")
        rf, rf_val_acc = train_random_forest(x_train, y_train, x_val, y_val)

        rf_path = os.path.join(MODELS_DIR, f"rf_{save_prefix}.joblib")
        joblib.dump(rf, rf_path)
        save_meta(
            rf_path,
            classifier="random_forest",
            feature_set=fset["label"],
            val_accuracy=rf_val_acc,
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            class_weight=RF_CLASS_WEIGHT,
            random_seed=RANDOM_SEED,
        )
        print(f"    Saved: {rf_path}")

        results.append((fset["label"], "Random Forest", rf_val_acc))

        # --- SVM ---
        print(f"\n  --- SVM ---")
        svm, scaler, svm_val_acc = train_svm(x_train, y_train, x_val, y_val)

        svm_path    = os.path.join(MODELS_DIR, f"svm_{save_prefix}.joblib")
        scaler_path = os.path.join(MODELS_DIR, f"scaler_{save_prefix}.joblib")
        joblib.dump(svm, svm_path)
        joblib.dump(scaler, scaler_path)
        save_meta(
            svm_path,
            classifier="svm",
            feature_set=fset["label"],
            val_accuracy=svm_val_acc,
            kernel=SVM_KERNEL,
            C=SVM_C,
            gamma=SVM_GAMMA,
            scaler_path=scaler_path,
            random_seed=RANDOM_SEED,
        )
        print(f"    Saved: {svm_path}")
        print(f"    Saved: {scaler_path}")

        results.append((fset["label"], "SVM", svm_val_acc))

    # Print ablation summary
    print(f"\n{'='*60}")
    print(f"  ABLATION RESULTS (Validation Accuracy)")
    print(f"{'='*60}")
    print(f"\n  {'Feature Set':<45} {'Classifier':<18} {'Val Acc':>10}")
    print(f"  {'-'*73}")
    for feature_label, clf_name, val_acc in results:
        print(f"  {feature_label:<45} {clf_name:<18} {val_acc:>9.4f}")

    # Identify best combination
    if results:
        best = max(results, key=lambda x: x[2])
        print(f"\n  Best combination: {best[0]} + {best[1]} ({best[2]:.4f})")

    print(f"\n  Models saved to: {MODELS_DIR}")
    print(f"\n  Next step: python -m src.step4_evaluate\n")


if __name__ == "__main__":
    main()
            
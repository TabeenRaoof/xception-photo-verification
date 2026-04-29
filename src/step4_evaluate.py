"""
step4_evaluate.py — Evaluation for TruPhoto

Evaluates all trained classifiers on the TEST set and generates:
  1. Per-class precision, recall, F1 score
  2. Overall accuracy
  3. Confusion matrices (saved as PNG)
  4. Classification reports (saved as TXT)
  5. Ablation comparison bar chart

BEFORE RUNNING:
  - Complete step3_train_classifiers.py first
  - Install: pip install scikit-learn numpy matplotlib seaborn joblib

USAGE:
  python src/step4_evaluate.py

"""

import os
import sys

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODELS_DIR, RESULTS_DIR, CLASS_NAMES,
    PRIMARY_MODEL, ABLATION_MODEL, FEATURES_DIR
)

LABEL_NAMES = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]


def load_features(model_name, split):
    """Load feature array and label array."""

    return (
        np.load(os.path.join(FEATURES_DIR, f"{model_name}_X_{split}.npy"))
        np.load(os.path.join(FEATURES_DIR, f"{model_name}_y_{split}.npy"))
    )

def evaluate_model(clf, x_test, y_test, model_name, clf_name, scaler=None)
    """
    Evaluate a classifier and return results dict.
    If scaler is provided (for SVM), scale the features first.
    """

    if scaler is not None:
        x_test = scaler.transform(x_test)

    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=LABEL_NAMES)
    CM = confusion_matrix(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )

    return {
        "model_name": model_name,
        "clf_name": clf_name,
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    }

def plot_confusion_matrix(cm, title, save_path):
    """Save a confusion matrix heatmap as PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",xticklables=LABEL_NAMES, yticklabels=LABEL_NAMES,
        ax=ax, cbar_kws={"shrink": 0.8}

    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {save_path}")


def plot_ablation_comparison(all_results, save_path):
    """Create a grouped bar chart comparing all model combinations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [f"{r['model_name']}\n+ {r['clf_name']}" for r in all_results]

    accuracies = [r["accuracy"] * 100 for r in all_results]

    colors= ["#2E75B6", "#4BACC6", "#F79646", "#9BBB59"]
    bars = ax.bar(range(len(labels)), accuracies, color=colors[:len(labels)],
                    edgecolor="white", linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy %", fontsize = 10)
    ax.set_title("TruPhoto Ablation Study - Test Set Accuracy", fontsize=14, fontweight="bold")

    # Add baseline reference line
    ax.axhline(y=92.23, color="red", linestyle="--", linewidth=1, alpha=7)
    ax.text(len(labels)-.5, 92.5, "Ali et al.baseline (92.23%)")

    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.close(fig)

    print(f"    Saved: {save_path}")


def plot_per_class_metrics(all_results, save_path):
    """Bar chart showing per-class F1 scores for each model combination."""
    fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 5), sharey=True)

    if len(all_results) == 1:
        axes = [axes]  # Ensure axes is always a list for consistent indexing

    colors = ["#2E75B6", "#F79646", "#9BBB59"]

    for idx, (ax, r) in enumerate(zip(axes, all_results)):
        bars = ax.bar(LABEL_NAMES, r["f1"] * 100, color=colors, edgecolor="white")
        for bar, f1_val in zip(bars, r["f1"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{f1_val*100:.1f}%", ha="center", va="bottom", fontsize=9)
            
        ax.set_title(f"{r['model_name']} + {r['clf_name']}", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.set_ylabel("F1 Score (%)" if idx == 0 else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Per-Class F1 Scores", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {save_path}")


def main():
    print("\n" + "="*60)
    print("  TruPhoto — Step 4: Evaluation")
    print("="*60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []

    for model_name in [PRIMARY_MODEL, ABLATION_MODEL]:
        # Load test features
        try:
            X_test, y_test = load_features(model_name, "test")
        except FileNotFoundError:
            print(f"  [SKIP] No test features for {model_name}")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_name}")
        print(f"  Test samples: {X_test.shape[0]:,}")
        print(f"{'='*60}")

        # --- Random Forest ---
        rf_path = os.path.join(MODELS_DIR, f"rf_{model_name}.joblib")
        if os.path.isfile(rf_path):
            print(f"\n  --- Random Forest ---")
            rf = joblib.load(rf_path)
            result = evaluate_model(rf, X_test, y_test, model_name, "Random Forest")
            all_results.append(result)

            print(f"    Overall accuracy: {result['accuracy']:.4f}")
            print(f"\n{result['report']}")

            plot_confusion_matrix(
                result["confusion_matrix"],
                f"Confusion Matrix: {model_name} + Random Forest",
                os.path.join(RESULTS_DIR, f"cm_{model_name}_rf.png")
            )

        # --- SVM ---
        svm_path    = os.path.join(MODELS_DIR, f"svm_{model_name}.joblib")
        scaler_path = os.path.join(MODELS_DIR, f"scaler_{model_name}.joblib")
        if os.path.isfile(svm_path) and os.path.isfile(scaler_path):
            print(f"\n  --- SVM ---")
            svm    = joblib.load(svm_path)
            scaler = joblib.load(scaler_path)
            result = evaluate_model(svm, X_test, y_test, model_name, "SVM", scaler)
            all_results.append(result)

            print(f"    Overall accuracy: {result['accuracy']:.4f}")
            print(f"\n{result['report']}")

            plot_confusion_matrix(
                result["confusion_matrix"],
                f"Confusion Matrix: {model_name} + SVM",
                os.path.join(RESULTS_DIR, f"cm_{model_name}_svm.png")
            )

    # Generate comparison plots
    if len(all_results) >= 2:
        print(f"\n  --- Generating comparison plots ---")
        plot_ablation_comparison(
            all_results,
            os.path.join(RESULTS_DIR, "ablation_comparison.png")
        )
        plot_per_class_metrics(
            all_results,
            os.path.join(RESULTS_DIR, "per_class_f1.png")
        )

    # Save all reports to a single text file
    report_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("TruPhoto — Evaluation Report\n")
        f.write("=" * 60 + "\n\n")

        f.write("Comparison baseline: Ali et al. (2022) = 92.23% binary accuracy on CASIA 2.0\n\n")

        for r in all_results:
            f.write(f"{'='*60}\n")
            f.write(f"{r['model_name']} + {r['clf_name']}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Overall accuracy: {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%)\n\n")
            f.write(r["report"])
            f.write(f"\nConfusion Matrix:\n{r['confusion_matrix']}\n\n")

        # Summary table
        f.write("\n" + "="*60 + "\n")
        f.write("ABLATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"{'Feature Extractor':<25} {'Classifier':<18} {'Accuracy':>10}\n")
        f.write(f"{'-'*53}\n")
        for r in all_results:
            f.write(f"{r['model_name']:<25} {r['clf_name']:<18} {r['accuracy']:>9.4f}\n")

    print(f"\n    Saved: {report_path}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\n  {'Feature Extractor':<25} {'Classifier':<18} {'Test Acc':>10}")
    print(f"  {'-'*53}")
    for r in all_results:
        print(f"  {r['model_name']:<25} {r['clf_name']:<18} {r['accuracy']:>9.4f}")

    if all_results:
        best = max(all_results, key=lambda x: x["accuracy"])
        print(f"\n  Best: {best['model_name']} + {best['clf_name']} ({best['accuracy']:.4f})")

    print(f"\n  Results saved to: {RESULTS_DIR}")
    print(f"\n  Next step: python src/step5_gradio_demo.py\n")


if __name__ == "__main__":
    main()


        
            

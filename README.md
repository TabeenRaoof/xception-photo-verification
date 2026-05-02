# TruPhoto

Image authenticity classifier that labels an image as **Real**, **Forged**, or
**AI-Generated**.

The pipeline uses a frozen pretrained CNN (Xception by default, MobileNetV2
for ablation) as a feature extractor, then trains lightweight scikit-learn
classifiers (Random Forest, SVM) on the resulting feature vectors. A Gradio
web demo serves the trained model.

CS483 Deep Learning — San Francisco Bay University.

---

## Project layout

```
photo_verification_with_xception/
├── src/                    # Pipeline scripts (importable as `src.<module>`)
│   ├── config.py           # All paths, hyperparameters, constants
│   ├── dataset_loader.py   # PyTorch Dataset / DataLoader helpers
│   ├── step1_prepare_dataset.py
│   ├── step2_extract_features.py
│   ├── step3_train_classifiers.py
│   ├── step4_evaluate.py
│   ├── step4b_cross_dataset_check.py
│   ├── step5_gradio_demo.py
│   └── verify_dataset.py
├── data/
│   ├── raw/                # raw datasets (you place these)
│   │   ├── artifact/       # ArtiFact dataset
│   │   └── casia2/         # CASIA 2.0 dataset
│   ├── processed/          # 299x299 JPEGs split by class/split (auto-created)
│   └── features/           # .npy feature arrays (auto-created)
├── models/                 # trained .joblib classifiers + .meta.json manifests (auto-created)
├── results/                # confusion matrices, plots, evaluation_report.txt (auto-created)
├── pyproject.toml
└── requirements.txt
```

---

## Prerequisites

- Python 3.9 or newer
- ~5 GB free disk for the datasets and intermediate artifacts
- A GPU is optional but speeds up feature extraction by ~10x

---

## Setup

All commands below are run from the **repo root**.

### 1. Create and activate a virtual environment

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (cmd.exe):**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If PowerShell rejects the activation script with an execution-policy error,
> run this once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

After activation, `python` resolves to the venv's interpreter on every
platform — every command that follows is identical on macOS, Linux, and
Windows.

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download the datasets

Both datasets are on Kaggle. Download and unzip into `data/raw/` so the
structure looks like:

```
photo_verification_with_xception/
└── data/
    └── raw/
        ├── artifact/    # from https://www.kaggle.com/datasets/awsaf49/artifact-dataset
        └── casia2/      # from https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset
```

CASIA 2.0 should contain `Au/` (authentic), `Tp/` (tampered), and
`CASIA 2 Groundtruth/` (only `Tp/` is used by the current pipeline).
ArtiFact should contain a generator subfolder per source (`stylegan2/`,
`stable_diffusion/`, etc.), each with its own `metadata.csv`.

If you put the datasets elsewhere, update `ARTIFACT_DIR` and `CASIA2_DIR`
in [`src/config.py`](src/config.py).

You do **not** need to create `data/processed/`, `data/features/`,
`models/`, or `results/` — they are created automatically the first time
each step runs.

---

## Running the pipeline

Run each step in order from the repo root. Every step prints the next
command to run when it finishes.

### Step 1 — Prepare dataset

```bash
python -m src.step1_prepare_dataset
```

Discovers images by class (using ArtiFact's `metadata.csv` and CASIA's
tampered folders), samples a balanced subset (5,000 per class by default),
resizes everything to 299x299 JPEG, and writes a 70/15/15 train/val/test
split to `data/processed/`. A `dataset_metadata.csv` is written to
`results/`.

### Step 1b — Verify (optional)

```bash
python -m src.verify_dataset
```

Sanity-checks counts per split/class, image dimensions, data leakage
across splits (via MD5), and class balance.

### Step 2 — Extract features

```bash
python -m src.step2_extract_features
```

Runs every processed image through frozen Xception **and** MobileNetV2
(both pretrained on ImageNet via `timm`), saving 2048-d / 1280-d feature
vectors to `data/features/` as `<model>_X_<split>.npy` and `<model>_y_<split>.npy`.

### Step 3 — Train classifiers

```bash
python -m src.step3_train_classifiers
```

Trains Random Forest and SVM on each feature set (4 combinations total).
Saved artifacts in `models/`:

- `rf_<model>.joblib` — Random Forest
- `svm_<model>.joblib` + `scaler_<model>.joblib` — SVM with its scaler
- `<artifact>.meta.json` — manifest with hyperparams, val accuracy, git SHA, timestamp

### Step 4 — Evaluate

```bash
python -m src.step4_evaluate
```

Evaluates each trained classifier on the test set. Outputs in `results/`:

- `cm_<model>_<clf>.png` — confusion matrices
- `ablation_comparison.png` — bar chart across all combinations
- `per_class_f1.png` — per-class F1 scores
- `evaluation_report.txt` — full classification report + ablation summary

### Step 4b — Cross-source sanity check (recommended)

```bash
python -m src.step4b_cross_dataset_check
```

Reports per-source accuracy on the test set (broken down by ArtiFact
generator subfolder and CASIA folder). Wide accuracy gaps between
sources indicate the model is learning dataset signatures (JPEG
quantization, sensor noise, generator fingerprints) rather than
authenticity cues.

### Step 5 — Gradio demo

```bash
python -m src.step5_gradio_demo                # localhost only (default)
python -m src.step5_gradio_demo --public       # also accept LAN connections
python -m src.step5_gradio_demo --share        # public Gradio share URL
```

Launches at <http://127.0.0.1:7860>. Upload an image and see the
predicted class along with per-class confidences.

---

## Configuration

All knobs live in [`src/config.py`](src/config.py):

| Knob | Default | Effect |
|---|---|---|
| `SAMPLES_PER_CLASS` | 5000 | Balanced per-class sample size |
| `IMG_SIZE` | 299 | Input resolution (Xception expects 299) |
| `BATCH_SIZE` | 32 | Feature-extraction batch size |
| `RF_N_ESTIMATORS` | 500 | Random Forest tree count |
| `RANDOM_SEED` | 42 | Deterministic splits and training |
| `PRIMARY_MODEL` | `xception` | Main feature extractor |
| `ABLATION_MODEL` | `mobilenetv2_100` | Comparison feature extractor |

---

## Known limitations

- "Real" comes only from ArtiFact and "Forged" only from CASIA. The
  classifier may exploit dataset-specific signatures rather than learn
  forgery cues. Run **Step 4b** to check this and consider mixing CASIA's
  authentic images into the Real class for a more honest experiment.
- Frozen ImageNet features are not specialized for forgery detection.
  Fine-tuning the last Xception block yields a meaningful accuracy lift
  but is not part of the default pipeline.
- Random Forest probabilities are uncalibrated; the demo's confidence
  bars are indicative but not literal probabilities.

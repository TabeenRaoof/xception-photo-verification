"""
step5_gradio_demo.py — Interactive Demo for TruPhoto

Launches a Gradio web interface where users can upload an image
and get a classification result: Real, Forged, or AI-Generated,
along with confidence scores for each class.

PIPELINE (default — best ablation result):
  Upload image -> Resize to 299x299
  -> MobileNetV2 CNN features (1280-d, frozen)     [semantic path]
  -> FFT + DCT frequency features (228-d)           [artifact path]
  -> Concatenate (1508-d) -> StandardScaler -> SVM (RBF)
  -> Class label + per-class confidence

USAGE (run from repo root):
  python -m src.step5_gradio_demo                                    # default: mobilenetv2_100_freq + SVM
  python -m src.step5_gradio_demo --backbone mobilenetv2_100        # CNN-only
  python -m src.step5_gradio_demo --backbone xception               # Xception CNN-only
  python -m src.step5_gradio_demo --backbone freq                   # frequency-only
  python -m src.step5_gradio_demo --classifier rf                   # use Random Forest
  python -m src.step5_gradio_demo --public                          # accept LAN connections
  python -m src.step5_gradio_demo --share                           # public Gradio URL

  Default URL: http://127.0.0.1:7860
"""

import argparse
import os
from typing import Optional, Any

import numpy as np
import torch
import timm
import joblib
import gradio as gr
from PIL import Image
from torchvision import transforms


from src.config import (
    MODELS_DIR, CLASS_NAMES, PRIMARY_MODEL, ABLATION_MODEL,
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    FREQ_MODEL_NAME, FREQ_N_FFT_BINS, FREQ_DCT_BLOCK_SIZE,
)
from src.frequency_features import extract_frequency_features_from_array

# Backbones that include the frequency feature path
FREQ_BACKBONES = {FREQ_MODEL_NAME, f"{ABLATION_MODEL}_freq"}

# The CNN backbone underlying each supported option
CNN_BACKBONE_MAP = {
    PRIMARY_MODEL:              PRIMARY_MODEL,
    ABLATION_MODEL:             ABLATION_MODEL,
    FREQ_MODEL_NAME:            None,                      # no CNN path
    f"{ABLATION_MODEL}_freq":   ABLATION_MODEL,            # CNN + freq
}


# -------------------------------------------------------------------
# Pipeline loading
# -------------------------------------------------------------------
def load_pipeline(backbone: str, classifier_name: str):
    """
    Load the full inference pipeline for the given backbone + classifier.

    backbone options:
      "xception"               — Xception CNN features only (2048-d)
      "mobilenetv2_100"        — MobileNetV2 CNN features only (1280-d)
      "freq"                   — FFT + DCT frequency features only (228-d)
      "mobilenetv2_100_freq"   — MobileNetV2 CNN + frequency features (1508-d)

    Returns (cnn_model_or_None, classifier, scaler_or_None, transform, device).
    cnn_model_or_None is None for the freq-only backbone.
    """
    print(" Loading TruPhoto demo pipeline...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    # --- CNN feature extractor (None for freq-only) ---
    cnn_backbone = CNN_BACKBONE_MAP[backbone]
    if cnn_backbone is not None:
        print(f"    Loading CNN: {cnn_backbone} from timm...")
        cnn_model = timm.create_model(cnn_backbone, pretrained=True, num_classes=0)
        cnn_model.eval()
        for param in cnn_model.parameters():
            param.requires_grad = False
        cnn_model = cnn_model.to(device)
    else:
        print(f"    Backbone: frequency-only (no CNN)")
        cnn_model = None

    # --- Classifier (and scaler if SVM) ---
    if classifier_name == "rf":
        clf_path = os.path.join(MODELS_DIR, f"rf_{backbone}.joblib")
        if not os.path.isfile(clf_path):
            raise FileNotFoundError(f"Trained RF not found: {clf_path}\nRun step3 first.")
        print(f"    Loading Random Forest from {clf_path}...")
        classifier = joblib.load(clf_path)
        scaler = None

    elif classifier_name == "svm":
        clf_path = os.path.join(MODELS_DIR, f"svm_{backbone}.joblib")
        scl_path = os.path.join(MODELS_DIR, f"scaler_{backbone}.joblib")
        if not os.path.isfile(clf_path):
            raise FileNotFoundError(f"Trained SVM not found: {clf_path}\nRun step3 first.")
        if not os.path.isfile(scl_path):
            raise FileNotFoundError(f"SVM scaler not found: {scl_path}\nRun step3 first.")
        print(f"    Loading SVM from {clf_path}...")
        print(f"    Loading scaler from {scl_path}...")
        classifier = joblib.load(clf_path)
        scaler = joblib.load(scl_path)

    else:
        raise ValueError(f"Unknown classifier: {classifier_name!r}. Use 'rf' or 'svm'.")

    # ImageNet normalization transform for the CNN path
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    print("Pipeline loaded successfully.\n")
    return cnn_model, classifier, scaler, transform, device


# Pipeline globals - populated in __main__ before launching the demo
feature_extractor: Optional[torch.nn.Module] = None
classifier: Any = None
scaler: Any = None
transform: Optional[transforms.Compose] = None
device: Optional[torch.device] = None
current_backbone: str = ""
current_classifier_name: str = ""

# Class names in label order
LABEL_NAMES = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]


# -------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------
def predict_image(image):
    """Take a PIL/numpy image, return {class_name: confidence}."""
    if image is None:
        return {" No image provided": 1.0}

    if classifier is None or transform is None or device is None:
        raise RuntimeError("Pipeline not loaded - run this module as __main__.")

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    feature_parts = []

    # --- CNN path (skipped for freq-only backbone) ---
    if feature_extractor is not None:
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            cnn_feats = feature_extractor(img_tensor)
        feature_parts.append(cnn_feats.cpu().numpy())   # (1, 1280) or (1, 2048)

    # --- Frequency path (FFT + DCT) ---
    if current_backbone in FREQ_BACKBONES:
        # Convert to grayscale float32 [0,255] — same as _load_grayscale in frequency_features.py
        img_resized = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        r, g, b = img_resized.split()
        gray = (0.299 * np.array(r, dtype=np.float32)
                + 0.587 * np.array(g, dtype=np.float32)
                + 0.114 * np.array(b, dtype=np.float32))
        freq_feats = extract_frequency_features_from_array(
            gray,
            n_fft_bins=FREQ_N_FFT_BINS,
            dct_block_size=FREQ_DCT_BLOCK_SIZE,
        )
        feature_parts.append(freq_feats.reshape(1, -1))   # (1, 228)

    # Concatenate all feature parts → (1, total_dim)
    features_np = np.concatenate(feature_parts, axis=1)

    # SVM requires the StandardScaler that was fit at training time
    if scaler is not None:
        features_np = scaler.transform(features_np)

    probabilities = classifier.predict_proba(features_np)[0]
    return {LABEL_NAMES[i]: float(probabilities[i]) for i in range(len(LABEL_NAMES))}


# -------------------------------------------------------------------
# Gradio interface
# -------------------------------------------------------------------
def create_demo():
    clf_label = "SVM" if current_classifier_name == "svm" else "Random Forest"
    demo = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil", label="Upload an Image"),
        outputs=gr.Label(num_top_classes=3, label="Classification Result"),
        title="TruPhoto: Image Authenticity Classifier",
        description=(
            "Upload an image to classify it as **Real** (authentic photograph), "
            "**Forged** (digitally manipulated via splicing or copy-move), or "
            "**AI-Generated** (fully synthetic from a generative model).\n\n"
            f"**Architecture:** Frozen {current_backbone} feature extractor + {clf_label} classifier\n\n"
            "**CS483 Fundamentals of Artificial Intelligence - San Francisco Bay University**"
        ),
        examples=None,
        allow_flagging="never",
        theme="default",
    )
    return demo


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    ALL_BACKBONES = list(CNN_BACKBONE_MAP.keys())
    DEFAULT_BACKBONE = f"{ABLATION_MODEL}_freq"   # best ablation result (75.70%)

    parser = argparse.ArgumentParser(description="TruPhoto Gradio demo")
    parser.add_argument(
        "--backbone", default=DEFAULT_BACKBONE,
        choices=ALL_BACKBONES,
        help=f"Feature set (default: {DEFAULT_BACKBONE}).",
    )
    parser.add_argument(
        "--classifier", default="svm", choices=["rf", "svm"],
        help="Classifier head (default: svm, the best combination's classifier).",
    )
    parser.add_argument(
        "--public", action="store_true",
        help="Bind to 0.0.0.0 (LAN-accessible). Default: 127.0.0.1 (localhost only).",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to serve on (default: 7860).",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share URL (tunneled through gradio.app).",
    )
    args = parser.parse_args()

    current_backbone = args.backbone
    current_classifier_name = args.classifier

    print("\n" + "=" * 60)
    print("  TruPhoto - Step 5: Gradio Demo")
    print("=" * 60)
    print(f"\n  Backbone:   {current_backbone}")
    print(f"  Classifier: {current_classifier_name.upper()}")
    print(f"  Classes:    {', '.join(LABEL_NAMES)}")

    feature_extractor, classifier, scaler, transform, device = load_pipeline(
        current_backbone, current_classifier_name
    )

    bind = "0.0.0.0" if args.public else "127.0.0.1"
    print(f"\n  Starting Gradio server on {bind}:{args.port}...\n")

    demo = create_demo()
    demo.launch(
        server_name=bind,
        server_port=args.port,
        share=args.share,
    )

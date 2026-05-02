"""
step5_gradio_demo.py — Interactive Demo for TruPhoto

Launches a Gradio web interface where users can upload an image
and get a classification result: Real, Forged, or AI-Generated,
along with confidence scores for each class.

PIPELINE (default):
  Upload image -> Resize to 299x299 -> Normalize
  -> MobileNetV2 features (1280-d, frozen)
  -> StandardScaler -> SVM (RBF) -> Class label + per-class confidence

The default uses the best combination from the ablation study
(mobilenetv2_100 + SVM, 77.62% val acc on the 12.5K-per-class run).

USAGE (run from repo root):
  python -m src.step5_gradio_demo                          # default best combo
  python -m src.step5_gradio_demo --backbone xception      # use Xception features
  python -m src.step5_gradio_demo --classifier rf          # use Random Forest head
  python -m src.step5_gradio_demo --public                 # accept LAN connections
  python -m src.step5_gradio_demo --share                  # public Gradio URL

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
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
)


# -------------------------------------------------------------------
# Pipeline loading
# -------------------------------------------------------------------
def load_pipeline(backbone: str, classifier_name: str):
    """
    Load (feature extractor, classifier, scaler_or_None, transform, device).

    Args:
        backbone:        timm model identifier ("xception" or "mobilenetv2_100")
        classifier_name: "rf" or "svm". SVM also loads a fitted StandardScaler.
    """
    print(" Loading TruPhoto demo pipeline...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    # Feature extractor (frozen, no head)
    print(f"    Loading {backbone} from timm...")
    feature_extractor = timm.create_model(backbone, pretrained=True, num_classes=0)
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor = feature_extractor.to(device)

    # Classifier (and scaler if SVM)
    if classifier_name == "rf":
        clf_path = os.path.join(MODELS_DIR, f"rf_{backbone}.joblib")
        if not os.path.isfile(clf_path):
            raise FileNotFoundError(
                f"Trained RF not found: {clf_path}\nRun step3 first."
            )
        print(f"    Loading Random Forest classifier from {clf_path}...")
        classifier = joblib.load(clf_path)
        scaler = None

    elif classifier_name == "svm":
        clf_path = os.path.join(MODELS_DIR, f"svm_{backbone}.joblib")
        scl_path = os.path.join(MODELS_DIR, f"scaler_{backbone}.joblib")
        if not os.path.isfile(clf_path):
            raise FileNotFoundError(
                f"Trained SVM not found: {clf_path}\nRun step3 first."
            )
        if not os.path.isfile(scl_path):
            raise FileNotFoundError(
                f"SVM scaler not found: {scl_path}\nRun step3 first."
            )
        print(f"    Loading SVM classifier from {clf_path}...")
        print(f"    Loading scaler from {scl_path}...")
        classifier = joblib.load(clf_path)
        scaler = joblib.load(scl_path)

    else:
        raise ValueError(f"Unknown classifier: {classifier_name!r}. Use 'rf' or 'svm'.")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    print("Pipeline loaded successfully.\n")
    return feature_extractor, classifier, scaler, transform, device


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

    if (feature_extractor is None or classifier is None
            or transform is None or device is None):
        raise RuntimeError("Pipeline not loaded - run this module as __main__.")

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    img_tensor = transform(image).unsqueeze(0).to(device)

    # Frozen CNN forward pass -> feature vector
    with torch.no_grad():
        features = feature_extractor(img_tensor)
    features_np = features.cpu().numpy()

    # SVM requires the same StandardScaler that was fit at training time.
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
    parser = argparse.ArgumentParser(description="TruPhoto Gradio demo")
    parser.add_argument(
        "--backbone", default=ABLATION_MODEL,
        choices=[PRIMARY_MODEL, ABLATION_MODEL],
        help=f"Feature extractor (default: {ABLATION_MODEL}, the best combination's backbone).",
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

"""
Central configuration for TruPhoto pipeline

all paths, hyperparameters and constants are contained here

"""

import os


# DIRECTORY PATHS

# root of the project 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ROW DATASET LOCATIONS
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifact")
CASIA2_DIR = os.path.join(PROJECT_ROOT, "casia2")

# processed data (resized images organized by class)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# where extracted feature arrays (.npy) get saved
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")

# trained model artifacts (joblib files)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# evaluation of outputs (plots, reports)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# CLASS DEFINITIONS
CLASS_NAMES = {
    0: "Real",
    1: "Forged",
    2: "AI_Generated"
}

NUM_CLASSES = 3


# DATASET PARAMETERS

# Target number of images per class (balanced sampling)
SAMPLES_PER_CLASS = 5000

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed
RANDOM_SEED = 42


# Image Preprocessing
# -------------------

# XceptionNet expects 299x299 input
IMG_SIZE = 299

# ImageNet Normalization Stats
# XceptionNet was pretrained on ImageNet so we use the same mean/std 
# reference: https://github.com/pytorch/vision/issues/1439
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# feature extraction
# --------------------
# main architecture
PRIMARY_MODEL = "xception"

# comparison architecture to check how well is the main CNN architecture above works
ABLATION_MODEL = "mobilenetv2_100"

# Batch size for feature extraction
BATCH_SIZE = 32

# Number of dataloader workers (set 0 on Windows if issues)
NUM_WORKERS = 4 


# CLASSIFIER HYPERPARAMETERS
#----------------------------
# Random Forest
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = None # let tree grow fully
RF_MIN_SAMPLES_SPLIT = 5
RF_CLASS_WEIGHT = "balanced" # to handle any residual imbalance

# SVM (ablation model)
SVM_KERNEL = "RBF"
SVM_C = 10.0
SVM_GAMMA = "scale"
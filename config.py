"""Configuration for baseline cGAN project.

Adjust paths and hyperparameters here.
"""
from typing import List
from pathlib import Path

# Dataset
DATA_ROOT = "./data/archive/img_align_celeba/img_align_celeba"  # CelebA image folder
ATTR_FILE = "./data/archive/list_attr_celeba.csv"  # CelebA attribute CSV
SELECTED_ATTRS: List[str] = ["Smiling", "Male", "Eyeglasses", "Blond_Hair", "Young"]
MAX_IMAGES = 30000  # set to None to use full dataset
SUBSET_SEED = 42  # reproducible subset selection

# Training hyperparameters (shared)
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
LATENT_DIM = 100  # total latent dimension
IMAGE_SIZE = 64
ATTR_DIM = len(SELECTED_ATTRS)

# Disentanglement hyperparameters
LATENT_DIM_C = 8  # controllable latent code dimension
LATENT_DIM_N = LATENT_DIM - LATENT_DIM_C  # random noise dimension (92)
LAMBDA_INFO = 1.0  # weight for disentanglement loss (z_c reconstruction)

# Checkpoints and outputs
OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = "./outputs/checkpoints"
SAMPLES_DIR = "./outputs/samples"
EVAL_DIR = "./outputs/eval"

# Experiment tracking (MLflow)
ENABLE_TRACKING = True
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME_BASELINE = "cgan_baseline"
EXPERIMENT_NAME_DISENTANGLED = "cgan_disentangled"

# Misc
SEED = 42
DEVICE = None  # if None, helper will pick cuda/mps/cpu

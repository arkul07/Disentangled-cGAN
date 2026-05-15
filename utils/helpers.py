"""General helper utilities: seeding and device selection."""
from typing import Optional
import random
import os

import torch
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS support (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

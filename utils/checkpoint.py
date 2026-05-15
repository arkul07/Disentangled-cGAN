"""Checkpoint save/load helpers."""
from typing import Dict, Any
import os
import torch


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)

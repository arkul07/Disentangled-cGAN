"""Image utilities: denormalize, fixed noise, and saving grids."""
from typing import Tuple, Optional
import os

import torch
from torchvision.utils import save_image


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """Convert tensor from [-1,1] to [0,1]."""
    return img.mul(0.5).add(0.5)


def save_image_grid(tensor: torch.Tensor, path: str, nrow: int = 8) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor = denormalize(tensor)
    save_image(tensor, path, nrow=nrow)


def fixed_noise(batch_size: int, latent_dim: int, device: torch.device, seed: int = 42) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(batch_size, latent_dim, generator=generator, device=device)

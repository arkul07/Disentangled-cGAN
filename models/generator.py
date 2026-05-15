"""Generator for conditional image generation.

Simple DCGAN-style generator that takes noise and attribute vectors.
"""
from typing import Tuple

import torch
import torch.nn as nn


class View(nn.Module):
    """Helper layer to reshape tensors inside Sequential."""

    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*self.shape)


class Generator(nn.Module):
    """DCGAN-style generator for 64x64 images.

    Takes concatenated latent code, noise, and attributes as input.
    """

    def __init__(
        self, 
        latent_dim: int = 100, 
        attr_dim: int = 5, 
        ngf: int = 64, 
        img_channels: int = 3
    ):
        """Initialize generator.

        Args:
            latent_dim: total latent dimension (z_c + z_n)
            attr_dim: attribute vector dimension
            ngf: base number of generator features
            img_channels: output image channels (3 for RGB)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.attr_dim = attr_dim
        input_dim = latent_dim + attr_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, ngf * 8 * 4 * 4),
            nn.BatchNorm1d(ngf * 8 * 4 * 4),
            nn.ReLU(True),
            View((-1, ngf * 8, 4, 4)),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(ngf, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate image from latent code and attributes.

        Args:
            z: (B, latent_dim) latent vector
            y: (B, attr_dim) attribute vector

        Returns:
            (B, img_channels, 64, 64) generated image in [-1, 1]
        """
        x = torch.cat([z, y], dim=1)
        return self.net(x)

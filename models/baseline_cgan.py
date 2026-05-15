"""Baseline conditional DCGAN models (Generator and Discriminator).

Clear, simple architectures suitable for 64x64 RGB images.
Conditioning is implemented by concatenating the attribute vector.
"""
from typing import Optional

import torch
import torch.nn as nn


class Generator(nn.Module):
    """DCGAN-style generator that conditions on attributes by concatenation.

    Args:
        latent_dim: size of z vector
        attr_dim: size of attribute vector y
        ngf: base feature maps
        img_channels: usually 3 for RGB
    """

    def __init__(self, latent_dim: int = 100, attr_dim: int = 5, ngf: int = 64, img_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.attr_dim = attr_dim
        input_dim = latent_dim + attr_dim

        self.net = nn.Sequential(
            # project and reshape
            nn.Linear(input_dim, ngf * 8 * 4 * 4),
            nn.BatchNorm1d(ngf * 8 * 4 * 4),
            nn.ReLU(True),
            # reshape to (ngf*8) x 4 x 4
            View((-1, ngf * 8, 4, 4)),
            # upsample to 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 64x64
            nn.ConvTranspose2d(ngf, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate images from noise z and attribute vector y.

        z: (B, latent_dim)
        y: (B, attr_dim)
        returns: (B, 3, 64, 64)
        """
        x = torch.cat([z, y], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    """DCGAN-style discriminator that conditions by concatenating attribute maps.

    The attribute vector y is expanded spatially and concatenated to image channels.
    """

    def __init__(self, attr_dim: int = 5, ndf: int = 64, img_channels: int = 3):
        super().__init__()
        input_channels = img_channels + attr_dim
        self.net = nn.Sequential(
            # 64x64
            nn.Conv2d(input_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4 -> final logit
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, img: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return logits for input images conditioned on y.

        img: (B, 3, 64, 64)
        y: (B, attr_dim)
        returns: (B, ) logits
        """
        b = img.size(0)
        # expand y to spatial map
        y_map = y.view(b, y.size(1), 1, 1).expand(-1, -1, img.size(2), img.size(3))
        x = torch.cat([img, y_map], dim=1)
        out = self.net(x)
        return out.view(-1)


class View(nn.Module):
    """Helper layer to reshape tensors inside Sequential."""

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

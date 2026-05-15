"""Auxiliary Q network for disentanglement.

Simple network that recovers the controllable latent code z_c from generated images.
Inspired by InfoGAN's mutual information maximization approach.
"""
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Q network for InfoGAN-style disentanglement.

    Takes a generated image and predicts the controllable latent code z_c.
    Used to enforce that generated images preserve information about z_c.
    """

    def __init__(self, latent_dim_c: int = 8, ndf: int = 64, img_channels: int = 3):
        """Initialize Q network.

        Args:
            latent_dim_c: dimension of controllable latent code z_c
            ndf: base number of features
            img_channels: input image channels (3 for RGB)
        """
        super().__init__()
        self.latent_dim_c = latent_dim_c

        # Shared feature extraction (similar to discriminator backbone)
        self.features = nn.Sequential(
            # 64x64
            nn.Conv2d(img_channels, ndf, 4, 2, 1, bias=False),
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
            # 4x4 -> flatten
            nn.Conv2d(ndf * 8, 256, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Predict z_c (continuous latent code)
        self.fc_z_c = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim_c),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Recover z_c from generated image.

        Args:
            img: (B, 3, 64, 64) generated image in [-1, 1]

        Returns:
            (B, latent_dim_c) predicted z_c
        """
        features = self.features(img)
        features = features.view(features.size(0), -1)
        z_c_pred = self.fc_z_c(features)
        return z_c_pred

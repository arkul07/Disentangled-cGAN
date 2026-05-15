"""Discriminator for conditional image classification.

DCGAN-style discriminator that conditions on attributes.
"""
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """DCGAN-style discriminator for 64x64 images.

    Takes image and attribute vectors, outputs real/fake logit.
    """

    def __init__(self, attr_dim: int = 5, ndf: int = 64, img_channels: int = 3):
        """Initialize discriminator.

        Args:
            attr_dim: attribute vector dimension
            ndf: base number of discriminator features
            img_channels: input image channels (3 for RGB)
        """
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
            # 4x4 -> logit
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, img: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Classify image as real/fake conditioned on attributes.

        Args:
            img: (B, 3, 64, 64) image in [-1, 1]
            y: (B, attr_dim) attribute vector

        Returns:
            (B,) logits for real/fake classification
        """
        b = img.size(0)
        y_map = y.view(b, y.size(1), 1, 1).expand(-1, -1, img.size(2), img.size(3))
        x = torch.cat([img, y_map], dim=1)
        out = self.net(x)
        return out.view(-1)

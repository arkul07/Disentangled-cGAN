"""Wrapper for the disentangled conditional GAN components.

This module bundles the generator, discriminator, and Q network so it is easy to
instantiate and compare against the baseline model.
"""
from dataclasses import dataclass

import torch

from models.generator import Generator
from models.discriminator import Discriminator
from models.q_network import QNetwork


@dataclass
class CGANDisentangled:
    """Container for the disentangled conditional GAN components."""

    generator: Generator
    discriminator: Discriminator
    q_network: QNetwork

    @classmethod
    def build(
        cls,
        latent_dim: int,
        latent_dim_c: int,
        attr_dim: int,
        ngf: int = 64,
        ndf: int = 64,
        img_channels: int = 3,
    ) -> "CGANDisentangled":
        """Construct the disentangled GAN components."""
        generator = Generator(latent_dim=latent_dim, attr_dim=attr_dim, ngf=ngf, img_channels=img_channels)
        discriminator = Discriminator(attr_dim=attr_dim, ndf=ndf, img_channels=img_channels)
        q_network = QNetwork(latent_dim_c=latent_dim_c, ndf=ndf, img_channels=img_channels)
        return cls(generator=generator, discriminator=discriminator, q_network=q_network)

    def to(self, device: torch.device) -> "CGANDisentangled":
        """Move all submodules to device."""
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        self.q_network = self.q_network.to(device)
        return self

    def train(self) -> "CGANDisentangled":
        """Set all modules to train mode."""
        self.generator.train()
        self.discriminator.train()
        self.q_network.train()
        return self

    def eval(self) -> "CGANDisentangled":
        """Set all modules to eval mode."""
        self.generator.eval()
        self.discriminator.eval()
        self.q_network.eval()
        return self

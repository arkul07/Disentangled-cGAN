"""Evaluate disentangled cGAN model by generating controlled image variations.

Visualize the effect of changing z_c, z_n, and attributes separately to inspect
whether the model has learned meaningful disentanglement.
"""
import os
from typing import List

import torch
from torchvision.utils import make_grid

from config import *
from utils.helpers import get_device, set_seed
from utils.checkpoint import load_checkpoint
from utils.image_utils import save_image_grid, fixed_noise
from models.generator import Generator


def load_generator(ckpt_path: str, device: torch.device) -> Generator:
    """Load a disentangled generator from checkpoint."""
    ckpt = load_checkpoint(ckpt_path, device=device)
    cfg = ckpt.get("config", {})
    latent_dim = cfg.get("latent_dim", LATENT_DIM)
    attr_dim = cfg.get("attr_dim", ATTR_DIM)
    net_g = Generator(latent_dim=latent_dim, attr_dim=attr_dim)
    net_g.load_state_dict(ckpt["netG_state"])
    net_g.to(device).eval()
    return net_g


def vary_z_c(
    net_g: Generator,
    z_c_dim: int,
    z_n: torch.Tensor,
    attrs: torch.Tensor,
    num_samples: int = 8,
    output_path: str = None,
) -> None:
    """Generate grid where one z_c dimension is smoothly traversed and z_n is fixed.

    This shows the effect of the controllable latent code.
    """
    device = next(net_g.parameters()).device
    z_c_base = torch.zeros(1, z_c_dim, device=device)
    traverse_values = torch.linspace(-2.0, 2.0, steps=num_samples, device=device)
    z_c_grid = []

    for value in traverse_values:
        z_c = z_c_base.clone()
        z_c[0, 0] = value
        z_c_grid.append(z_c)

    z_c_cat = torch.cat(z_c_grid, dim=0)  # (num_samples, z_c_dim)
    z_n_repeated = z_n.repeat(num_samples, 1)  # (num_samples, z_n_dim)
    z = torch.cat([z_c_cat, z_n_repeated], dim=1)

    with torch.no_grad():
        imgs = net_g(z, attrs.repeat(num_samples, 1))

    if output_path:
        save_image_grid(imgs, output_path, nrow=num_samples)


def vary_z_n(
    net_g: Generator,
    z_c: torch.Tensor,
    z_n_dim: int,
    attrs: torch.Tensor,
    num_samples: int = 8,
    output_path: str = None,
) -> None:
    """Generate grid where z_n values are varied and z_c is fixed.

    This shows the effect of the random noise component.
    """
    device = next(net_g.parameters()).device
    z_n_grid = []

    for i in range(num_samples):
        z_n = torch.randn(1, z_n_dim, device=device)
        z_n_grid.append(z_n)

    # Repeat z_c for each z_n
    z_n_cat = torch.cat(z_n_grid, dim=0)  # (num_samples, z_n_dim)
    z_c_repeated = z_c.repeat(num_samples, 1)  # (num_samples, z_c_dim)
    z = torch.cat([z_c_repeated, z_n_cat], dim=1)

    with torch.no_grad():
        imgs = net_g(z, attrs.repeat(num_samples, 1))

    if output_path:
        save_image_grid(imgs, output_path, nrow=num_samples)


def vary_attributes(
    net_g: Generator,
    z_c: torch.Tensor,
    z_n: torch.Tensor,
    attr_indices: List[int],
    num_samples: int = 8,
    output_path: str = None,
) -> None:
    """Generate grid where specific attributes are varied.

    This shows the effect of attribute control.
    """
    device = next(net_g.parameters()).device
    attr_grid = []

    for i in range(num_samples):
        attrs = torch.zeros(1, ATTR_DIM, device=device)
        for attr_idx in attr_indices:
            if i >= num_samples // 2:  # Second half: attribute on
                attrs[0, attr_idx] = 1.0
        attr_grid.append(attrs)

    attrs_cat = torch.cat(attr_grid, dim=0)  # (num_samples, attr_dim)
    z = torch.cat([z_c.repeat(num_samples, 1), z_n.repeat(num_samples, 1)], dim=1)

    with torch.no_grad():
        imgs = net_g(z, attrs_cat)

    if output_path:
        save_image_grid(imgs, output_path, nrow=num_samples)


def main(ckpt_path: str) -> None:
    """Generate evaluation grids for disentangled model."""
    set_seed(SEED)
    device = get_device() if DEVICE is None else torch.device(DEVICE)
    net_g = load_generator(ckpt_path, device)

    os.makedirs(EVAL_DIR, exist_ok=True)

    # Fixed latent codes and attributes for evaluation
    z_c_fixed = torch.randn(1, LATENT_DIM_C, device=device)
    z_n_fixed = torch.randn(1, LATENT_DIM_N, device=device)
    attrs_default = torch.zeros(1, ATTR_DIM, device=device)

    print("Generating evaluation grids...")
    print("Fixed latent settings:")
    print(f"  z_c dim: {LATENT_DIM_C}")
    print(f"  z_n dim: {LATENT_DIM_N}")
    print(f"  attributes: {ATTR_DIM}")

    # 1. Vary z_c (controllable code) with fixed z_n and attributes
    output_1 = os.path.join(EVAL_DIR, "01_vary_zc.png")
    vary_z_c(net_g, LATENT_DIM_C, z_n_fixed, attrs_default, num_samples=8, output_path=output_1)
    print(f"Saved: {output_1}")

    # 2. Vary z_n (noise) with fixed z_c and attributes
    output_2 = os.path.join(EVAL_DIR, "02_vary_zn.png")
    vary_z_n(net_g, z_c_fixed, LATENT_DIM_N, attrs_default, num_samples=8, output_path=output_2)
    print(f"Saved: {output_2}")

    # 3. Vary attribute 0 (Smiling)
    output_3 = os.path.join(EVAL_DIR, "03_vary_attr0_smiling.png")
    vary_attributes(net_g, z_c_fixed, z_n_fixed, [0], num_samples=8, output_path=output_3)
    print(f"Saved: {output_3}")

    # 4. Vary attribute 1 (Male)
    output_4 = os.path.join(EVAL_DIR, "04_vary_attr1_male.png")
    vary_attributes(net_g, z_c_fixed, z_n_fixed, [1], num_samples=8, output_path=output_4)
    print(f"Saved: {output_4}")

    # 5. Vary attribute 2 (Eyeglasses)
    output_5 = os.path.join(EVAL_DIR, "05_vary_attr2_eyeglasses.png")
    vary_attributes(net_g, z_c_fixed, z_n_fixed, [2], num_samples=8, output_path=output_5)
    print(f"Saved: {output_5}")

    print(f"\nAll evaluation grids saved to: {EVAL_DIR}")
    print("\nInterpretation guide:")
    print("  01_vary_zc.png     - If z_c is disentangled, changing it should vary image details/pose/expression")
    print("  02_vary_zn.png     - z_n (noise) changes should add randomness/texture variations")
    print("  03-05_vary_attr*.png - Attribute control should change specific facial features")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to disentangled generator checkpoint")
    args = parser.parse_args()
    main(args.ckpt)

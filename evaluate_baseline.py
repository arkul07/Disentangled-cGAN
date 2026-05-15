"""Evaluate the baseline generator and produce image grids for attribute combinations.

Loads a generator checkpoint and saves generated grids to outputs/eval/.
"""
from typing import List
import os

import torch
from torchvision.utils import make_grid

from config import *
from utils.helpers import get_device, set_seed
from utils.checkpoint import load_checkpoint
from utils.image_utils import save_image_grid, fixed_noise
from models.baseline_cgan import Generator


def load_generator(ckpt_path: str, device: torch.device) -> Generator:
    ckpt = load_checkpoint(ckpt_path, device=device)
    cfg = ckpt.get("config", {})
    latent_dim = cfg.get("latent_dim", LATENT_DIM)
    attr_dim = cfg.get("attr_dim", ATTR_DIM)
    netG = Generator(latent_dim=latent_dim, attr_dim=attr_dim)
    netG.load_state_dict(ckpt["netG_state"])
    netG.to(device).eval()
    return netG


def generate_for_attributes(netG: Generator, attr_vectors: torch.Tensor, out_path: str, batch_size: int = 64) -> None:
    device = next(netG.parameters()).device
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with torch.no_grad():
        z = fixed_noise(attr_vectors.size(0), netG.latent_dim, device)
        imgs = netG(z, attr_vectors.to(device))
        save_image_grid(imgs, out_path, nrow=8)


def main(ckpt_path: str, manual_attrs: List[List[int]] = None) -> None:
    set_seed(SEED)
    device = get_device() if DEVICE is None else torch.device(DEVICE)
    netG = load_generator(ckpt_path, device)
    os.makedirs(EVAL_DIR, exist_ok=True)

    if manual_attrs is None:
        # default: vary single attribute on/off across rows
        manual_attrs = []
        base = [0] * ATTR_DIM
        for i in range(ATTR_DIM):
            row_on = base.copy()
            row_on[i] = 1
            manual_attrs.append(row_on)
            manual_attrs.append(base)

    attr_tensors = torch.tensor(manual_attrs, dtype=torch.float32)
    out_path = os.path.join(EVAL_DIR, "manual_attrs.png")
    generate_for_attributes(netG, attr_tensors, out_path)
    print(f"Saved evaluation images to: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to generator checkpoint")
    args = parser.parse_args()
    main(args.ckpt)

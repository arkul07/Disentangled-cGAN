"""Train a disentangled conditional GAN with InfoGAN-style auxiliary network.

This script trains the main model with both adversarial and disentanglement losses.
"""
import os
import time
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import *
from utils.helpers import set_seed, get_device
from utils.checkpoint import save_checkpoint
from utils.image_utils import save_image_grid, fixed_noise
from utils.logger import log_epoch
from utils.experiment_tracking import create_tracker
from data.celeba_dataset import CelebADataset
from models.generator import Generator
from models.discriminator import Discriminator
from models.q_network import QNetwork


def build_dataloaders() -> DataLoader:
    dataset = CelebADataset(
        images_dir=DATA_ROOT,
        attr_file=ATTR_FILE,
        selected_attrs=SELECTED_ATTRS,
        image_size=IMAGE_SIZE,
        max_images=MAX_IMAGES,
        subset_seed=SUBSET_SEED,
        training=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    return loader


def train() -> None:
    """Train the disentangled conditional GAN."""
    set_seed(SEED)
    device = get_device() if DEVICE is None else torch.device(DEVICE)
    print(f"Using device: {device}")
    print(f"Disentanglement config: latent_dim_c={LATENT_DIM_C}, latent_dim_n={LATENT_DIM_N}, lambda_info={LAMBDA_INFO}")

    tracker = create_tracker(
        enabled=ENABLE_TRACKING,
        experiment_name=EXPERIMENT_NAME_DISENTANGLED,
        tracking_uri=TRACKING_URI,
        run_name=f"disentangled_lambda_{LAMBDA_INFO}",
    )
    tracker.start()
    tracker.log_params(
        {
            "model": "disentangled_cgan",
            "latent_dim": LATENT_DIM,
            "latent_dim_c": LATENT_DIM_C,
            "latent_dim_n": LATENT_DIM_N,
            "attr_dim": ATTR_DIM,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "beta1": BETA1,
            "beta2": BETA2,
            "lambda_info": LAMBDA_INFO,
            "image_size": IMAGE_SIZE,
            "selected_attrs": ",".join(SELECTED_ATTRS),
            "device": str(device),
        }
    )

    dataloader = build_dataloaders()

    # Initialize networks
    net_g = Generator(latent_dim=LATENT_DIM, attr_dim=ATTR_DIM).to(device)
    net_d = Discriminator(attr_dim=ATTR_DIM).to(device)
    net_q = QNetwork(latent_dim_c=LATENT_DIM_C).to(device)

    # Loss functions
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_info = nn.MSELoss()  # for z_c reconstruction

    # Optimizers
    opt_g = torch.optim.Adam(net_g.parameters(), lr=LR, betas=(BETA1, BETA2))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=LR, betas=(BETA1, BETA2))
    opt_q = torch.optim.Adam(net_q.parameters(), lr=LR, betas=(BETA1, BETA2))

    # Fixed latent codes for sampling during training
    fixed_z_c = torch.randn(64, LATENT_DIM_C, device=device)
    fixed_z_n = torch.randn(64, LATENT_DIM_N, device=device)
    fixed_z = torch.cat([fixed_z_c, fixed_z_n], dim=1)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    real_label = 1.0
    fake_label = 0.0

    print("\n--- Starting disentangled cGAN training ---\n")

    for epoch in range(1, EPOCHS + 1):
        d_losses = []
        g_losses = []
        q_losses = []
        info_losses = []
        start_time = time.time()

        for real_imgs, attrs in dataloader:
            bs = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            attrs = attrs.to(device)

            # Sample latent codes: z_c (controllable) and z_n (random)
            z_c = torch.randn(bs, LATENT_DIM_C, device=device)
            z_n = torch.randn(bs, LATENT_DIM_N, device=device)
            z = torch.cat([z_c, z_n], dim=1)

            # ============================================
            # Train Discriminator
            # ============================================
            net_d.zero_grad()

            # Real images
            labels_real = torch.full((bs,), real_label, dtype=torch.float32, device=device)
            logits_real = net_d(real_imgs, attrs)
            loss_d_real = criterion_gan(logits_real, labels_real)

            # Fake images
            with torch.no_grad():
                fake_imgs = net_g(z, attrs)
            labels_fake = torch.full((bs,), fake_label, dtype=torch.float32, device=device)
            logits_fake = net_d(fake_imgs.detach(), attrs)
            loss_d_fake = criterion_gan(logits_fake, labels_fake)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            # ============================================
            # Train Generator and Q Network
            # ============================================
            for param in net_d.parameters():
                param.requires_grad = False

            net_g.zero_grad()
            net_q.zero_grad()

            # Generate fake images
            fake_imgs = net_g(z, attrs)

            # Adversarial loss: fool the discriminator
            labels_g = torch.full((bs,), real_label, dtype=torch.float32, device=device)
            logits_fake_for_g = net_d(fake_imgs, attrs)
            loss_g_adv = criterion_gan(logits_fake_for_g, labels_g)

            # Disentanglement loss: Q network reconstructs z_c
            z_c_pred = net_q(fake_imgs)
            loss_info = criterion_info(z_c_pred, z_c)

            # Total generator loss
            loss_g_total = loss_g_adv + LAMBDA_INFO * loss_info

            loss_g_total.backward()
            opt_g.step()
            opt_q.step()

            for param in net_d.parameters():
                param.requires_grad = True

            d_losses.append(loss_d.item())
            g_losses.append(loss_g_adv.item())
            q_losses.append(loss_info.item())
            info_losses.append(loss_info.item())

        epoch_time = time.time() - start_time
        avg_d = sum(d_losses) / len(d_losses) if d_losses else 0.0
        avg_g = sum(g_losses) / len(g_losses) if g_losses else 0.0
        avg_info = sum(info_losses) / len(info_losses) if info_losses else 0.0
        avg_total_g = avg_g + LAMBDA_INFO * avg_info

        tracker.log_metrics(
            {
                "d_loss": avg_d,
                "g_adv_loss": avg_g,
                "info_loss": avg_info,
                "g_total_loss": avg_total_g,
            },
            step=epoch,
        )

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"D_loss {avg_d:.4f} | G_adv {avg_g:.4f} | "
            f"Info {avg_info:.4f} | G_total {avg_total_g:.4f} | "
            f"Time {epoch_time:.1f}s"
        )

        # Save samples
        if epoch % 5 == 0 or epoch == 1:
            net_g.eval()
            with torch.no_grad():
                sample = net_g(fixed_z, torch.zeros(fixed_z.size(0), ATTR_DIM, device=device))
                path = os.path.join(SAMPLES_DIR, f"disentangled_epoch_{epoch:03d}.png")
                save_image_grid(sample, path, nrow=8)
                tracker.log_artifact(path)

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "netG_state": net_g.state_dict(),
            "netD_state": net_d.state_dict(),
            "netQ_state": net_q.state_dict(),
            "optG_state": opt_g.state_dict(),
            "optD_state": opt_d.state_dict(),
            "optQ_state": opt_q.state_dict(),
            "config": {
                "latent_dim": LATENT_DIM,
                "latent_dim_c": LATENT_DIM_C,
                "latent_dim_n": LATENT_DIM_N,
                "attr_dim": ATTR_DIM,
                "lambda_info": LAMBDA_INFO,
            },
        }
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"disentangled_ckpt_epoch_{epoch:03d}.pth")
        save_checkpoint(ckpt, ckpt_path)
        if epoch == EPOCHS:
            tracker.log_artifact(ckpt_path)

    tracker.end()
    print("Training finished.")


if __name__ == "__main__":
    train()

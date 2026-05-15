"""Train a baseline conditional GAN on CelebA-style data.

Simple, readable training loop with alternating updates.
"""
from typing import Tuple
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from config import *
from utils.helpers import set_seed, get_device
from utils.checkpoint import save_checkpoint
from utils.image_utils import save_image_grid, fixed_noise
from utils.logger import log_epoch
from utils.experiment_tracking import create_tracker
from data.celeba_dataset import CelebADataset
from models.baseline_cgan import Generator, Discriminator


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
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    return loader


def train():
    set_seed(SEED)
    device = get_device() if DEVICE is None else torch.device(DEVICE)
    print(f"Using device: {device}")

    tracker = create_tracker(
        enabled=ENABLE_TRACKING,
        experiment_name=EXPERIMENT_NAME_BASELINE,
        tracking_uri=TRACKING_URI,
        run_name="baseline_cgan",
    )
    tracker.start()
    tracker.log_params(
        {
            "model": "baseline_cgan",
            "latent_dim": LATENT_DIM,
            "attr_dim": ATTR_DIM,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "beta1": BETA1,
            "beta2": BETA2,
            "image_size": IMAGE_SIZE,
            "selected_attrs": ",".join(SELECTED_ATTRS),
            "device": str(device),
        }
    )

    dataloader = build_dataloaders()

    netG = Generator(latent_dim=LATENT_DIM, attr_dim=ATTR_DIM).to(device)
    netD = Discriminator(attr_dim=ATTR_DIM).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, BETA2))

    fixed_z = fixed_noise(64, LATENT_DIM, device, seed=SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    real_label = 1.0
    fake_label = 0.0

    step = 0
    for epoch in range(1, EPOCHS + 1):
        d_losses = []
        g_losses = []
        start = time.time()
        for i, (real_imgs, attrs) in enumerate(dataloader, 1):
            bs = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            attrs = attrs.to(device)

            # Train Discriminator: real
            netD.zero_grad()
            labels_real = torch.full((bs,), real_label, dtype=torch.float32, device=device)
            logits_real = netD(real_imgs, attrs)
            lossD_real = criterion(logits_real, labels_real)

            # Fake
            noise = torch.randn(bs, LATENT_DIM, device=device)
            fake_imgs = netG(noise, attrs)
            labels_fake = torch.full((bs,), fake_label, dtype=torch.float32, device=device)
            logits_fake = netD(fake_imgs.detach(), attrs)
            lossD_fake = criterion(logits_fake, labels_fake)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimD.step()

            # Train Generator
            netG.zero_grad()
            labels_g = torch.full((bs,), real_label, dtype=torch.float32, device=device)
            logits_fake_for_g = netD(fake_imgs, attrs)
            lossG = criterion(logits_fake_for_g, labels_g)
            lossG.backward()
            optimG.step()

            d_losses.append(lossD.item())
            g_losses.append(lossG.item())

            step += 1

        avg_d = sum(d_losses) / len(d_losses) if d_losses else 0.0
        avg_g = sum(g_losses) / len(g_losses) if g_losses else 0.0
        log_epoch(epoch, EPOCHS, avg_d, avg_g)
        tracker.log_metrics({"d_loss": avg_d, "g_loss": avg_g}, step=epoch)

        # save samples
        if epoch % 5 == 0 or epoch == 1:
            with torch.no_grad():
                sample = netG(fixed_z, torch.zeros(fixed_z.size(0), ATTR_DIM, device=device))
                path = os.path.join(SAMPLES_DIR, f"epoch_{epoch:03d}.png")
                save_image_grid(sample, path, nrow=8)
            tracker.log_artifact(path)

        # save checkpoint
        ckpt = {
            "epoch": epoch,
            "netG_state": netG.state_dict(),
            "netD_state": netD.state_dict(),
            "optimG_state": optimG.state_dict(),
            "optimD_state": optimD.state_dict(),
            "config": {
                "latent_dim": LATENT_DIM,
                "attr_dim": ATTR_DIM,
            },
        }
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_epoch_{epoch:03d}.pth")
        save_checkpoint(ckpt, ckpt_path)
        if epoch == EPOCHS:
            tracker.log_artifact(ckpt_path)

    tracker.end()
    print("Training finished.")


if __name__ == "__main__":
    train()

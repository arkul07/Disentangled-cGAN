# Conditional GAN for Face Image Generation: Baseline and Disentangled Models

This project provides two PyTorch implementations for generating 64x64 face images conditioned on attributes (CelebA subset):

1. **Baseline cGAN**: A straightforward DCGAN-style conditional generator and discriminator
2. **Disentangled cGAN**: An extended model with InfoGAN-style auxiliary network for improved attribute control

## Project Structure

```
data/
    celeba_dataset.py        # CelebA dataset loader
    archive/                 # Your CelebA images and attributes CSV
models/
    generator.py             # Generator network
    discriminator.py         # Discriminator network
    q_network.py             # Q network for disentanglement (auxiliary)
    baseline_cgan.py         # Baseline models
utils/
    helpers.py               # Seed, device selection
    checkpoint.py            # Checkpoint save/load
    image_utils.py           # Image denormalize, fixed noise, grid saving
    logger.py                # Simple logging
outputs/
    checkpoints/             # Model checkpoints
    samples/                 # Generated image grids during training
    eval/                    # Evaluation grids with controlled variations
    plots/                   # Training metric visualizations
    reports/                 # CSV summary tables
train_baseline.py            # Train the baseline cGAN
train_disentangled.py        # Train the disentangled cGAN
evaluate_baseline.py         # Evaluate baseline generator
evaluate_disentangled.py     # Evaluate disentangled generator
compare_experiments.py       # Build comparison CSV
config.py                    # All hyperparameters and paths
requirements.txt             # Dependencies
```

## Key Concepts

### Baseline Model
- Generator takes: random noise `z` + attribute vector `y` → 64×64 face
- Discriminator takes: image + attributes → real/fake logit (no sigmoid; uses `BCEWithLogitsLoss`)
- Loss: Standard binary cross-entropy GAN loss

### Disentangled Model
Split the latent space into two components:

1. **z_c** (8 dimensions): Controllable latent code — encouraged to encode structured variation
2. **z_n** (92 dimensions): Random noise — captures high-frequency details
3. **Q Network**: Independent CNN that takes a generated image and predicts `z_c`, enforcing that the generator encodes `z_c` in a recoverable way

**Loss Functions**:
- Adversarial loss: Standard GAN loss
- Disentanglement loss: MSE between true `z_c` and Q-network's prediction
- Total generator loss: `L_adv + lambda_info * L_info`

## Getting Started

### 1. Prepare Dataset

Place CelebA images in `data/archive/img_align_celeba/img_align_celeba/` and the attribute CSV at `data/archive/list_attr_celeba.csv`.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train

```bash
python train_baseline.py
python train_disentangled.py
```

### 4. Evaluate

```bash
python evaluate_baseline.py --ckpt outputs/checkpoints/ckpt_epoch_050.pth
python evaluate_disentangled.py --ckpt outputs/checkpoints/disentangled_ckpt_epoch_050.pth
```

Generates:
- `outputs/eval/01_vary_zc.png` — z_c traversal (structured facial changes)
- `outputs/eval/02_vary_zn.png` — z_n traversal (texture/detail changes)
- `outputs/eval/03-05_vary_attr*.png` — per-attribute sweeps

### 5. Compare Runs

```bash
python compare_experiments.py
```

Writes a summary CSV to `outputs/reports/experiment_comparison.csv`.

## Experiment Tracking

Training logs to a local MLflow store. To open the dashboard after training:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Baseline** logs: `d_loss`, `g_loss`, hyperparameters  
**Disentangled** logs: `d_loss`, `g_adv_loss`, `info_loss`, `g_total_loss`, hyperparameters

## Configuration

Edit `config.py` to tune hyperparameters:

```python
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.0002
LATENT_DIM = 100
LATENT_DIM_C = 8       # controllable code dimension
LATENT_DIM_N = 92      # noise dimension
LAMBDA_INFO = 1.0      # disentanglement weight
MAX_IMAGES = 30000     # set to None for full dataset
```

## Results (30k CelebA subset, 50 epochs)

| Metric | Baseline | Disentangled |
|--------|----------|--------------|
| D loss | 0.2592 | 0.1448 |
| G adv loss | 5.1907 | 4.9332 |
| Info loss | — | 0.0546 |
| G total loss | 5.1907 | 4.9877 |

All values are epoch-averaged training losses from the final training epoch.

## References

- **DCGAN**: Radford et al., 2016
- **InfoGAN**: Chen et al., 2016
- **Conditional GANs**: Mirza & Osindero, 2014

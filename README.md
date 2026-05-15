# Disentangled Conditional GAN for Attribute-Controlled Face Generation

This repository contains a baseline conditional GAN and an enhanced disentangled conditional GAN for face generation on CelebA. The goal is to improve attribute control (e.g., Smiling, Eyeglasses) without changing unrelated features.

Use this README as the complete project description. It is written to be fed into a report‑generation tool.

## Abstract
We study attribute‑controlled face generation with conditional GANs and improve control by disentangling the latent input. The noise vector is split into a controllable part $z_c$ and an unconstrained part $z_n$, and an auxiliary network is trained to reconstruct $z_c$ from generated images. This adds an InfoGAN‑style regularization term that encourages predictable changes when $z_c$ is varied. We train a baseline cGAN and the disentangled model on a 30k‑image CelebA subset and compare training behavior and qualitative controllability. The disentangled model produces clearer, more independent attribute manipulations while preserving overall image quality.

**Keywords:** conditional GAN; disentanglement; InfoGAN; latent space; attribute control; CelebA; face generation.

## Introduction
Conditional GANs are widely used for attribute‑controlled face generation, but in practice their latent variables remain entangled. As a result, modifying a single attribute often introduces unintended changes in identity, pose, or texture. This limits both interpretability and fine‑grained control, which are important for downstream analysis and user‑driven synthesis.

To address this, we introduce a minimal extension to the baseline cGAN that encourages disentanglement in the latent space without changing the core generator or discriminator design. The noise vector is partitioned into a small controllable code and a residual noise component. An auxiliary predictor is trained to recover the controllable code from generated images, adding a regularization term that promotes consistent and independent variation. We evaluate this approach against the baseline using a 30k‑image CelebA subset and analyze both training behavior and qualitative controllability.

## Problem Statement
The main challenge is to control specific facial attributes while preserving other visual factors. Standard conditional GANs can follow attribute labels, but the latent space mixes multiple factors. We aim to separate controllable and uncontrollable variations so that changes to a small latent code produce predictable and isolated effects.

## Methodology
**Baseline cGAN.** The baseline uses a DCGAN‑style generator and discriminator. The generator takes noise $z$ and attribute labels $y$ and outputs a $64\times64$ RGB image. The discriminator takes the image and labels and predicts real vs. fake. The loss is standard binary cross‑entropy GAN loss.

**Disentangled cGAN.** We split the noise vector into $z=[z_c,z_n]$, where $z_c$ is a small controllable code and $z_n$ captures remaining variation. The generator receives $(z_c,z_n,y)$. An auxiliary network $Q$ predicts $z_c$ from generated images, encouraging $z_c$ to encode meaningful and independent factors. The generator loss is

$$
\mathcal{L}_G = \mathcal{L}_{GAN} + \lambda\,\mathcal{L}_{info}, \quad \mathcal{L}_{info}=\|Q(G(z_c,z_n,y)) - z_c\|_2^2.
$$

## Dataset
We use CelebA with five attributes:
- Smiling
- Male
- Eyeglasses
- Blond_Hair
- Young

The project is configured to use a fixed subset of 30,000 images for faster training and fair comparison across models.

## Training Setup
Shared training settings:
- Image size: $64\times64$
- Batch size: 64
- Epochs: 50
- Learning rate: 0.0002
- Adam betas: $(0.5, 0.999)$
- Latent dimension: 100

Disentangled model settings:
- $z_c$ dimension: 8
- $z_n$ dimension: 92
- $\lambda$ (disentanglement weight): 1.0

## Evaluation
We evaluate in two ways:
1) **Training metrics** (loss curves for generator and discriminator, plus info loss for the disentangled model).
2) **Qualitative controllability** by generating grids where only $z_c$, only $z_n$, or a single attribute is varied.

## Results (Latest Runs on 30k Subset)
From [outputs/reports/experiment_comparison.csv](outputs/reports/experiment_comparison.csv):

**Baseline cGAN**
- $d\_loss = 0.2592$
- $g\_loss = 5.1907$

**Disentangled cGAN**
- $d\_loss = 0.1448$
- $g\_{adv} = 4.9332$
- $info\_loss = 0.0546$
- $g\_{total} = 4.9877$

Qualitatively, the disentangled model shows clearer and more isolated changes when $z_c$ is varied, while $z_n$ primarily changes texture and minor details. Attribute toggles (Smiling, Male, Eyeglasses) show targeted changes with less drift in other factors.

## Project Structure
```
data/
    celeba_dataset.py        # CelebA dataset loader
    archive/                 # CelebA images and attributes CSV
models/
    generator.py             # Generator network
    discriminator.py         # Discriminator network
    q_network.py             # Q network for disentanglement
    baseline_cgan.py         # Baseline models
utils/
    helpers.py               # Seed, device selection
    checkpoint.py            # Checkpoint save/load
    image_utils.py           # Image grids and noise helpers
    logger.py                # Logging
outputs/
    checkpoints/             # Model checkpoints
    samples/                 # Training sample grids
    eval/                    # Evaluation grids
    plots/                   # Training plots
    reports/                 # CSV summary tables
train_baseline.py            # Train baseline cGAN
train_disentangled.py        # Train disentangled cGAN
evaluate_baseline.py         # Evaluate baseline generator
evaluate_disentangled.py     # Evaluate disentangled generator
compare_experiments.py       # Build comparison CSV
config.py                    # Hyperparameters and paths
```

## How to Reproduce
1) Place CelebA images and the attributes CSV in `data/archive/`.
2) Install dependencies: `pip install -r requirements.txt`.
3) Train baseline: `python train_baseline.py`.
4) Train disentangled: `python train_disentangled.py`.
5) Evaluate disentangled: `python evaluate_disentangled.py --ckpt outputs/checkpoints/disentangled_ckpt_epoch_050.pth`.
6) Compare runs: `python compare_experiments.py`.

## Figures to Include in the Report
**Training Dynamics**
- [outputs/plots/01_loss_trajectory.png](outputs/plots/01_loss_trajectory.png)
- [outputs/plots/04_training_stability.png](outputs/plots/04_training_stability.png)

**Qualitative Controllability (Disentangled Model)**
- [outputs/eval/01_vary_zc.png](outputs/eval/01_vary_zc.png)
- [outputs/eval/02_vary_zn.png](outputs/eval/02_vary_zn.png)
- [outputs/eval/03_vary_attr0_smiling.png](outputs/eval/03_vary_attr0_smiling.png)
- [outputs/eval/04_vary_attr1_male.png](outputs/eval/04_vary_attr1_male.png)
- [outputs/eval/05_vary_attr2_eyeglasses.png](outputs/eval/05_vary_attr2_eyeglasses.png)

**Comparison Table**
- [outputs/reports/experiment_comparison.csv](outputs/reports/experiment_comparison.csv)

## Notes
This project uses a 30k‑image subset of CelebA for faster iteration. To switch to the full dataset, set `MAX_IMAGES = None` in `config.py` and retrain.# Disentangled Conditional GAN for Attribute-Controlled Face Generation

Use this text as the source content for the final report. It is written in simple academic language and aligns with the current code and results.

## Abstract
We study attribute‑controlled face generation with conditional GANs and improve control by disentangling the latent input. The noise vector is split into a controllable part $z_c$ and an unconstrained part $z_n$, and an auxiliary network is trained to reconstruct $z_c$ from generated images. This adds an InfoGAN‑style regularization term that encourages predictable changes when $z_c$ is varied. We train a baseline cGAN and the disentangled model on a 30k‑image CelebA subset and compare training behavior and qualitative controllability. The disentangled model produces clearer, more independent attribute manipulations while preserving overall image quality.

**Keywords:** conditional GAN; disentanglement; InfoGAN; latent space; attribute control; CelebA; face generation.

## Introduction
Conditional GANs are widely used for attribute‑controlled face generation, but in practice their latent variables remain entangled. As a result, modifying a single attribute often introduces unintended changes in identity, pose, or texture. This limits both interpretability and fine‑grained control, which are important for downstream analysis and user‑driven synthesis.

To address this, we introduce a minimal extension to the baseline cGAN that encourages disentanglement in the latent space without changing the core generator or discriminator design. The noise vector is partitioned into a small controllable code and a residual noise component. An auxiliary predictor is trained to recover the controllable code from generated images, adding a regularization term that promotes consistent and independent variation. We evaluate this approach against the baseline using a 30k‑image CelebA subset and analyze both training behavior and qualitative controllability.

## Methodology
**Baseline cGAN.** The baseline uses a DCGAN‑style generator and discriminator. The generator takes noise $z$ and attribute labels $y$ and outputs a $64\times64$ RGB image. The discriminator takes the image and labels and predicts real vs. fake. The loss is standard binary cross‑entropy GAN loss.

**Disentangled cGAN.** We split the noise vector into $z=[z_c,z_n]$, where $z_c$ is a small controllable code and $z_n$ captures remaining variation. The generator receives $(z_c,z_n,y)$. An auxiliary network $Q$ predicts $z_c$ from generated images, encouraging $z_c$ to encode meaningful and independent factors. The generator loss is

$$
\mathcal{L}_G = \mathcal{L}_{GAN} + \lambda\,\mathcal{L}_{info}, \quad \mathcal{L}_{info}=\|Q(G(z_c,z_n,y)) - z_c\|_2^2.
$$

## Experiments
**Dataset.** CelebA images with five attributes: Smiling, Male, Eyeglasses, Blond_Hair, Young. We use a fixed subset of 30,000 images for all runs.

**Training.** Both baseline and disentangled models are trained for 50 epochs with batch size 64, learning rate 0.0002, and Adam betas $(0.5,0.999)$. The disentangled model uses $\lambda=1.0$, latent dimension 100 with $z_c=8$ and $z_n=92$.

**Evaluation.** We report training losses and qualitative controllability. For qualitative checks, we generate grids where only $z_c$, only $z_n$, or a single attribute is varied while other inputs are fixed.

## Results
**Training metrics (latest runs).**
- Baseline (30k subset): $d\_loss=0.2592$, $g\_loss=5.1907$
- Disentangled (30k subset): $d\_loss=0.1448$, $g\_{adv}=4.9332$, $info\_loss=0.0546$, $g\_{total}=4.9877$

These values come from [outputs/reports/experiment_comparison.csv](outputs/reports/experiment_comparison.csv).

**Qualitative controllability.** The disentangled model shows clearer changes when $z_c$ is varied, while $z_n$ mainly adds random texture or small details. Attribute toggles (e.g., Smiling, Male, Eyeglasses) show targeted changes without large identity shifts.

## Conclusion
We presented a simple disentanglement mechanism for a conditional GAN by splitting the latent code and adding an auxiliary reconstruction loss. This modification improves controllability while keeping the core GAN architecture unchanged. Results on a 30k‑image CelebA subset show stable training and stronger qualitative control over attributes.

## Figures to Include (with placement suggestions)
**Methodology / Architecture:**
- If you want a method figure, use a simple schematic of $z_c$, $z_n$, and $Q$ network (not provided in the repo). This can be drawn in the report.

**Results / Training Dynamics:**
- [outputs/plots/01_loss_trajectory.png](outputs/plots/01_loss_trajectory.png) — place in Results under “Training dynamics.”
- [outputs/plots/04_training_stability.png](outputs/plots/04_training_stability.png) — place after loss trajectories.

**Qualitative Controllability (Disentangled model):**
- [outputs/eval/01_vary_zc.png](outputs/eval/01_vary_zc.png) — place in Results under “Latent traversal (z_c).”
- [outputs/eval/02_vary_zn.png](outputs/eval/02_vary_zn.png) — place near z_c figure as contrast.
- [outputs/eval/03_vary_attr0_smiling.png](outputs/eval/03_vary_attr0_smiling.png) — place in Results under “Attribute control.”
- [outputs/eval/04_vary_attr1_male.png](outputs/eval/04_vary_attr1_male.png)
- [outputs/eval/05_vary_attr2_eyeglasses.png](outputs/eval/05_vary_attr2_eyeglasses.png)

**Comparison Table:**
- [outputs/reports/experiment_comparison.csv](outputs/reports/experiment_comparison.csv) — convert to a table in the Results section.# Conditional GAN for Face Image Generation: Baseline and Disentangled Models

This project provides two PyTorch implementations for generating 64x64 face images conditioned on attributes (CelebA/CelebA-HQ subset):

1. **Baseline cGAN**: A straightforward DCGAN-style conditional generator and discriminator
2. **Disentangled cGAN**: An extended model with InfoGAN-style auxiliary network for improved attribute control

Both models are designed as clean, beginner-friendly student research code.

## Project Structure

```
data/
    celeba_dataset.py        # CelebA dataset loader
    archive/                 # Your CelebA images and attributes CSV
models/
    generator.py             # Generator network
    discriminator.py         # Discriminator network
    q_network.py             # Q network for disentanglement (auxiliary)
    baseline_cgan.py         # Legacy baseline models (for backward compat)
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
train_baseline.py            # Train the baseline cGAN
train_disentangled.py        # Train the disentangled cGAN (main model)
evaluate_baseline.py         # Evaluate baseline generator
evaluate_disentangled.py     # Evaluate disentangled generator (main model)
config.py                    # All hyperparameters and paths
requirements.txt             # Dependencies
README.md                     # This file
```

## Key Concepts

### Baseline Model
- **Simple conditional GAN** for face generation
- Generator takes: random noise `z` + attribute vector `y` → 64×64 face
- Discriminator takes: image + attributes → real/fake classification
- Loss: Standard binary cross-entropy GAN loss

### Disentangled Model (Main)
Split the latent space into two interpretable components:

1. **z_c**: Controllable latent code (8 dimensions)
   - Intended to capture interpretable variations
   - Should reflect identity, pose, expression, etc.
   - Directly controllable by the user

2. **z_n**: Random noise (92 dimensions, if total latent_dim = 100)
   - Captures high-frequency details and randomness
   - Not directly controlled

3. **Q Network**: Auxiliary network
   - Takes generated image → predicts z_c
   - Encourages generator to preserve information about z_c in the image
   - Inspired by **InfoGAN** (Mutual Information Maximization)

**Loss Functions**:
- Adversarial loss: Standard GAN loss (discriminator fools generator)
- Disentanglement loss: MSE between true z_c and Q-network's prediction of z_c
- Total generator loss: `L_adv + lambda_info * L_info`

This encourages:
- **Good generation quality** (via adversarial loss)
- **Attribute control** (via conditioning on y)
- **Disentanglement** (via z_c reconstruction via Q network)

## Getting Started

### 1. Prepare Dataset

Place your CelebA images in `data/archive/img_align_celeba/img_align_celeba/` and the attribute CSV in `data/archive/list_attr_celeba.csv`.

Expected CSV format (first few rows):
```
image_id,5_o_Clock_Shadow,Arched_Eyebrows,...,Smiling,Male,Eyeglasses,Blond_Hair,Young,...
000001.jpg,-1,1,...,1,0,0,1,0,...
000002.jpg,-1,-1,...,-1,0,0,0,1,...
```

The code automatically selects 5 attributes: `Smiling`, `Male`, `Eyeglasses`, `Blond_Hair`, `Young`.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch torchvision Pillow numpy matplotlib scipy mlflow
```

### 3. Train Baseline Model

```bash
python train_baseline.py
```

Outputs:
- Checkpoints: `outputs/checkpoints/ckpt_epoch_*.pth`
- Samples: `outputs/samples/epoch_*.png`
- MLflow logs: `mlflow.db` (tracking history) and `mlruns/` (artifacts)

### 4. Train Disentangled Model (Main)

```bash
python train_disentangled.py
```

Outputs:
- Checkpoints: `outputs/checkpoints/disentangled_ckpt_epoch_*.pth`
- Samples: `outputs/samples/disentangled_epoch_*.png`
- MLflow logs: `mlflow.db` (tracking history) and `mlruns/` (artifacts)

To adjust disentanglement weight:
```python
# In config.py, change:
LAMBDA_INFO = 2.0  # higher = stronger disentanglement pressure
```

Then retrain.

### 5. Evaluate and Visualize

**Baseline evaluation**:
```bash
python evaluate_baseline.py --ckpt outputs/checkpoints/ckpt_epoch_050.pth
```

**Disentangled evaluation** (visualize controllability):
```bash
python evaluate_disentangled.py --ckpt outputs/checkpoints/disentangled_ckpt_epoch_050.pth
```

This generates:
- `outputs/eval/01_vary_zc.png` - Changing z_c (should show pose/expression changes)
- `outputs/eval/02_vary_zn.png` - Changing z_n (should show texture/detail changes)
- `outputs/eval/03-05_vary_attr*.png` - Varying each attribute individually

## Experiment Tracking

Training scripts log to a local **MLflow** store by default. This gives you a dashboard for training history and hyperparameter tuning.

### Open the Dashboard

After training, run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open the local URL that MLflow prints in your browser.

### Logged Values

**Baseline run** logs:
- `d_loss`
- `g_loss`
- latent dimension, batch size, learning rate, beta values, selected attributes

**Disentangled run** logs:
- `d_loss`
- `g_adv_loss`
- `info_loss`
- `g_total_loss`
- `lambda_info`, `latent_dim_c`, `latent_dim_n`, and the other training hyperparameters

Sample grids and final checkpoints are logged as MLflow artifacts so they are visible in the run page.

### Comparing Runs

To compare different controllability settings, change `LAMBDA_INFO` in `config.py` and rerun training.

Useful comparison values:

```python
LAMBDA_INFO = 0.5
LAMBDA_INFO = 1.0
LAMBDA_INFO = 2.0
```

Each run appears separately in MLflow, making it easy to compare the quality vs. controllability tradeoff.

### Report Summary Table

Generate a compact comparison table and CSV from the latest tracked baseline and disentangled runs:

```bash
python compare_experiments.py
```

This writes a report-ready CSV to `outputs/reports/experiment_comparison.csv` and prints a terminal table.

## Configuration

Edit `config.py` to tune hyperparameters:

```python
# Dataset
DATA_ROOT = "./data/archive/img_align_celeba/img_align_celeba"
ATTR_FILE = "./data/archive/list_attr_celeba.csv"
SELECTED_ATTRS = ["Smiling", "Male", "Eyeglasses", "Blond_Hair", "Young"]

# Training
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.0002  # Adam learning rate
LATENT_DIM = 100

# Disentanglement (for main model)
LATENT_DIM_C = 8      # controllable code dimension
LATENT_DIM_N = 92     # noise dimension
LAMBDA_INFO = 1.0     # disentanglement weight
```

## Training Tips

1. **Baseline first**: Train the baseline to get a feel for convergence and data loading
2. **Monitor metrics**: Check loss curves in `outputs/plots/` after training
3. **Disentanglement tradeoff**: 
   - High `LAMBDA_INFO` → stronger disentanglement but may reduce generation quality
   - Low `LAMBDA_INFO` → better generation but less controllable
   - Start with `LAMBDA_INFO = 1.0`
4. **Device selection**: Automatic (CUDA > MPS > CPU)

## Model Comparison

| Aspect | Baseline | Disentangled |
|--------|----------|--------------|
| Latent control | z only | z_c + z_n |
| Attribute control | Yes (y vector) | Yes (y vector) |
| Interpretability | Limited | Higher (z_c is learned code) |
| Q network | No | Yes |
| Complexity | Simple | Moderate |
| Training time | Faster | Slower (~1.2x) |

## Next Steps / Future Work

1. **Compute FID score** for quantitative quality metrics
2. **Ablation studies**: Train with different `LAMBDA_INFO` values
3. **Latent traversals**: Smooth interpolations along z_c dimensions
4. **Attribute entanglement analysis**: Measure how z_c correlates with each attribute
5. **Extended attributes**: Add more facial attributes or expressions
6. **Higher resolution**: Extend to 128×128 or 256×256 images

## Code Quality Notes

- **Readable and modular**: Each component (G, D, Q) is in its own file
- **Clear loss functions**: Separate adversarial and disentanglement losses
- **Docstrings**: All major functions documented
- **Type hints**: Used where helpful for clarity
- **Error handling**: Missing files and checkpoint issues are caught and reported

## References

- **DCGAN**: *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks* (Radford et al., 2016)
- **InfoGAN**: *InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets* (Chen et al., 2016)
- **Conditional GANs**: *Conditional Generative Adversarial Nets* (Mirza & Osindski, 2014)

## Support

- For missing dataset: Ensure CSV file is at `ATTR_FILE` path
- For CUDA out of memory: Reduce `BATCH_SIZE` in `config.py`
- For slow training: Ensure GPU is being used (check device output at start)
- For convergence issues: Try lower learning rate or different `LAMBDA_INFO`

---

**This is a baseline research project.** The goal is clarity and understanding over state-of-the-art performance.

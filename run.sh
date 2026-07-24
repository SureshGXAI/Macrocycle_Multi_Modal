#!/bin/bash

# Macrocycle Multi-Modal Model

## Train

python3 train.py \
  --sdf_path data/curated_macrocycles_f300.sdf \
  --epochs 100 \
  --batch_size 32 \
  --limit 200 \
  --ckpt_dir checkpoints


## Generate molecules

# Sample fresh macrocycles from the learned latent space
python3 generate.py --ckpt checkpoints/macrocycle_epoch99.pt --mode sample --num_samples 20

# Reconstruct a seed molecule through the fused latent (autoencoding check)
python3 generate.py --ckpt checkpoints/macrocycle_epoch99.pt --mode reconstruct \
  --smiles "C1CCCCCCCCCC1"

# Interpolate between two macrocycles in latent space
python3 generate.py --ckpt checkpoints/macrocycle_epoch99.pt --mode interpolate \
  --smiles "C1CCCCCCCCCC1" "C1CCOCCOCCOCC1" --steps 8


## Assessing model & training quality

### a. Training dynamics
python3 plot_training_curves.py --ckpt_dir checkpoints --out_dir plots

### b. Full quality evaluation
python3 eval.py --ckpt checkpoints/macrocycle_epoch99.pt --sdf_path data/curated_macrocycles_f300.sdf

## Predict properties

python3 predict.py --ckpt checkpoints/macrocycle_epoch99.pt \
  --smiles "C1CCCCCCCCCC1" "C1CCOCCOCCOCC1"

# from a new SDF file:
python3 predict.py --ckpt checkpoints/macrocycle_epoch99.pt --sdf_path MCD10000.sdf

"""
Generate novel macrocycle SMILES from a trained checkpoint.

Two modes:
  1. Reconstruction / interpolation: encode existing seed molecule(s) to the
     fused latent space and decode (optionally after latent interpolation).
  2. Prior sampling: sample random Gaussian latents (approx. the aggregate
     posterior, since there's no explicit VAE prior here) and decode -- a
     simple way to explore the learned macrocycle latent space.

Usage:
    python generate.py --ckpt checkpoints/macrocycle_epoch99.pt --mode sample --num_samples 20
    python generate.py --ckpt checkpoints/macrocycle_epoch99.pt --mode reconstruct --smiles "C1CCCCCCCCCC1"
    python generate.py --ckpt checkpoints/macrocycle_epoch99.pt --mode interpolate --smiles "C1CCCCCCCCCC1" "C1CCOCCOCCOCC1" --steps 8
"""
import argparse
import logging

import torch
from rdkit import Chem

from config import DataConfig, ModelConfig
from data.tokenizer import SmilesTokenizer
from models.model import MacrocycleModel
from utils import setup_logging
from predict import records_from_smiles, build_batch

logger = logging.getLogger("generate")


def is_valid_smiles(smi: str) -> bool:
    return Chem.MolFromSmiles(smi) is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--mode", choices=["sample", "reconstruct", "interpolate"], default="sample")
    parser.add_argument("--smiles", type=str, nargs="*", default=None)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=8, help="interpolation steps")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    setup_logging()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    tokenizer = SmilesTokenizer(ckpt["vocab"])
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    data_cfg = DataConfig(**ckpt["data_cfg"])

    model = MacrocycleModel(model_cfg, pad_id=tokenizer.pad_id).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if args.mode == "sample":
        latent = torch.randn(args.num_samples, model_cfg.latent_dim, device=device)
        ids = model.generate_from_latent(latent, tokenizer, max_len=data_cfg.max_smiles_len,
                                          temperature=args.temperature, top_k=args.top_k)
        smiles_out = [tokenizer.decode(row.tolist()) for row in ids]

    elif args.mode == "reconstruct":
        if not args.smiles:
            raise ValueError("--smiles required for reconstruct mode")
        recs = records_from_smiles(args.smiles)
        batch = build_batch(recs, tokenizer, data_cfg.max_smiles_len, tokenizer.pad_id)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        ids = model.generate_from_batch(batch, tokenizer, max_len=data_cfg.max_smiles_len,
                                         temperature=args.temperature, top_k=args.top_k)
        smiles_out = [tokenizer.decode(row.tolist()) for row in ids]
        for orig, recon in zip(args.smiles, smiles_out):
            print(f"input : {orig}\nrecon : {recon}\n")
        return

    else:  # interpolate
        if not args.smiles or len(args.smiles) != 2:
            raise ValueError("--smiles requires exactly 2 SMILES for interpolate mode")
        recs = records_from_smiles(args.smiles)
        batch = build_batch(recs, tokenizer, data_cfg.max_smiles_len, tokenizer.pad_id)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        with torch.no_grad():
            fused, _ = model.encode(batch)
        z0, z1 = fused[0], fused[1]
        alphas = torch.linspace(0, 1, args.steps, device=device)
        latents = torch.stack([(1 - a) * z0 + a * z1 for a in alphas], dim=0)
        ids = model.generate_from_latent(latents, tokenizer, max_len=data_cfg.max_smiles_len,
                                          temperature=args.temperature, top_k=args.top_k)
        smiles_out = [tokenizer.decode(row.tolist()) for row in ids]

    n_valid = sum(is_valid_smiles(s) for s in smiles_out)
    logger.info(f"Generated {len(smiles_out)} molecules, {n_valid} chemically valid "
                f"({100.0 * n_valid / max(1, len(smiles_out)):.1f}%)")
    for s in smiles_out:
        flag = "valid  " if is_valid_smiles(s) else "invalid"
        print(f"[{flag}] {s}")


if __name__ == "__main__":
    main()

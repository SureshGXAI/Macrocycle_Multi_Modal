"""
Train the macrocycle multi-modal model end-to-end on a 50K-molecule SDF file.

Usage:
    python train.py --sdf_path data/macrocycles_50k.sdf --epochs 100 --batch_size 32
"""
import argparse
import csv
import logging
import math
import os

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import DataConfig, ModelConfig, TrainConfig
from data.dataset import load_sdf_records, split_records, MacrocycleDataset
from data.tokenizer import SmilesTokenizer
from data.collate import macrocycle_collate
from models.model import MacrocycleModel
from losses import MultiTaskLoss
from utils import set_seed, setup_logging, save_checkpoint, count_parameters

logger = logging.getLogger("train")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--sdf_path", type=str, required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--ckpt_dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--limit", type=int, default=None, help="Debug: cap number of SDF records loaded")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return p


def get_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return lr_lambda


def evaluate(model, loader, criterion, device):
    model.eval()
    totals = {}
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            _, metrics = criterion(outputs, batch)
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1
    return {k: v / max(1, n_batches) for k, v in totals.items()}


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


class CsvLogger:
    """Appends one row per call to a CSV, writing the header on first use.
    Produces train_log.csv (per logged step) and val_log.csv (per epoch) in
    ckpt_dir, which plot_training_curves.py reads to render loss curves."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._header_written = os.path.exists(path)

    def log(self, row: dict):
        write_header = not self._header_written
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


def main():
    args = build_argparser().parse_args()
    setup_logging()

    data_cfg = DataConfig(sdf_path=args.sdf_path)
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    if args.epochs: train_cfg.epochs = args.epochs
    if args.batch_size: train_cfg.batch_size = args.batch_size
    if args.lr: train_cfg.lr = args.lr
    if args.ckpt_dir: train_cfg.ckpt_dir = args.ckpt_dir
    if args.device: train_cfg.device = args.device

    set_seed(data_cfg.seed)
    device = torch.device(train_cfg.device if torch.cuda.is_available() or train_cfg.device == "cpu" else "cpu")
    logger.info(f"Using device: {device}")

    # ---- Load SDF & split ----
    logger.info(f"Loading SDF from {data_cfg.sdf_path} ...")
    records = load_sdf_records(
        data_cfg.sdf_path, data_cfg.property_fields, max_atoms=data_cfg.max_atoms, limit=args.limit,
    )
    if len(records) == 0:
        raise RuntimeError("No valid molecules loaded from SDF. Check the file path/format.")
    train_recs, val_recs, test_recs = split_records(
        records, data_cfg.val_fraction, data_cfg.test_fraction, seed=data_cfg.seed
    )
    logger.info(f"Split sizes -> train: {len(train_recs)}, val: {len(val_recs)}, test: {len(test_recs)}")

    # ---- Tokenizer (fit on train SMILES only) ----
    tokenizer = SmilesTokenizer.build_from_smiles_list([r.smiles for r in train_recs])
    logger.info(f"SMILES vocab size: {len(tokenizer)}")
    model_cfg.smiles_vocab_size = len(tokenizer)

    # ---- Feature dims (set dynamically from actual featurizer output) ----
    model_cfg.graph_atom_feat_dim = train_recs[0].atom_feats.shape[-1]
    model_cfg.graph_bond_feat_dim = train_recs[0].edge_feats.shape[-1] if train_recs[0].edge_feats.shape[0] > 0 else 8
    model_cfg.num_properties = len(data_cfg.property_fields)
    logger.info(f"atom_feat_dim={model_cfg.graph_atom_feat_dim}, bond_feat_dim={model_cfg.graph_bond_feat_dim}")

    # ---- Datasets / loaders ----
    train_ds = MacrocycleDataset(train_recs, tokenizer, data_cfg.max_smiles_len)
    val_ds = MacrocycleDataset(val_recs, tokenizer, data_cfg.max_smiles_len,
                                prop_mean=train_ds.prop_mean, prop_std=train_ds.prop_std)

    collate = lambda b: macrocycle_collate(b, pad_id=tokenizer.pad_id)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True,
                               num_workers=data_cfg.num_workers, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False,
                             num_workers=data_cfg.num_workers, collate_fn=collate)

    # ---- Model / optimizer / loss ----
    model = MacrocycleModel(model_cfg, pad_id=tokenizer.pad_id).to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    optimizer = AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    total_steps = train_cfg.epochs * len(train_loader)
    scheduler = LambdaLR(optimizer, get_lr_lambda(train_cfg.warmup_steps, total_steps))
    criterion = MultiTaskLoss(
        pad_id=tokenizer.pad_id, w_property=train_cfg.w_property, w_generation=train_cfg.w_generation,
        w_contrastive=train_cfg.w_contrastive, contrastive_temperature=train_cfg.contrastive_temperature,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.amp and device.type == "cuda")

    train_logger = CsvLogger(os.path.join(train_cfg.ckpt_dir, "train_log.csv"))
    val_logger = CsvLogger(os.path.join(train_cfg.ckpt_dir, "val_log.csv"))

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, train_cfg.epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=train_cfg.amp and device.type == "cuda"):
                outputs = model(batch)
                loss, metrics = criterion(outputs, batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            if step % train_cfg.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                logger.info(
                    f"epoch {epoch} step {step}/{total_steps} lr {lr_now:.2e} | "
                    + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                )
                train_logger.log({"epoch": epoch, "step": step, "lr": lr_now, **metrics})

        val_metrics = evaluate(model, val_loader, criterion, device)
        logger.info(f"[epoch {epoch}] VAL " + " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
        val_logger.log({"epoch": epoch, "step": step, **val_metrics})

        if (epoch + 1) % train_cfg.save_every_epochs == 0 or epoch == train_cfg.epochs - 1:
            ckpt_path = os.path.join(train_cfg.ckpt_dir, f"macrocycle_epoch{epoch}.pt")
            save_checkpoint(ckpt_path, model, optimizer, epoch, tokenizer, data_cfg, model_cfg,
                             train_ds.prop_mean, train_ds.prop_std)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()

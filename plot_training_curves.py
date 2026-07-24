"""
Renders training-quality diagnostic plots from the CSV logs produced by
train.py (checkpoints/train_log.csv, checkpoints/val_log.csv):
  1. Total loss: train vs val, per epoch  -> overfitting check
  2. Each loss component (property / generation / contrastive) over training
  3. Learning rate schedule sanity check

Usage:
    python plot_training_curves.py --ckpt_dir checkpoints --out_dir plots
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--out_dir", type=str, default="plots")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train_path = os.path.join(args.ckpt_dir, "train_log.csv")
    val_path = os.path.join(args.ckpt_dir, "val_log.csv")
    train_rows = read_csv(train_path) if os.path.exists(train_path) else []
    val_rows = read_csv(val_path) if os.path.exists(val_path) else []

    if not train_rows and not val_rows:
        print(f"No logs found in {args.ckpt_dir}. Run train.py first.")
        return

    # ---- 1. Train vs Val total loss (per epoch, averaged for train) ----
    if train_rows:
        by_epoch = {}
        for r in train_rows:
            by_epoch.setdefault(r["epoch"], []).append(r["loss_total"])
        epochs = sorted(by_epoch)
        train_epoch_loss = [sum(by_epoch[e]) / len(by_epoch[e]) for e in epochs]

        plt.figure(figsize=(7, 5))
        plt.plot(epochs, train_epoch_loss, label="train (epoch avg)")
        if val_rows:
            plt.plot([r["epoch"] for r in val_rows], [r["loss_total"] for r in val_rows],
                     label="val", marker="o")
        plt.xlabel("epoch")
        plt.ylabel("total loss")
        plt.title("Train vs Val Loss (diverging curves = overfitting)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(args.out_dir, "loss_train_vs_val.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # ---- 2. Loss components over training steps ----
    if train_rows:
        components = ["loss_property", "loss_generation", "loss_contrastive"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        steps = [r["step"] for r in train_rows]
        for ax, comp in zip(axes, components):
            if comp in train_rows[0]:
                ax.plot(steps, [r[comp] for r in train_rows])
                ax.set_title(comp)
                ax.set_xlabel("step")
                ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "loss_components.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # ---- 3. LR schedule ----
    if train_rows and "lr" in train_rows[0]:
        plt.figure(figsize=(7, 4))
        plt.plot([r["step"] for r in train_rows], [r["lr"] for r in train_rows])
        plt.xlabel("step")
        plt.ylabel("learning rate")
        plt.title("LR schedule (warmup + cosine decay)")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(args.out_dir, "lr_schedule.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved plots to {args.out_dir}/")


if __name__ == "__main__":
    main()

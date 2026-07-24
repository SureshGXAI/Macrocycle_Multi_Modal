import logging
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def save_checkpoint(path, model, optimizer, epoch, tokenizer, data_cfg, model_cfg,
                     prop_mean, prop_std, extra: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "vocab": tokenizer.vocab,
        "data_cfg": data_cfg.__dict__,
        "model_cfg": model_cfg.__dict__,
        "prop_mean": prop_mean,
        "prop_std": prop_std,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

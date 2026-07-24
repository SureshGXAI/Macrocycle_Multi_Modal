"""
Multi-task loss combining:
  1. Property regression (masked MSE — handles missing/NaN targets gracefully)
  2. SMILES generation cross-entropy (teacher forcing, ignoring PAD)
  3. Cross-modal contrastive (InfoNCE) loss that pulls each molecule's SMILES,
     graph, and 3D embeddings together and pushes apart different molecules
     in the batch — this is what actually shapes a *shared* macrocycle latent
     space rather than three independent ones.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def property_loss(pred, target):
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    diff = (pred - torch.nan_to_num(target)) ** 2
    return (diff * mask).sum() / mask.sum().clamp(min=1)


def generation_loss(logits, target_ids, pad_id: int):
    """logits: [B, L, V], target_ids: [B, L] (tokens[1:], i.e. shifted)"""
    V = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, V), target_ids.reshape(-1),
        ignore_index=pad_id,
    )


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07):
    """Symmetric InfoNCE between two batches of aligned embeddings [B, D]."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / temperature  # [B, B]
    labels = torch.arange(a.size(0), device=a.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.t(), labels)
    return (loss_ab + loss_ba) / 2.0


def cross_modal_contrastive_loss(modality_embs: dict, temperature: float = 0.07):
    pairs = [("smiles", "graph"), ("smiles", "coord"), ("graph", "coord")]
    total = 0.0
    for m1, m2 in pairs:
        total = total + info_nce(modality_embs[m1], modality_embs[m2], temperature)
    return total / len(pairs)


class MultiTaskLoss(nn.Module):
    def __init__(self, pad_id: int, w_property: float = 1.0, w_generation: float = 1.0,
                 w_contrastive: float = 0.5, contrastive_temperature: float = 0.07):
        super().__init__()
        self.pad_id = pad_id
        self.w_property = w_property
        self.w_generation = w_generation
        self.w_contrastive = w_contrastive
        self.temperature = contrastive_temperature

    def forward(self, outputs, batch):
        target_ids = batch["smiles_ids"][:, 1:]  # shifted target for generation

        l_prop = property_loss(outputs["property_pred"], batch["props"])
        l_gen = generation_loss(outputs["generation_logits"], target_ids, self.pad_id)
        l_con = cross_modal_contrastive_loss(outputs["modality_embs"], self.temperature)

        total = self.w_property * l_prop + self.w_generation * l_gen + self.w_contrastive * l_con
        return total, {
            "loss_total": total.item(),
            "loss_property": l_prop.item(),
            "loss_generation": l_gen.item(),
            "loss_contrastive": l_con.item(),
        }

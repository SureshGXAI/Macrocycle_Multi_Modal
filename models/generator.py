"""
Molecule generation branch: an autoregressive Transformer decoder that
generates SMILES token-by-token, conditioned on the fused macrocycle latent
vector (used as a length-1 memory / cross-attention context). This lets the
model be used as:
  - an encoder-decoder autoencoder (reconstruct SMILES from fused latent), or
  - a conditional generator (sample new latents, e.g. via a prior/optimization
    over the latent space, then decode to novel macrocycle SMILES).
"""
import math
import torch
import torch.nn as nn

from .smiles_encoder import PositionalEncoding


def generate_causal_mask(size: int, device) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    return mask  # True = disallowed


class SmilesDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        latent_dim: int = 256,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_len: int = 512,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.latent_to_memory = nn.Linear(latent_dim, d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, latent, tgt_ids, tgt_pad_mask=None):
        """
        latent:       [B, latent_dim]  fused macrocycle latent (conditioning)
        tgt_ids:      [B, L]           teacher-forcing input ids (shifted right)
        tgt_pad_mask: [B, L]           bool, True = PAD
        Returns logits: [B, L, vocab_size]
        """
        memory = self.latent_to_memory(latent).unsqueeze(1)  # [B, 1, d_model]

        x = self.embed(tgt_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        causal_mask = generate_causal_mask(tgt_ids.size(1), tgt_ids.device)

        h = self.decoder(
            tgt=x, memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )
        logits = self.out_proj(h)
        return logits

    @torch.no_grad()
    def generate(self, latent, tokenizer, max_len: int = 200, temperature: float = 1.0, top_k: int = 0):
        """Greedy / sampled autoregressive decoding from a fused latent vector."""
        device = latent.device
        B = latent.size(0)
        memory = self.latent_to_memory(latent).unsqueeze(1)

        ids = torch.full((B, 1), tokenizer.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            x = self.embed(ids) * math.sqrt(self.d_model)
            x = self.pos_enc(x)
            causal_mask = generate_causal_mask(ids.size(1), device)
            h = self.decoder(tgt=x, memory=memory, tgt_mask=causal_mask)
            logits = self.out_proj(h[:, -1])  # [B, vocab]

            if temperature != 1.0:
                logits = logits / max(temperature, 1e-6)
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
                probs = torch.zeros_like(logits).scatter_(-1, topk_idx, torch.softmax(topk_vals, dim=-1))
            else:
                probs = torch.softmax(logits, dim=-1)

            next_id = torch.multinomial(probs, num_samples=1)  # [B,1]
            next_id = torch.where(finished.unsqueeze(-1), torch.full_like(next_id, tokenizer.pad_id), next_id)
            ids = torch.cat([ids, next_id], dim=1)
            finished = finished | (next_id.squeeze(-1) == tokenizer.eos_id)
            if finished.all():
                break

        return ids  # [B, L] token ids, decode with tokenizer.decode(...)

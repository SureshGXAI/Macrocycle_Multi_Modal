"""
SMILES branch: standard Transformer encoder with learned positional
embeddings, followed by masked mean-pooling to a fixed-size vector.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SmilesTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        latent_dim: int = 256,
        pad_id: int = 0,
        max_len: int = 512,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, latent_dim)
        self.d_model = d_model

    def forward(self, smiles_ids: torch.Tensor, pad_mask: torch.Tensor):
        """
        smiles_ids: [B, L] token ids
        pad_mask:   [B, L] bool, True where PAD (matches nn.Transformer convention)
        Returns:
          pooled:   [B, latent_dim]   masked mean-pooled representation
          tokens:   [B, L, d_model]   full per-token hidden states (useful as
                                       cross-attention memory for the decoder)
        """
        x = self.embed(smiles_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        h = self.encoder(x, src_key_padding_mask=pad_mask)  # [B, L, d_model]

        valid = (~pad_mask).unsqueeze(-1).float()  # [B, L, 1]
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        pooled = self.out_proj(pooled)
        return pooled, h

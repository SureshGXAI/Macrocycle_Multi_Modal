"""
Fuses the three modality embeddings (SMILES-transformer, Graph-GNN, 3D-SE(3))
into a single "macrocycle latent space" vector.

Approach: treat the three pooled embeddings as a length-3 sequence, prepend a
learnable [FUSE] token, run a small Transformer encoder (cross-modal
self-attention) over them, and take the [FUSE] token's output as the fused
latent. This lets each modality attend to the others and lets the model learn
to weight modalities per-molecule (e.g. relying more on 3D shape for
macrocycle-conformer-sensitive properties).
"""
import torch
import torch.nn as nn


class ModalityFusion(nn.Module):
    def __init__(self, latent_dim: int = 256, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.fuse_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        # modality-type embeddings so the model knows which vector is which
        self.modality_embed = nn.Parameter(torch.randn(1, 4, latent_dim) * 0.02)  # fuse+3 modalities

        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=nhead, dim_feedforward=latent_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(self, smiles_emb, graph_emb, coord_emb):
        """
        smiles_emb, graph_emb, coord_emb: each [B, latent_dim]
        Returns: fused latent [B, latent_dim]
        """
        B = smiles_emb.size(0)
        fuse_tok = self.fuse_token.expand(B, -1, -1)  # [B,1,D]
        seq = torch.stack([smiles_emb, graph_emb, coord_emb], dim=1)  # [B,3,D]
        seq = torch.cat([fuse_tok, seq], dim=1)  # [B,4,D]
        seq = seq + self.modality_embed

        out = self.encoder(seq)  # [B,4,D]
        fused = self.out_norm(out[:, 0])  # take [FUSE] token
        return fused

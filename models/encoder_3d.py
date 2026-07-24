"""
3D branch: E(n)-Equivariant Graph Neural Network (EGNN, Satorras et al. 2021).
Operates on a fully-connected (or radius) graph over 3D coordinates. Node
*feature* updates are invariant to rotation/translation/reflection of the
input coordinates (coordinates only enter through pairwise distances), which
is what we want feeding into a rotation-invariant molecular latent space.

Works on padded batches [B, N, 3] + [B, N, F] with a validity mask, using
dense pairwise operations (fine for macrocycles, which are small: <=~128 atoms).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EGNNLayer(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # edge/message MLP: takes (h_i, h_j, ||x_i-x_j||^2) -> message
        self.edge_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_dim),
        )
        self.norm = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, coords, mask):
        """
        h:      [B, N, F]
        coords: [B, N, 3]
        mask:   [B, N] bool, True = valid atom
        """
        B, N, _ = h.shape
        # pairwise squared distances: [B, N, N]
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)          # [B,N,N,3]
        dist2 = (diff ** 2).sum(-1, keepdim=True)                  # [B,N,N,1]

        h_i = h.unsqueeze(2).expand(B, N, N, -1)
        h_j = h.unsqueeze(1).expand(B, N, N, -1)
        edge_input = torch.cat([h_i, h_j, dist2], dim=-1)          # [B,N,N,2F+1]
        m_ij = self.edge_mlp(edge_input)                           # [B,N,N,H]

        # mask out invalid pairs (padding atoms and self-loops)
        pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).float()  # [B,N,N]
        eye = torch.eye(N, device=h.device).unsqueeze(0)
        pair_mask = pair_mask * (1.0 - eye)
        m_ij = m_ij * pair_mask.unsqueeze(-1)

        m_i = m_ij.sum(dim=2)  # aggregate messages -> [B, N, H]

        node_input = torch.cat([h, m_i], dim=-1)
        update = self.node_mlp(node_input)
        h_new = self.norm(h + self.dropout(update))
        return h_new * mask.unsqueeze(-1).float()


class Encoder3D(nn.Module):
    def __init__(
        self,
        atom_feat_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.in_proj = nn.Linear(atom_feat_dim, hidden_dim)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, atom_feats, coords, mask):
        """
        atom_feats: [B, N, atom_feat_dim] padded
        coords:     [B, N, 3] padded
        mask:       [B, N] bool, True = valid atom
        Returns:
          pooled: [B, latent_dim]  (invariant to rotation/translation of coords)
          node_h: [B, N, hidden_dim]
        """
        # center coordinates per-molecule (translation invariance)
        m = mask.unsqueeze(-1).float()
        centroid = (coords * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp(min=1.0)
        coords = (coords - centroid) * m

        h = F.gelu(self.in_proj(atom_feats)) * m
        for layer in self.layers:
            h = layer(h, coords, mask)

        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        pooled = self.out_proj(pooled)
        return pooled, h

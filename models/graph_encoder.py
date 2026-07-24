"""
Graph branch: a dependency-free message-passing GNN (MPNN/GIN-style).
Uses torch.index_add for scatter-aggregation so it runs with plain PyTorch
(no torch_geometric / torch_scatter required).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MPNNLayer(nn.Module):
    """One round of edge-conditioned message passing + GRU-style node update."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.update_gru = nn.GRUCell(node_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feats, edge_index, edge_feats):
        """
        node_feats: [N, node_dim]
        edge_index: [2, E]  (src, dst)
        edge_feats: [E, edge_dim]
        """
        if edge_index.numel() == 0:
            return node_feats
        src, dst = edge_index[0], edge_index[1]
        msg_input = torch.cat([node_feats[src], node_feats[dst], edge_feats], dim=-1)
        messages = self.msg_mlp(msg_input)  # [E, node_dim]

        agg = torch.zeros_like(node_feats)
        agg.index_add_(0, dst, messages.to(agg.dtype))  # cast: index_add_ requires exact dtype match (breaks under AMP otherwise)

        updated = self.update_gru(agg, node_feats)
        updated = self.norm(updated)
        updated = self.dropout(updated)
        return updated


class GraphEncoder(nn.Module):
    def __init__(
        self,
        atom_feat_dim: int,
        bond_feat_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.node_in = nn.Linear(atom_feat_dim, hidden_dim)
        self.edge_in = nn.Linear(bond_feat_dim, hidden_dim)
        self.layers = nn.ModuleList([
            MPNNLayer(hidden_dim, hidden_dim, hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, atom_feats, edge_index, edge_feats, batch_idx, num_graphs: int):
        """
        atom_feats: [total_atoms, atom_feat_dim] (flat, concatenated across batch)
        edge_index: [2, total_edges]
        edge_feats: [total_edges, bond_feat_dim]
        batch_idx:  [total_atoms] which graph each atom belongs to
        Returns:
          pooled: [num_graphs, latent_dim]
          node_h: [total_atoms, hidden_dim]  (per-atom hidden states)
        """
        h = F.gelu(self.node_in(atom_feats))
        e = F.gelu(self.edge_in(edge_feats)) if edge_feats.numel() > 0 else edge_feats

        for layer in self.layers:
            h = h + layer(h, edge_index, e)

        # masked mean pool per graph via index_add
        hidden_dim = h.size(-1)
        sums = torch.zeros(num_graphs, hidden_dim, device=h.device, dtype=h.dtype)
        sums.index_add_(0, batch_idx, h.to(sums.dtype))
        counts = torch.zeros(num_graphs, device=h.device, dtype=h.dtype)
        counts.index_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=counts.dtype))
        pooled = sums / counts.clamp(min=1.0).unsqueeze(-1)
        pooled = self.out_proj(pooled)
        return pooled, h

"""
Custom collate_fn for DataLoader: pads SMILES sequences, concatenates graphs
with a batch-index vector (PyG-style, but dependency-free), and pads 3D coords.
"""
import torch


def macrocycle_collate(batch, pad_id: int = 0):
    B = len(batch)

    # ---- SMILES: pad to max length in batch ----
    smiles_lens = [b["smiles_ids"].size(0) for b in batch]
    max_smiles_len = max(smiles_lens)
    smiles_ids = torch.full((B, max_smiles_len), pad_id, dtype=torch.long)
    smiles_mask = torch.zeros((B, max_smiles_len), dtype=torch.bool)  # True = PAD
    for i, b in enumerate(batch):
        L = b["smiles_ids"].size(0)
        smiles_ids[i, :L] = b["smiles_ids"]
        smiles_mask[i, L:] = True

    # ---- Graph: concat all atoms/edges, track batch index per atom ----
    atom_feats_list, edge_index_list, edge_feats_list, batch_idx_list = [], [], [], []
    node_offset = 0
    for i, b in enumerate(batch):
        n_atoms = b["atom_feats"].size(0)
        atom_feats_list.append(b["atom_feats"])
        batch_idx_list.append(torch.full((n_atoms,), i, dtype=torch.long))
        if b["edge_index"].numel() > 0:
            edge_index_list.append(b["edge_index"] + node_offset)
            edge_feats_list.append(b["edge_feats"])
        node_offset += n_atoms

    atom_feats = torch.cat(atom_feats_list, dim=0)
    graph_batch_idx = torch.cat(batch_idx_list, dim=0)
    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_feats = torch.cat(edge_feats_list, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_feats = torch.zeros((0, batch[0]["edge_feats"].size(-1)), dtype=torch.float32)

    # ---- 3D coords: pad per-molecule to max atoms in batch, with mask ----
    max_atoms = max(b["coords"].size(0) for b in batch)
    coord_dim = batch[0]["coords"].size(-1)
    coords = torch.zeros((B, max_atoms, coord_dim), dtype=torch.float32)
    coords_mask = torch.zeros((B, max_atoms), dtype=torch.bool)  # True = valid atom
    for i, b in enumerate(batch):
        n = b["coords"].size(0)
        coords[i, :n] = b["coords"]
        coords_mask[i, :n] = True

    # ---- Also provide padded per-molecule atom features (for the 3D encoder,
    # which needs aligned [B, N, F] atom features alongside coords) ----
    atom_feat_dim = batch[0]["atom_feats"].size(-1)
    atom_feats_padded = torch.zeros((B, max_atoms, atom_feat_dim), dtype=torch.float32)
    for i, b in enumerate(batch):
        n = b["atom_feats"].size(0)
        atom_feats_padded[i, :n] = b["atom_feats"]

    props = torch.stack([b["props"] for b in batch], dim=0)
    smiles_strs = [b["smiles_str"] for b in batch]

    return {
        "smiles_ids": smiles_ids,
        "smiles_pad_mask": smiles_mask,
        "atom_feats": atom_feats,              # [total_atoms, F] flat, for GNN
        "edge_index": edge_index,              # [2, total_edges]
        "edge_feats": edge_feats,              # [total_edges, B_feat]
        "graph_batch_idx": graph_batch_idx,    # [total_atoms] -> which molecule
        "coords": coords,                      # [B, N_max, 3] padded
        "coords_atom_feats": atom_feats_padded,  # [B, N_max, F] padded (aligned w/ coords)
        "coords_mask": coords_mask,            # [B, N_max] True=valid
        "props": props,                        # [B, num_props]
        "smiles_strs": smiles_strs,
        "batch_size": B,
    }

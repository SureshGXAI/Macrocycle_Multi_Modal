"""
Loads a 50K-molecule SDF file of macrocycles and produces, per molecule:
  - canonical SMILES token ids
  - graph (atom feats, edge_index, edge feats)
  - 3D coordinates (from SDF conformer, or embedded if absent)
  - property target vector
"""
import logging
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem, RDLogger

from .featurize import mol_to_graph, get_3d_coords, compute_property_vector
from .tokenizer import SmilesTokenizer

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


class MacrocycleRecord:
    __slots__ = ["smiles", "atom_feats", "edge_index", "edge_feats", "coords", "props"]

    def __init__(self, smiles, atom_feats, edge_index, edge_feats, coords, props):
        self.smiles = smiles
        self.atom_feats = atom_feats
        self.edge_index = edge_index
        self.edge_feats = edge_feats
        self.coords = coords
        self.props = props


def load_sdf_records(
    sdf_path: str,
    property_fields: List[str],
    max_atoms: int = 128,
    limit: Optional[int] = None,
) -> List[MacrocycleRecord]:
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
    records = []
    n_seen, n_kept = 0, 0
    for mol in supplier:
        n_seen += 1
        if limit is not None and n_kept >= limit:
            break
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        if mol.GetNumAtoms() == 0 or mol.GetNumAtoms() > max_atoms:
            continue
        try:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            atom_feats, edge_index, edge_feats = mol_to_graph(mol)
            coords = get_3d_coords(mol)
            if coords.shape[0] != atom_feats.shape[0]:
                continue
            props = compute_property_vector(mol, property_fields)
        except Exception as e:
            logger.debug(f"Skipping molecule {n_seen}: {e}")
            continue

        records.append(MacrocycleRecord(smiles, atom_feats, edge_index, edge_feats, coords, props))
        n_kept += 1
        if n_seen % 5000 == 0:
            logger.info(f"Processed {n_seen} SDF records, kept {n_kept}")

    logger.info(f"Finished loading SDF: {n_seen} seen, {n_kept} kept -> {sdf_path}")
    return records


class MacrocycleDataset(Dataset):
    """
    Wraps a list of MacrocycleRecord and tokenizes SMILES on the fly with a
    fitted SmilesTokenizer. Property targets are optionally standardized
    (mean/std) — pass in precomputed stats for val/test splits so there's no
    leakage from train statistics.
    """

    def __init__(
        self,
        records: List[MacrocycleRecord],
        tokenizer: SmilesTokenizer,
        max_smiles_len: int = 200,
        prop_mean: Optional[np.ndarray] = None,
        prop_std: Optional[np.ndarray] = None,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_smiles_len = max_smiles_len

        all_props = np.stack([r.props for r in records]) if records else np.zeros((0, 1))
        self.prop_mean = prop_mean if prop_mean is not None else all_props.mean(axis=0)
        self.prop_std = prop_std if prop_std is not None else all_props.std(axis=0) + 1e-6

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        smiles_ids = self.tokenizer.encode(rec.smiles, max_len=self.max_smiles_len)
        norm_props = (rec.props - self.prop_mean) / self.prop_std
        return {
            "smiles_ids": torch.tensor(smiles_ids, dtype=torch.long),
            "atom_feats": torch.tensor(rec.atom_feats, dtype=torch.float32),
            "edge_index": torch.tensor(rec.edge_index, dtype=torch.long),
            "edge_feats": torch.tensor(rec.edge_feats, dtype=torch.float32),
            "coords": torch.tensor(rec.coords, dtype=torch.float32),
            "props": torch.tensor(norm_props, dtype=torch.float32),
            "smiles_str": rec.smiles,
        }


def split_records(records: List[MacrocycleRecord], val_frac: float, test_frac: float, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n_val = int(len(records) * val_frac)
    n_test = int(len(records) * test_frac)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]
    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    test = [records[i] for i in test_idx]
    return train, val, test

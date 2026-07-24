"""
Predict properties for new macrocycles (given as SMILES or an SDF file) using
a trained checkpoint.

Usage:
    python predict.py --ckpt checkpoints/macrocycle_epoch99.pt --smiles "C1CCCCCCCCCC1" "C1CCOCCOCCOCC1"
    python predict.py --ckpt checkpoints/macrocycle_epoch99.pt --sdf_path new_molecules.sdf
"""
import argparse
import logging

import torch
from rdkit import Chem

from config import DataConfig, ModelConfig
from data.tokenizer import SmilesTokenizer
from data.featurize import mol_to_graph, get_3d_coords
from data.collate import macrocycle_collate
from models.model import MacrocycleModel
from utils import setup_logging

logger = logging.getLogger("predict")


def records_from_smiles(smiles_list):
    recs = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning(f"Could not parse SMILES: {smi}")
            continue
        atom_feats, edge_index, edge_feats = mol_to_graph(mol)
        coords = get_3d_coords(mol)
        recs.append({"smiles": Chem.MolToSmiles(mol), "atom_feats": atom_feats,
                     "edge_index": edge_index, "edge_feats": edge_feats, "coords": coords})
    return recs


def build_batch(recs, tokenizer, max_len, pad_id):
    import numpy as np
    items = []
    for r in recs:
        ids = tokenizer.encode(r["smiles"], max_len=max_len)
        items.append({
            "smiles_ids": torch.tensor(ids, dtype=torch.long),
            "atom_feats": torch.tensor(r["atom_feats"], dtype=torch.float32),
            "edge_index": torch.tensor(r["edge_index"], dtype=torch.long),
            "edge_feats": torch.tensor(r["edge_feats"], dtype=torch.float32),
            "coords": torch.tensor(r["coords"], dtype=torch.float32),
            "props": torch.zeros(1, dtype=torch.float32),  # unused at inference
            "smiles_str": r["smiles"],
        })
    return macrocycle_collate(items, pad_id=pad_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--smiles", type=str, nargs="*", default=None)
    parser.add_argument("--sdf_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    setup_logging()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)

    tokenizer = SmilesTokenizer(ckpt["vocab"])
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    data_cfg = DataConfig(**ckpt["data_cfg"])

    model = MacrocycleModel(model_cfg, pad_id=tokenizer.pad_id).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    prop_mean = ckpt["prop_mean"]
    prop_std = ckpt["prop_std"]

    if args.smiles:
        recs = records_from_smiles(args.smiles)
    elif args.sdf_path:
        from data.dataset import load_sdf_records
        raw = load_sdf_records(args.sdf_path, data_cfg.property_fields, max_atoms=data_cfg.max_atoms)
        recs = [{"smiles": r.smiles, "atom_feats": r.atom_feats, "edge_index": r.edge_index,
                 "edge_feats": r.edge_feats, "coords": r.coords} for r in raw]
    else:
        raise ValueError("Provide --smiles or --sdf_path")

    if not recs:
        logger.warning("No valid molecules to predict on.")
        return

    batch = build_batch(recs, tokenizer, data_cfg.max_smiles_len, tokenizer.pad_id)
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    with torch.no_grad():
        preds_std = model.predict_properties(batch).cpu().numpy()
    preds = preds_std * prop_std + prop_mean  # de-standardize

    print(f"\n{'SMILES':<50s} " + " ".join(f"{f:>14s}" for f in data_cfg.property_fields))
    for smi, p in zip(batch["smiles_strs"], preds):
        vals = " ".join(f"{v:14.3f}" for v in p)
        print(f"{smi:<50s} {vals}")


if __name__ == "__main__":
    main()

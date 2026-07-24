"""
Comprehensive quality assessment for a trained MacrocycleModel checkpoint,
on a held-out test split. Reports metrics along the three axes that matter
for a multi-modal, multi-task molecular model:

  A) PROPERTY PREDICTION  — is the regression head actually accurate?
     MAE / RMSE / R^2 per property, vs. a mean-predictor baseline.

  B) GENERATION QUALITY    — can the decoder turn latents back into real,
     diverse, non-trivial molecules?
     - Reconstruction: exact-match rate + Tanimoto similarity (encode a real
       molecule -> fused latent -> decode, compare to the original)
     - Prior sampling: validity, uniqueness, novelty (vs. training set),
       internal diversity, and how well generated-molecule property
       distributions match the training distribution (Wasserstein distance)
     - Macrocycle-specific: fraction of generated molecules that actually
       contain a macrocycle (largest ring >= 12 atoms), since a model can
       cheat by generating easy small-ring-only molecules.

  C) LATENT SPACE STRUCTURE — did fusion actually build one coherent space?
     - Cross-modal alignment: cosine similarity of SMILES/graph/3D embeddings
       for the *same* molecule vs. different molecules (this should be high
       vs. low; a collapsed/uninformative space will show both near equal).
     - Modality ablation: property-prediction R^2 using fused latent vs.
       using only one modality's pooled embedding — quantifies whether
       multi-modal fusion is actually adding value over the best single view.

Usage:
    python eval.py --ckpt checkpoints/macrocycle_epoch99.pt --sdf_path data/macrocycles_50k.sdf
"""
import argparse
import logging
from collections import defaultdict

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.DataStructs import TanimotoSimilarity

from config import DataConfig, ModelConfig
from data.dataset import load_sdf_records, split_records, MacrocycleDataset
from data.tokenizer import SmilesTokenizer
from data.collate import macrocycle_collate
from data.featurize import largest_ring_size
from models.model import MacrocycleModel
from models.property_head import PropertyPredictionHead
from utils import setup_logging

logger = logging.getLogger("eval")


# ----------------------------------------------------------------------------
# A) Property prediction metrics
# ----------------------------------------------------------------------------
def property_metrics(preds: np.ndarray, targets: np.ndarray, field_names, train_mean: np.ndarray):
    """MAE / RMSE / R^2 per property, plus R^2 of a naive mean-predictor
    baseline so you can tell if the model is beating "always predict the
    training mean" (a shockingly common failure mode for weak regressors)."""
    results = {}
    for i, name in enumerate(field_names):
        p, t = preds[:, i], targets[:, i]
        mae = np.mean(np.abs(p - t))
        rmse = np.sqrt(np.mean((p - t) ** 2))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2) + 1e-8
        r2 = 1 - ss_res / ss_tot

        baseline_pred = np.full_like(t, train_mean[i])
        ss_res_base = np.sum((t - baseline_pred) ** 2)
        r2_baseline = 1 - ss_res_base / ss_tot

        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "R2_mean_baseline": r2_baseline}
    return results


# ----------------------------------------------------------------------------
# B) Generation quality metrics
# ----------------------------------------------------------------------------
def morgan_fp(mol, radius=2, nbits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def tanimoto(smi_a, smi_b):
    mol_a, mol_b = Chem.MolFromSmiles(smi_a), Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return 0.0
    return TanimotoSimilarity(morgan_fp(mol_a), morgan_fp(mol_b))


def reconstruction_metrics(model, tokenizer, test_recs, data_cfg, device, n_samples=200, temperature=0.7):
    sample = test_recs[:n_samples]
    items = []
    for r in sample:
        ids = tokenizer.encode(r.smiles, max_len=data_cfg.max_smiles_len)
        items.append({
            "smiles_ids": torch.tensor(ids, dtype=torch.long),
            "atom_feats": torch.tensor(r.atom_feats, dtype=torch.float32),
            "edge_index": torch.tensor(r.edge_index, dtype=torch.long),
            "edge_feats": torch.tensor(r.edge_feats, dtype=torch.float32),
            "coords": torch.tensor(r.coords, dtype=torch.float32),
            "props": torch.tensor(r.props, dtype=torch.float32),
            "smiles_str": r.smiles,
        })
    batch = macrocycle_collate(items, pad_id=tokenizer.pad_id)
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    ids = model.generate_from_batch(batch, tokenizer, max_len=data_cfg.max_smiles_len, temperature=temperature)
    recon_smiles = [tokenizer.decode(row.tolist()) for row in ids]

    exact_matches, tanimotos, valid = 0, [], 0
    for orig, recon in zip(batch["smiles_strs"], recon_smiles):
        recon_mol = Chem.MolFromSmiles(recon)
        if recon_mol is None:
            continue
        valid += 1
        canon_recon = Chem.MolToSmiles(recon_mol)
        if canon_recon == orig:
            exact_matches += 1
        tanimotos.append(tanimoto(orig, recon))

    n = len(sample)
    return {
        "recon_validity_rate": valid / n,
        "recon_exact_match_rate": exact_matches / n,
        "recon_mean_tanimoto": float(np.mean(tanimotos)) if tanimotos else 0.0,
    }


def prior_sampling_metrics(model, tokenizer, train_smiles_set, train_props_ref, data_cfg,
                            model_cfg, device, n_samples=500, temperature=1.0):
    latent = torch.randn(n_samples, model_cfg.latent_dim, device=device)
    ids = model.generate_from_latent(latent, tokenizer, max_len=data_cfg.max_smiles_len, temperature=temperature)
    gen_smiles_raw = [tokenizer.decode(row.tolist()) for row in ids]

    valid_mols, valid_canon = [], []
    for s in gen_smiles_raw:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            valid_mols.append(mol)
            valid_canon.append(Chem.MolToSmiles(mol))

    validity = len(valid_mols) / n_samples
    unique_smiles = set(valid_canon)
    uniqueness = len(unique_smiles) / max(1, len(valid_canon))
    novelty = len(unique_smiles - train_smiles_set) / max(1, len(unique_smiles))

    # internal diversity: mean pairwise Tanimoto distance among a subsample
    div_sample = list(unique_smiles)[:100]
    fps = [morgan_fp(Chem.MolFromSmiles(s)) for s in div_sample]
    dists = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            dists.append(1 - TanimotoSimilarity(fps[i], fps[j]))
    internal_diversity = float(np.mean(dists)) if dists else 0.0

    # macrocycle-specific: did it actually keep generating macrocycles (largest ring >= 12)?
    macrocycle_frac = np.mean([largest_ring_size(m) >= 12 for m in valid_mols]) if valid_mols else 0.0

    # property-distribution match vs. training set (Wasserstein / earth-mover distance per property)
    from data.featurize import compute_property_vector
    gen_props = np.stack([compute_property_vector(m, data_cfg.property_fields) for m in valid_mols]) \
        if valid_mols else np.zeros((0, len(data_cfg.property_fields)))
    wasserstein = {}
    if len(gen_props) > 0:
        from scipy.stats import wasserstein_distance
        for i, field in enumerate(data_cfg.property_fields):
            wasserstein[field] = wasserstein_distance(gen_props[:, i], train_props_ref[:, i])

    return {
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "internal_diversity": internal_diversity,
        "macrocycle_fraction": macrocycle_frac,
        "property_wasserstein_distance": wasserstein,
    }


# ----------------------------------------------------------------------------
# C) Latent space structure
# ----------------------------------------------------------------------------
def cross_modal_alignment(model, loader, device, max_batches=20):
    """Cosine sim of SMILES/graph/3D embeddings for the same molecule
    (diagonal) vs. different molecules (off-diagonal). A well-aligned space
    has diagonal >> off-diagonal; a collapsed/degenerate space has both similar."""
    diag_sims = defaultdict(list)
    offdiag_sims = defaultdict(list)
    pairs = [("smiles", "graph"), ("smiles", "coord"), ("graph", "coord")]

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            _, embs = model.encode(batch)
            for m1, m2 in pairs:
                a = torch.nn.functional.normalize(embs[m1], dim=-1)
                b = torch.nn.functional.normalize(embs[m2], dim=-1)
                sim = a @ b.t()  # [B, B]
                diag_sims[f"{m1}-{m2}"].append(sim.diag().mean().item())
                off_mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
                offdiag_sims[f"{m1}-{m2}"].append(sim[off_mask].mean().item())

    return {
        pair: {"same_molecule_sim": float(np.mean(diag_sims[pair])),
               "different_molecule_sim": float(np.mean(offdiag_sims[pair])),
               "gap": float(np.mean(diag_sims[pair]) - np.mean(offdiag_sims[pair]))}
        for pair in diag_sims
    }


def modality_ablation_r2(model, loader, device, prop_mean, prop_std, model_cfg, num_properties, train_props_std_targets=None):
    """Trains a quick linear-probe-equivalent (reuses model's own property
    head weights applied to each modality's pooled embedding won't work
    dimension-wise across different encoders trivially, so instead we report
    R^2 of the *fused* latent's prediction, and separately fit tiny ridge
    regressions from each raw modality embedding to the same targets, for a
    fair single-vs-fused comparison)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    fused_embs, smiles_embs, graph_embs, coord_embs, targets = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            fused, embs = model.encode(batch)
            fused_embs.append(fused.cpu().numpy())
            smiles_embs.append(embs["smiles"].cpu().numpy())
            graph_embs.append(embs["graph"].cpu().numpy())
            coord_embs.append(embs["coord"].cpu().numpy())
            targets.append(batch["props"].cpu().numpy())

    fused_embs = np.concatenate(fused_embs)
    smiles_embs = np.concatenate(smiles_embs)
    graph_embs = np.concatenate(graph_embs)
    coord_embs = np.concatenate(coord_embs)
    targets = np.concatenate(targets)

    def cv_r2(X, y):
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        r2s = []
        for tr_idx, te_idx in kf.split(X):
            reg = Ridge(alpha=1.0).fit(X[tr_idx], y[tr_idx])
            pred = reg.predict(X[te_idx])
            ss_res = np.sum((y[te_idx] - pred) ** 2)
            ss_tot = np.sum((y[te_idx] - y[te_idx].mean(axis=0)) ** 2) + 1e-8
            r2s.append(1 - ss_res / ss_tot)
        return float(np.mean(r2s))

    return {
        "fused_latent_r2": cv_r2(fused_embs, targets),
        "smiles_only_r2": cv_r2(smiles_embs, targets),
        "graph_only_r2": cv_r2(graph_embs, targets),
        "coord_3d_only_r2": cv_r2(coord_embs, targets),
    }


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--sdf_path", type=str, required=True, help="Full SDF (same one used for training) to rebuild the held-out test split")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_ablation", action="store_true", help="Skip the sklearn ridge-regression ablation (slower)")
    args = parser.parse_args()
    setup_logging()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    tokenizer = SmilesTokenizer(ckpt["vocab"])
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    data_cfg = DataConfig(**ckpt["data_cfg"])
    data_cfg.sdf_path = args.sdf_path
    prop_mean, prop_std = ckpt["prop_mean"], ckpt["prop_std"]

    model = MacrocycleModel(model_cfg, pad_id=tokenizer.pad_id).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info("Reloading SDF and rebuilding the same train/val/test split used at training time...")
    records = load_sdf_records(data_cfg.sdf_path, data_cfg.property_fields, max_atoms=data_cfg.max_atoms)
    train_recs, val_recs, test_recs = split_records(records, data_cfg.val_fraction, data_cfg.test_fraction, seed=data_cfg.seed)
    logger.info(f"Test set size: {len(test_recs)}")

    test_ds = MacrocycleDataset(test_recs, tokenizer, data_cfg.max_smiles_len, prop_mean=prop_mean, prop_std=prop_std)
    collate = lambda b: macrocycle_collate(b, pad_id=tokenizer.pad_id)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate)

    # ==================== A) Property prediction ====================
    logger.info("=" * 70)
    logger.info("A) PROPERTY PREDICTION QUALITY")
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            pred_std = model.predict_properties(batch).cpu().numpy()
            all_preds.append(pred_std * prop_std + prop_mean)
            all_targets.append(batch["props"].cpu().numpy() * prop_std + prop_mean)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    prop_results = property_metrics(all_preds, all_targets, data_cfg.property_fields, prop_mean)
    for name, m in prop_results.items():
        logger.info(f"  {name:<16s} MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
                     f"R2={m['R2']:.4f}  (mean-baseline R2={m['R2_mean_baseline']:.4f})")

    # ==================== B) Generation quality ====================
    logger.info("=" * 70)
    logger.info("B) GENERATION QUALITY")
    recon = reconstruction_metrics(model, tokenizer, test_recs, data_cfg, device)
    logger.info(f"  Reconstruction -> validity={recon['recon_validity_rate']:.3f} "
                 f"exact_match={recon['recon_exact_match_rate']:.3f} "
                 f"mean_tanimoto={recon['recon_mean_tanimoto']:.3f}")

    train_smiles_set = set(r.smiles for r in train_recs)
    train_props_ref = np.stack([r.props for r in train_recs])
    sampling = prior_sampling_metrics(model, tokenizer, train_smiles_set, train_props_ref,
                                       data_cfg, model_cfg, device)
    logger.info(f"  Prior sampling -> validity={sampling['validity']:.3f} "
                 f"uniqueness={sampling['uniqueness']:.3f} novelty={sampling['novelty']:.3f} "
                 f"internal_diversity={sampling['internal_diversity']:.3f} "
                 f"macrocycle_fraction={sampling['macrocycle_fraction']:.3f}")
    for field, w in sampling["property_wasserstein_distance"].items():
        logger.info(f"    property dist. shift ({field}): W-distance={w:.4f} (lower = closer to training distribution)")

    # ==================== C) Latent space structure ====================
    logger.info("=" * 70)
    logger.info("C) LATENT SPACE STRUCTURE")
    alignment = cross_modal_alignment(model, test_loader, device)
    for pair, m in alignment.items():
        logger.info(f"  {pair:<14s} same-molecule sim={m['same_molecule_sim']:.3f}  "
                     f"different-molecule sim={m['different_molecule_sim']:.3f}  gap={m['gap']:.3f}")

    if not args.skip_ablation:
        logger.info("  Running modality ablation (5-fold ridge regression probe)...")
        ablation = modality_ablation_r2(model, test_loader, device, prop_mean, prop_std,
                                         model_cfg, len(data_cfg.property_fields))
        logger.info(f"  Fused latent R2:   {ablation['fused_latent_r2']:.4f}")
        logger.info(f"  SMILES-only R2:    {ablation['smiles_only_r2']:.4f}")
        logger.info(f"  Graph-only R2:     {ablation['graph_only_r2']:.4f}")
        logger.info(f"  3D-only R2:        {ablation['coord_3d_only_r2']:.4f}")
        logger.info("  (fused should be >= max(single-modality) — if not, fusion isn't adding value)")

    logger.info("=" * 70)
    logger.info("Done.")


if __name__ == "__main__":
    main()

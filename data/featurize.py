"""
RDKit featurization utilities: atom features, bond features, 3D coordinate
extraction, and macrocycle-specific descriptors (largest ring size, etc.).
"""
from typing import List, Tuple
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

ATOM_LIST = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si", "B", "Se", "H", "Other"]
HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
    Chem.HybridizationType.OTHER,
]
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]


def _one_hot(value, choices) -> List[float]:
    vec = [0.0] * (len(choices) + 1)  # last slot = "other"/unknown
    if value in choices:
        vec[choices.index(value)] = 1.0
    else:
        vec[-1] = 1.0
    return vec


def atom_features(atom: Chem.Atom) -> np.ndarray:
    symbol = atom.GetSymbol()
    feats = []
    feats += _one_hot(symbol, ATOM_LIST)                       # 15
    feats += _one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])     # 7
    feats += _one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])  # 6
    feats += _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)  # 7
    feats += [
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.IsInRing() else 0.0,
        atom.GetTotalNumHs() / 4.0,
        atom.GetMass() / 100.0,
    ]
    # macrocycle ring-size membership (size of the largest ring this atom sits in)
    ri = atom.GetOwningMol().GetRingInfo()
    max_ring = 0
    for ring in ri.AtomRings():
        if atom.GetIdx() in ring:
            max_ring = max(max_ring, len(ring))
    feats += [max_ring / 20.0]
    return np.array(feats, dtype=np.float32)  # dim = 15+7+6+7+4+1 = 40 (+4 pad -> handled by config default 44)


def bond_features(bond: Chem.Bond) -> np.ndarray:
    feats = []
    feats += _one_hot(bond.GetBondType(), BOND_TYPES)  # 5
    feats += [
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.IsInRing() else 0.0,
        1.0 if bond.GetStereo() != Chem.BondStereo.STEREONONE else 0.0,
    ]
    return np.array(feats, dtype=np.float32)  # dim = 5+3 = 8


def mol_to_graph(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (atom_feats [N,F], edge_index [2,E], edge_feats [E,B])."""
    atom_feats = np.stack([atom_features(a) for a in mol.GetAtoms()])
    edges, edge_feats = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edges += [(i, j), (j, i)]
        edge_feats += [bf, bf]
    if len(edges) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_feats = np.zeros((0, 8), dtype=np.float32)
    else:
        edge_index = np.array(edges, dtype=np.int64).T
        edge_feats = np.stack(edge_feats)
    return atom_feats, edge_index, edge_feats


def get_3d_coords(mol: Chem.Mol) -> np.ndarray:
    """Extract 3D coords if present; else embed with ETKDG + MMFF optimize."""
    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC0FFEE
        cid = AllChem.EmbedMolecule(mol, params)
        if cid == -1:
            # fallback: random coordinates
            n = mol.GetNumAtoms()
            return np.random.randn(n, 3).astype(np.float32) * 0.1
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
        mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    coords = conf.GetPositions().astype(np.float32)
    return coords


def largest_ring_size(mol: Chem.Mol) -> int:
    ri = mol.GetRingInfo()
    sizes = [len(r) for r in ri.AtomRings()]
    return max(sizes) if sizes else 0


def compute_property_vector(mol: Chem.Mol, fields: List[str]) -> np.ndarray:
    """Read requested property fields from SDF props if present, else compute
    with RDKit descriptors. 'MacrocycleSize' always computed from ring info."""
    values = []
    for field in fields:
        if field == "MacrocycleSize":
            values.append(float(largest_ring_size(mol)))
            continue
        if mol.HasProp(field):
            try:
                values.append(float(mol.GetProp(field)))
                continue
            except ValueError:
                pass
        # fallback to RDKit descriptors
        if field == "MolWt":
            values.append(Descriptors.MolWt(mol))
        elif field == "LogP":
            values.append(Descriptors.MolLogP(mol))
        elif field == "TPSA":
            values.append(Descriptors.TPSA(mol))
        elif field == "RingCount":
            values.append(float(rdMolDescriptors.CalcNumRings(mol)))
        else:
            values.append(0.0)
    return np.array(values, dtype=np.float32)

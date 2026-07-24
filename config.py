"""
Central configuration for the macrocycle multi-modal model.
Edit these values or override via CLI flags in train.py / generate.py / predict.py.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    sdf_path: str = "data/curated_macrocycles.sdf"
    max_smiles_len: int = 200          # max SMILES token length (incl. BOS/EOS)
    max_atoms: int = 128               # max atoms per macrocycle (pad/truncate graphs)
    val_fraction: float = 0.1
    test_fraction: float = 0.05
    # SDF property fields to use as prediction targets. If a molecule is missing
    # a field, it is computed on the fly with RDKit descriptors (see data/dataset.py).
    property_fields: List[str] = field(default_factory=lambda: [
        "MolWt", "LogP", "TPSA", "RingCount", "MacrocycleSize"
    ])
    num_workers: int = 4
    seed: int = 42


@dataclass
class ModelConfig:
    # Shared latent space
    latent_dim: int = 256

    # SMILES transformer encoder
    smiles_vocab_size: int = 128       # set from tokenizer at runtime
    smiles_d_model: int = 256
    smiles_nhead: int = 8
    smiles_num_layers: int = 4
    smiles_dim_ff: int = 1024
    smiles_dropout: float = 0.1

    # Graph (GNN) encoder
    graph_atom_feat_dim: int = 44      # set from featurizer at runtime
    graph_bond_feat_dim: int = 12      # set from featurizer at runtime
    graph_hidden_dim: int = 256
    graph_num_layers: int = 4
    graph_dropout: float = 0.1

    # 3D E(n)-equivariant encoder
    coord_hidden_dim: int = 256
    coord_num_layers: int = 4
    coord_dropout: float = 0.1

    # Fusion
    fusion_nhead: int = 8
    fusion_num_layers: int = 2
    fusion_dropout: float = 0.1

    # Property prediction head
    num_properties: int = 5            # must match len(DataConfig.property_fields)
    property_hidden_dim: int = 256

    # Generation (SMILES decoder)
    decoder_num_layers: int = 4
    decoder_nhead: int = 8
    decoder_dim_ff: int = 1024
    decoder_dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    # multi-task loss weights
    w_property: float = 1.0
    w_generation: float = 1.0
    w_contrastive: float = 0.5
    contrastive_temperature: float = 0.07
    log_every: int = 50
    ckpt_dir: str = "checkpoints"
    device: str = "cuda"
    amp: bool = True
    save_every_epochs: int = 5

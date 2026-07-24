"""
MacrocycleModel: wires together
  SMILES Transformer Encoder ─┐
  Graph Encoder (GNN)         ├─► ModalityFusion ─► macrocycle latent ─┬─► PropertyPredictionHead
  3D Encoder (SE(3)/EGNN)    ─┘                                        └─► SmilesDecoder (generation)
"""
import torch
import torch.nn as nn

from .smiles_encoder import SmilesTransformerEncoder
from .graph_encoder import GraphEncoder
from .encoder_3d import Encoder3D
from .fusion import ModalityFusion
from .property_head import PropertyPredictionHead
from .generator import SmilesDecoder


class MacrocycleModel(nn.Module):
    def __init__(self, cfg, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg

        self.smiles_encoder = SmilesTransformerEncoder(
            vocab_size=cfg.smiles_vocab_size, d_model=cfg.smiles_d_model,
            nhead=cfg.smiles_nhead, num_layers=cfg.smiles_num_layers,
            dim_ff=cfg.smiles_dim_ff, dropout=cfg.smiles_dropout,
            latent_dim=cfg.latent_dim, pad_id=pad_id,
        )
        self.graph_encoder = GraphEncoder(
            atom_feat_dim=cfg.graph_atom_feat_dim, bond_feat_dim=cfg.graph_bond_feat_dim,
            hidden_dim=cfg.graph_hidden_dim, num_layers=cfg.graph_num_layers,
            dropout=cfg.graph_dropout, latent_dim=cfg.latent_dim,
        )
        self.encoder_3d = Encoder3D(
            atom_feat_dim=cfg.graph_atom_feat_dim, hidden_dim=cfg.coord_hidden_dim,
            num_layers=cfg.coord_num_layers, dropout=cfg.coord_dropout,
            latent_dim=cfg.latent_dim,
        )
        self.fusion = ModalityFusion(
            latent_dim=cfg.latent_dim, nhead=cfg.fusion_nhead,
            num_layers=cfg.fusion_num_layers, dropout=cfg.fusion_dropout,
        )
        self.property_head = PropertyPredictionHead(
            latent_dim=cfg.latent_dim, hidden_dim=cfg.property_hidden_dim,
            num_properties=cfg.num_properties,
        )
        self.decoder = SmilesDecoder(
            vocab_size=cfg.smiles_vocab_size, latent_dim=cfg.latent_dim,
            d_model=cfg.smiles_d_model, nhead=cfg.decoder_nhead,
            num_layers=cfg.decoder_num_layers, dim_ff=cfg.decoder_dim_ff,
            dropout=cfg.decoder_dropout, pad_id=pad_id,
        )

    def encode(self, batch):
        """Run all three modality encoders + fusion. Returns fused latent and
        the individual pooled embeddings (needed for the contrastive loss)."""
        smiles_emb, _ = self.smiles_encoder(batch["smiles_ids"], batch["smiles_pad_mask"])
        graph_emb, _ = self.graph_encoder(
            batch["atom_feats"], batch["edge_index"], batch["edge_feats"],
            batch["graph_batch_idx"], num_graphs=batch["batch_size"],
        )
        coord_emb, _ = self.encoder_3d(batch["coords_atom_feats"], batch["coords"], batch["coords_mask"])

        fused = self.fusion(smiles_emb, graph_emb, coord_emb)
        return fused, {"smiles": smiles_emb, "graph": graph_emb, "coord": coord_emb}

    def forward(self, batch):
        fused, modality_embs = self.encode(batch)

        prop_pred = self.property_head(fused)

        # teacher forcing: input = tokens[:-1], target = tokens[1:]
        dec_input = batch["smiles_ids"][:, :-1]
        dec_pad_mask = batch["smiles_pad_mask"][:, :-1]
        gen_logits = self.decoder(fused, dec_input, dec_pad_mask)

        return {
            "fused_latent": fused,
            "modality_embs": modality_embs,
            "property_pred": prop_pred,
            "generation_logits": gen_logits,
        }

    @torch.no_grad()
    def predict_properties(self, batch):
        fused, _ = self.encode(batch)
        return self.property_head(fused)

    @torch.no_grad()
    def generate_from_batch(self, batch, tokenizer, max_len=200, temperature=1.0, top_k=0):
        fused, _ = self.encode(batch)
        return self.decoder.generate(fused, tokenizer, max_len=max_len, temperature=temperature, top_k=top_k)

    @torch.no_grad()
    def generate_from_latent(self, latent, tokenizer, max_len=200, temperature=1.0, top_k=0):
        return self.decoder.generate(latent, tokenizer, max_len=max_len, temperature=temperature, top_k=top_k)

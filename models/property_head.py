import torch.nn as nn


class PropertyPredictionHead(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 256, num_properties: int = 5, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_properties),
        )

    def forward(self, latent):
        return self.net(latent)  # [B, num_properties], standardized-scale predictions

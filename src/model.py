"""
GIN (Graph Isomorphism Network) model for Bitcoin Heist Ransomware Detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, BatchNorm


class GINLayer(nn.Module):
    """Single GIN convolution layer with BatchNorm + ReLU."""

    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.0):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.BatchNorm1d(out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.conv = GINConv(mlp, eps=eps, train_eps=True)
        self.bn = BatchNorm(out_dim)

    def forward(self, x, edge_index):
        return F.relu(self.bn(self.conv(x, edge_index)))


class GINClassifier(nn.Module):
    """
    Node-classification GIN with Jumping Knowledge (JK) concatenation.

    Architecture:
        Input → [GIN × num_layers] → Concat all layer outputs → MLP → num_classes

    Args:
        in_dim (int): Number of input node features.
        hidden_dim (int): Hidden dimension for GIN layers.
        num_layers (int): Number of GIN layers.
        num_classes (int): Number of output classes (2 for binary classification).
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.dropout = dropout

        self.gin_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gin_layers.append(
                GINLayer(in_dim if i == 0 else hidden_dim, hidden_dim)
            )

        # JK-concat: concatenate ALL layer outputs
        concat_dim = hidden_dim * num_layers
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index):
        embeddings = []
        h = x
        for layer in self.gin_layers:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = layer(h, edge_index)
            embeddings.append(h)

        # Jumping Knowledge (JK) concatenation across all layers
        out = torch.cat(embeddings, dim=-1)
        return self.mlp(out)  # logits


class EmbeddingExtractor(nn.Module):
    """Extracts JK-concatenated node embeddings from a trained GINClassifier."""

    def __init__(self, gin_model: GINClassifier):
        super().__init__()
        self.gin_layers = gin_model.gin_layers
        self.dropout = gin_model.dropout

    def forward(self, x, edge_index):
        embeddings = []
        h = x
        for layer in self.gin_layers:
            h = F.dropout(h, p=self.dropout, training=False)
            h = layer(h, edge_index)
            embeddings.append(h)
        return torch.cat(embeddings, dim=-1)

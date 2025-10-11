"""
Graph Convolutional Network for earthquake magnitude prediction.
Uses spectral convolutions to aggregate neighborhood information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class EarthquakeGCN(nn.Module):
    """
    GCN model for earthquake magnitude prediction with residual connections.

    Architecture:
    - Multiple GCN layers with layer normalization and residual connections
    - ReLU activation and dropout
    - MLP regression head

    Args:
        num_features: Number of input node features
        hidden_dim: Hidden dimension size
        num_layers: Number of GCN layers
        dropout: Dropout rate
    """

    def __init__(self, num_features, hidden_dim=64, num_layers=2, dropout=0.2, output_dim=1):
        super(EarthquakeGCN, self).__init__()

        self.num_layers = num_layers
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)

        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # GCN layers with residual connections
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Output head (MLP) - supports both regression (output_dim=1) and classification (output_dim=2)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

        self.dropout = dropout

    def forward(self, data):
        """
        Forward pass with residual connections.

        Args:
            data: PyG Data object with x (features) and edge_index

        Returns:
            Predicted output for each node (magnitude for regression or logits for classification)
        """
        x, edge_index = data.x, data.edge_index

        # Project input to hidden dimension
        x = self.input_proj(x)
        x = F.relu(x)

        # Apply GCN layers with residual connections
        for i in range(self.num_layers):
            identity = x  # Save for residual connection
            x = self.convs[i](x, edge_index)
            x = self.layer_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + identity  # Residual connection

        # Output head
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        # For regression (output_dim=1), squeeze to 1D
        # For classification (output_dim>1), return 2D logits
        if self.output_dim == 1:
            return x.squeeze()
        else:
            return x
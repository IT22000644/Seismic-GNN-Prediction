"""
Graph Attention Network for earthquake magnitude prediction.
Uses attention mechanism to learn importance of different neighbors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class EarthquakeGAT(nn.Module):
    """
    GAT model for earthquake magnitude prediction with residual connections.

    Architecture:
    - Multi-head attention layers with layer normalization and residual connections
    - ELU activation (works better with attention)
    - MLP regression head

    Args:
        num_features: Number of input node features
        hidden_dim: Hidden dimension size (per head)
        num_layers: Number of GAT layers
        heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, num_features, hidden_dim=64, num_layers=2, heads=4, dropout=0.2, output_dim=1):
        super(EarthquakeGAT, self).__init__()

        self.num_layers = num_layers
        self.heads = heads
        self.output_dim = output_dim

        # Input projection to match hidden_dim * heads
        self.input_proj = nn.Linear(num_features, hidden_dim * heads)

        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # GAT layers with residual connections
        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.layer_norms.append(nn.LayerNorm(hidden_dim * heads))

        # Output head - supports both regression (output_dim=1) and classification (output_dim=2)
        self.fc1 = nn.Linear(hidden_dim * heads, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.dropout = dropout

    def forward(self, data):
        """
        Forward pass with attention mechanism and residual connections.

        Args:
            data: PyG Data object with x (features) and edge_index

        Returns:
            Predicted magnitude for each node
        """
        x, edge_index = data.x, data.edge_index

        # Project input to hidden dimension
        x = self.input_proj(x)
        x = F.elu(x)

        # Apply GAT layers with residual connections
        for i in range(self.num_layers):
            identity = x  # Save for residual connection
            x = self.convs[i](x, edge_index)
            x = self.layer_norms[i](x)
            x = F.elu(x)
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
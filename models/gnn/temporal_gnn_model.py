"""
Temporal GNN with real temporal sequences for earthquake prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TemporalEarthquakeGNN(nn.Module):
    """
    Temporal GNN model that processes sequences of earthquake events.
    
    Creates temporal windows where each node embedding is informed by:
    1. Its spatial neighbors (via GCN)
    2. Recent historical events in the same region (via LSTM)
    
    Args:
        num_features: Number of input node features
        hidden_dim: Hidden dimension for GCN
        lstm_hidden: Hidden dimension for LSTM
        num_gnn_layers: Number of GCN layers
        sequence_length: Number of historical events to consider
        dropout: Dropout rate
    """
    
    def __init__(self, num_features, hidden_dim=64, lstm_hidden=64,
                 num_gnn_layers=2, sequence_length=10, dropout=0.2, output_dim=1):
        super(TemporalEarthquakeGNN, self).__init__()

        self.sequence_length = sequence_length
        self.num_gnn_layers = num_gnn_layers
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)

        # Spatial GNN component with residual connections
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_gnn_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Temporal LSTM component (simplified to 1 layer)
        self.lstm = nn.LSTM(
            hidden_dim,
            lstm_hidden,
            batch_first=True,
            num_layers=1,
            dropout=0
        )

        # Output head - supports both regression (output_dim=1) and classification (output_dim=2)
        self.fc1 = nn.Linear(lstm_hidden, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

        self.dropout = dropout
        
    def create_temporal_sequences(self, x_spatial, batch_size=None):
        """
        Create temporal sequences from spatially-encoded features.
        
        For each earthquake, create a sequence of the K most recent
        earthquakes in the same spatial region.
        
        Args:
            x_spatial: Spatially-encoded node features [num_nodes, hidden_dim]
            batch_size: Optional batch size for memory efficiency
        
        Returns:
            sequences: [num_nodes, sequence_length, hidden_dim]
        """
        num_nodes = x_spatial.shape[0]
        
        # Simple approach: Create sliding windows
        # In practice, you'd sort by time and group by spatial region
        
        sequences = []
        for i in range(num_nodes):
            # Get indices of K nearest previous events
            # For simplicity, we take the last K events before current
            start_idx = max(0, i - self.sequence_length)
            
            # Create sequence (pad if needed)
            if i < self.sequence_length:
                # Pad with zeros at the beginning
                padding = torch.zeros(
                    self.sequence_length - i - 1, 
                    x_spatial.shape[1],
                    device=x_spatial.device
                )
                seq = torch.cat([padding, x_spatial[0:i+1]], dim=0)
            else:
                # Take last sequence_length events
                seq = x_spatial[start_idx:i+1][-self.sequence_length:]
            
            sequences.append(seq)
        
        return torch.stack(sequences)  # [num_nodes, sequence_length, hidden_dim]
        
    def forward(self, data):
        """
        Forward pass with real temporal sequences and residual connections.

        Args:
            data: PyG Data object with x (features) and edge_index

        Returns:
            Predicted magnitude for each node
        """
        x, edge_index = data.x, data.edge_index

        # Project input to hidden dimension
        x = self.input_proj(x)
        x = F.relu(x)

        # Step 1: Spatial encoding with GNN and residual connections
        for i in range(self.num_gnn_layers):
            identity = x  # Save for residual connection
            x = self.convs[i](x, edge_index)
            x = self.layer_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + identity  # Residual connection

        # Step 2: Create temporal sequences from spatial features
        x_sequences = self.create_temporal_sequences(x)
        # Shape: [num_nodes, sequence_length, hidden_dim]

        # Step 3: LSTM processing of sequences
        lstm_out, _ = self.lstm(x_sequences)
        x = lstm_out[:, -1, :]  # Take last timestep (current event)

        # Step 4: Output head
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        # For regression (output_dim=1), squeeze to 1D
        # For classification (output_dim>1), return 2D logits
        if self.output_dim == 1:
            return x.squeeze()
        else:
            return x
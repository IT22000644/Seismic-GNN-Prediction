"""
Dense Neural Network for aftershock prediction (binary classification).

This model serves as a baseline to demonstrate the value of graph structure
in GNN models. It uses the same features as GNNs but without the graph edges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AftershockDNN(nn.Module):
    """
    DNN model for aftershock triggering prediction (binary classification).

    This is a baseline model that uses only node features (no graph structure)
    to predict whether an earthquake will trigger aftershocks.

    Architecture:
    - Input: 28 engineered features per earthquake
    - 3 hidden layers with batch normalization and dropout
    - Output: 2 classes (no aftershock vs triggers aftershock)

    Args:
        num_features: Number of input features (28)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate for regularization
        output_dim: Number of output classes (2 for binary classification)
    """

    def __init__(self, num_features, hidden_dims=[128, 64, 32], dropout=0.3, output_dim=2):
        super(AftershockDNN, self).__init__()

        self.num_features = num_features
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.output_dim = output_dim

        # Build network layers
        layers = []
        in_dim = num_features

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input features [batch_size, num_features]

        Returns:
            logits: Output logits [batch_size, output_dim]
        """
        return self.network(x)


class AftershockDNN_Shallow(nn.Module):
    """
    Shallow DNN baseline - minimal architecture for comparison.

    Architecture:
    - Input: 28 features
    - 2 hidden layers
    - Output: 2 classes

    This simpler model helps understand if complexity matters.
    """

    def __init__(self, num_features, hidden_dim=64, dropout=0.2, output_dim=2):
        super(AftershockDNN_Shallow, self).__init__()

        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        self.dropout = dropout

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass."""
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc3(x)

        return x


class AftershockDNN_Deep(nn.Module):
    """
    Deep DNN baseline - more complex architecture.

    Architecture:
    - Input: 28 features
    - 5 hidden layers with residual connections
    - Output: 2 classes

    Tests if deeper networks help without graph structure.
    """

    def __init__(self, num_features, hidden_dim=128, dropout=0.3, output_dim=2):
        super(AftershockDNN_Deep, self).__init__()

        self.input_layer = nn.Linear(num_features, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)

        # Hidden layers with residual connections
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.hidden3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)

        self.hidden4 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 2)

        # Projection for residual connection
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim // 2)

        self.output_layer = nn.Linear(hidden_dim // 2, output_dim)

        self.dropout = dropout

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass with residual connections."""
        # Input
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Residual block 1
        identity = x
        x = self.hidden1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity  # Residual connection

        # Residual block 2
        identity = x
        x = self.hidden2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity  # Residual connection

        # Dimension reduction
        residual = self.residual_proj(x)
        x = self.hidden3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Residual block 3
        identity = residual
        x = self.hidden4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity  # Residual connection

        # Output
        x = self.output_layer(x)

        return x

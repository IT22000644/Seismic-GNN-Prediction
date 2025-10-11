"""
LSTM Neural Network for aftershock prediction (binary classification).

This model uses temporal sequences of earthquake events to predict whether
an earthquake will trigger aftershocks. It captures temporal patterns and
dependencies in seismic activity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AftershockLSTM(nn.Module):
    """
    LSTM model for aftershock triggering prediction (binary classification).

    Uses sequences of earthquakes to learn temporal patterns that lead to
    aftershock triggering events.

    Architecture:
    - Input: Sequences of earthquakes with features
    - 2-layer LSTM to capture temporal dependencies
    - Fully connected layers for classification
    - Output: 2 classes (no aftershock vs triggers aftershock)

    Args:
        num_features: Number of input features per earthquake (28)
        hidden_dim: Hidden dimension of LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout rate for regularization
        output_dim: Number of output classes (2 for binary classification)
    """

    def __init__(self, num_features, hidden_dim=128, num_layers=2, dropout=0.3, output_dim=2):
        super(AftershockLSTM, self).__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        # Batch normalization for LSTM output
        self.bn_lstm = nn.BatchNorm1d(hidden_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, output_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input sequences [batch_size, seq_len, num_features]

        Returns:
            logits: Output logits [batch_size, output_dim]
        """
        # LSTM forward pass
        # lstm_out: [batch_size, seq_len, hidden_dim]
        # h_n: [num_layers, batch_size, hidden_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state from the last LSTM layer
        # h_n[-1]: [batch_size, hidden_dim]
        h_last = h_n[-1]

        # Batch normalization
        h_last = self.bn_lstm(h_last)

        # Fully connected layers
        x = self.fc1(h_last)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc3(x)

        return x


class AftershockLSTM_Shallow(nn.Module):
    """
    Shallow LSTM baseline - single LSTM layer.

    Architecture:
    - 1 LSTM layer
    - 2 fully connected layers
    - Simpler model for comparison
    """

    def __init__(self, num_features, hidden_dim=64, dropout=0.2, output_dim=2):
        super(AftershockLSTM_Shallow, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Single LSTM layer
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.bn_lstm = nn.BatchNorm1d(hidden_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

        # Initialize weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        """Forward pass."""
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]

        h_last = self.bn_lstm(h_last)

        x = self.fc1(h_last)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc2(x)

        return x


class AftershockLSTM_Bidirectional(nn.Module):
    """
    Bidirectional LSTM - processes sequences in both directions.

    Architecture:
    - 2-layer Bidirectional LSTM
    - Captures both past and future context
    - 3 fully connected layers

    Note: In real-time prediction, we can't use future events,
    but this helps understand if bidirectional context improves performance.
    """

    def __init__(self, num_features, hidden_dim=128, num_layers=2, dropout=0.3, output_dim=2):
        super(AftershockLSTM_Bidirectional, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Bidirectional
        )

        # Batch norm (hidden_dim * 2 because bidirectional)
        self.bn_lstm = nn.BatchNorm1d(hidden_dim * 2)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        # Initialize weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        """Forward pass."""
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Concatenate forward and backward hidden states
        # h_n shape: [num_layers * 2, batch_size, hidden_dim]
        h_forward = h_n[-2]  # Last layer forward
        h_backward = h_n[-1]  # Last layer backward
        h_last = torch.cat([h_forward, h_backward], dim=1)

        h_last = self.bn_lstm(h_last)

        x = self.fc1(h_last)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc3(x)

        return x


class AftershockGRU(nn.Module):
    """
    GRU variant for comparison with LSTM.

    GRU is a simpler alternative to LSTM with fewer parameters.
    Often performs similarly to LSTM but trains faster.

    Architecture:
    - 2-layer GRU
    - 3 fully connected layers
    """

    def __init__(self, num_features, hidden_dim=128, num_layers=2, dropout=0.3, output_dim=2):
        super(AftershockGRU, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # GRU layers
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        self.bn_gru = nn.BatchNorm1d(hidden_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, output_dim)

        # Initialize weights
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        """Forward pass."""
        gru_out, h_n = self.gru(x)
        h_last = h_n[-1]

        h_last = self.bn_gru(h_last)

        x = self.fc1(h_last)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc3(x)

        return x

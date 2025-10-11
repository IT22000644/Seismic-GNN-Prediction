"""
Convolutional Neural Network for seismic risk zone classification.

This model uses 2D spatial grids of earthquake data to classify regions as
high-risk or low-risk for significant seismic activity.

Input: 32x32 grids with 2 channels (earthquake density + average magnitude)
Output: Binary classification (low-risk vs high-risk zone)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeismicRiskCNN(nn.Module):
    """
    CNN model for seismic risk zone classification.

    Uses 2D convolutional layers to detect spatial patterns in earthquake
    distribution and magnitude patterns across a region.

    Architecture:
    - Input: [batch_size, 2, 32, 32] (2 channels: density + avg magnitude)
    - 4 convolutional blocks with batch normalization and max pooling
    - Fully connected layers for classification
    - Output: [batch_size, 2] (low-risk vs high-risk)

    Args:
        in_channels: Number of input channels (2: density + magnitude)
        num_classes: Number of output classes (2 for binary classification)
        dropout: Dropout rate for regularization
    """

    def __init__(self, in_channels=2, num_classes=2, dropout=0.3):
        super(SeismicRiskCNN, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout

        # Convolutional block 1: 32x32 -> 16x16
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional block 2: 16x16 -> 8x8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional block 3: 8x8 -> 4x4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional block 4: 4x4 -> 2x2
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After 4 pooling layers: 32 -> 16 -> 8 -> 4 -> 2
        # Feature map size: 256 * 2 * 2 = 1024
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.fc_bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor [batch_size, 2, 32, 32]

        Returns:
            logits: Output logits [batch_size, num_classes]
        """
        # Convolutional blocks with ReLU and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = F.dropout2d(x, p=self.dropout * 0.5, training=self.training)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = F.dropout2d(x, p=self.dropout * 0.5, training=self.training)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = F.dropout2d(x, p=self.dropout * 0.7, training=self.training)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = F.dropout2d(x, p=self.dropout * 0.7, training=self.training)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # [batch_size, 256*2*2]

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc_bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc2(x)
        x = self.fc_bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc3(x)

        return x


class SeismicRiskCNN_Shallow(nn.Module):
    """
    Shallow CNN baseline - simpler architecture for comparison.

    Architecture:
    - 2 convolutional blocks
    - 2 fully connected layers
    - Fewer parameters than main model
    """

    def __init__(self, in_channels=2, num_classes=2, dropout=0.25):
        super(SeismicRiskCNN_Shallow, self).__init__()

        # Convolutional block 1: 32x32 -> 16x16
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional block 2: 16x16 -> 8x8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After 2 pooling: 32 -> 16 -> 8
        # Feature map: 64 * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = dropout

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = F.dropout2d(x, p=self.dropout * 0.5, training=self.training)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = F.dropout2d(x, p=self.dropout, training=self.training)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc2(x)

        return x


class SeismicRiskCNN_Deep(nn.Module):
    """
    Deep CNN with residual connections - more complex architecture.

    Architecture:
    - 6 convolutional blocks with residual connections
    - 3 fully connected layers
    - Tests if deeper networks improve risk classification
    """

    def __init__(self, in_channels=2, num_classes=2, dropout=0.35):
        super(SeismicRiskCNN_Deep, self).__init__()

        self.dropout = dropout

        # Initial convolution
        self.conv_init = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(32)

        # Residual block 1: 32x32 -> 16x16
        self.conv1a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(64)
        self.residual1 = nn.Conv2d(32, 64, kernel_size=1)  # Match dimensions
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Residual block 2: 16x16 -> 8x8
        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(128)
        self.residual2 = nn.Conv2d(64, 128, kernel_size=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Residual block 3: 8x8 -> 4x4
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(256)
        self.residual3 = nn.Conv2d(128, 256, kernel_size=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After 3 pooling: 32 -> 16 -> 8 -> 4
        # Feature map: 256 * 4 * 4 = 4096
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass with residual connections."""
        # Initial convolution
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = F.relu(x)

        # Residual block 1
        identity = self.residual1(x)
        x = self.conv1a(x)
        x = self.bn1a(x)
        x = F.relu(x)
        x = self.conv1b(x)
        x = self.bn1b(x)
        x = x + identity
        x = F.relu(x)
        x = self.pool1(x)
        x = F.dropout2d(x, p=self.dropout * 0.5, training=self.training)

        # Residual block 2
        identity = self.residual2(x)
        x = self.conv2a(x)
        x = self.bn2a(x)
        x = F.relu(x)
        x = self.conv2b(x)
        x = self.bn2b(x)
        x = x + identity
        x = F.relu(x)
        x = self.pool2(x)
        x = F.dropout2d(x, p=self.dropout * 0.7, training=self.training)

        # Residual block 3
        identity = self.residual3(x)
        x = self.conv3a(x)
        x = self.bn3a(x)
        x = F.relu(x)
        x = self.conv3b(x)
        x = self.bn3b(x)
        x = x + identity
        x = F.relu(x)
        x = self.pool3(x)
        x = F.dropout2d(x, p=self.dropout, training=self.training)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = self.fc1(x)
        x = self.fc_bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc2(x)
        x = self.fc_bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc3(x)

        return x

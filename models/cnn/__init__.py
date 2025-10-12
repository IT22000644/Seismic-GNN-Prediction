"""
Convolutional Neural Network models for seismic analysis.
"""

from .risk_cnn import SeismicRiskCNN, SeismicRiskCNN_Shallow, SeismicRiskCNN_Deep
from .aftershock_cnn import AftershockCNN, AftershockCNN_Shallow, AftershockCNN_Deep

__all__ = [
    'SeismicRiskCNN', 'SeismicRiskCNN_Shallow', 'SeismicRiskCNN_Deep',
    'AftershockCNN', 'AftershockCNN_Shallow', 'AftershockCNN_Deep'
]

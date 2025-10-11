"""
Convolutional Neural Network models for seismic risk classification.
"""

from .risk_cnn import SeismicRiskCNN, SeismicRiskCNN_Shallow, SeismicRiskCNN_Deep

__all__ = ['SeismicRiskCNN', 'SeismicRiskCNN_Shallow', 'SeismicRiskCNN_Deep']

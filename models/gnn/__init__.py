"""
GNN models for earthquake magnitude prediction.
"""

from .gcn_model import EarthquakeGCN
from .gat_model import EarthquakeGAT
from .graphsage_model import EarthquakeGraphSAGE
from .temporal_gnn_model import TemporalEarthquakeGNN

__all__ = [
    'EarthquakeGCN',
    'EarthquakeGAT', 
    'EarthquakeGraphSAGE',
    'TemporalEarthquakeGNN'
]
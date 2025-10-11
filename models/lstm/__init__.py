"""
LSTM and RNN models for earthquake sequence prediction.
"""

from .earthquake_lstm import (
    AftershockLSTM,
    AftershockLSTM_Shallow,
    AftershockLSTM_Bidirectional,
    AftershockGRU
)

__all__ = [
    'AftershockLSTM',
    'AftershockLSTM_Shallow',
    'AftershockLSTM_Bidirectional',
    'AftershockGRU'
]

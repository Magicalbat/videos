"""
Differentiable layers for genomics machine learning.

This module provides neural network layers specifically designed
for processing genomic sequences:
- Convolutional layers for motif detection
- Position Weight Matrix (PWM) layers
- Attention mechanisms for sequence analysis
- Pooling and aggregation operations

Supports both pure NumPy implementation and PyTorch integration.
"""

from genoml.layers.conv import (
    Conv1D,
    MaxPool1D,
    GlobalMaxPool1D,
    GlobalAvgPool1D,
)

from genoml.layers.pwm import (
    PWMScanner,
    MotifLayer,
    create_pwm_from_sequences,
)

from genoml.layers.attention import (
    SelfAttention,
    PositionalEncoding,
    MultiHeadAttention,
)

from genoml.layers.dense import (
    Dense,
    BatchNorm1D,
    Dropout,
    LayerNorm,
)

__all__ = [
    # Convolutions
    "Conv1D",
    "MaxPool1D",
    "GlobalMaxPool1D",
    "GlobalAvgPool1D",
    # PWM
    "PWMScanner",
    "MotifLayer",
    "create_pwm_from_sequences",
    # Attention
    "SelfAttention",
    "PositionalEncoding",
    "MultiHeadAttention",
    # Dense
    "Dense",
    "BatchNorm1D",
    "Dropout",
    "LayerNorm",
]

"""
Loss functions and metrics for genomics machine learning.

This module provides:
- Standard classification and regression losses
- Genomics-specific losses (e.g., for peak calling, expression prediction)
- Evaluation metrics common in computational biology
"""

from genoml.losses.classification import (
    cross_entropy_loss,
    binary_cross_entropy,
    focal_loss,
    label_smoothing_loss,
)

from genoml.losses.regression import (
    mse_loss,
    mae_loss,
    huber_loss,
    poisson_loss,
    negative_binomial_loss,
)

from genoml.losses.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    auroc,
    aupr,
    pearson_correlation,
    spearman_correlation,
)

__all__ = [
    # Classification losses
    "cross_entropy_loss",
    "binary_cross_entropy",
    "focal_loss",
    "label_smoothing_loss",
    # Regression losses
    "mse_loss",
    "mae_loss",
    "huber_loss",
    "poisson_loss",
    "negative_binomial_loss",
    # Metrics
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "auroc",
    "aupr",
    "pearson_correlation",
    "spearman_correlation",
]

"""
Regression loss functions for genomics.

These losses are used for tasks like:
- Gene expression prediction
- ChIP-seq signal prediction
- Read count modeling (RNA-seq)
"""

import numpy as np
from typing import Tuple


def mse_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    reduction: str = "mean"
) -> Tuple[float, np.ndarray]:
    """
    Mean Squared Error loss.

    Standard regression loss for continuous outputs.

    Args:
        predictions: Model predictions
        targets: Ground truth values
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Tuple of (loss value, gradient)
    """
    diff = predictions - targets
    losses = 0.5 * diff ** 2

    # Gradient: d/dx (0.5 * (x - t)^2) = (x - t)
    grad = diff

    if reduction == "mean":
        loss = np.mean(losses)
        grad = grad / predictions.size
    elif reduction == "sum":
        loss = np.sum(losses)
    else:
        loss = losses

    return float(loss) if reduction != "none" else loss, grad


def mae_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    reduction: str = "mean"
) -> Tuple[float, np.ndarray]:
    """
    Mean Absolute Error loss.

    More robust to outliers than MSE.

    Args:
        predictions: Model predictions
        targets: Ground truth values
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Tuple of (loss value, gradient)
    """
    diff = predictions - targets
    losses = np.abs(diff)

    # Gradient: sign of difference
    grad = np.sign(diff)

    if reduction == "mean":
        loss = np.mean(losses)
        grad = grad / predictions.size
    elif reduction == "sum":
        loss = np.sum(losses)
    else:
        loss = losses

    return float(loss) if reduction != "none" else loss, grad


def huber_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    delta: float = 1.0,
    reduction: str = "mean"
) -> Tuple[float, np.ndarray]:
    """
    Huber loss (smooth L1 loss).

    Combines benefits of MSE (smooth around zero) and MAE (robust to outliers).

    Args:
        predictions: Model predictions
        targets: Ground truth values
        delta: Threshold for switching between L1 and L2
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Tuple of (loss value, gradient)
    """
    diff = predictions - targets
    abs_diff = np.abs(diff)

    # Quadratic for small errors, linear for large
    quadratic = 0.5 * diff ** 2
    linear = delta * abs_diff - 0.5 * delta ** 2

    losses = np.where(abs_diff <= delta, quadratic, linear)

    # Gradient
    grad = np.where(abs_diff <= delta, diff, delta * np.sign(diff))

    if reduction == "mean":
        loss = np.mean(losses)
        grad = grad / predictions.size
    elif reduction == "sum":
        loss = np.sum(losses)
    else:
        loss = losses

    return float(loss) if reduction != "none" else loss, grad


def poisson_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    log_input: bool = True,
    reduction: str = "mean"
) -> Tuple[float, np.ndarray]:
    """
    Poisson negative log-likelihood loss.

    Appropriate for count data like RNA-seq read counts.
    Assumes targets follow Poisson distribution with rate = predictions.

    Args:
        predictions: Predicted rates (or log-rates if log_input=True)
        targets: Observed counts (non-negative integers)
        log_input: If True, predictions are log(rate)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Tuple of (loss value, gradient)

    Note:
        Loss = exp(pred) - target * pred (if log_input=True)
        Loss = pred - target * log(pred) (if log_input=False)
    """
    if log_input:
        # predictions = log(rate)
        exp_pred = np.exp(np.clip(predictions, -20, 20))
        losses = exp_pred - targets * predictions

        # Gradient: exp(pred) - target
        grad = exp_pred - targets
    else:
        # predictions = rate
        eps = 1e-8
        rate = np.maximum(predictions, eps)
        losses = rate - targets * np.log(rate)

        # Gradient: 1 - target/rate
        grad = 1 - targets / rate

    if reduction == "mean":
        loss = np.mean(losses)
        grad = grad / predictions.size
    elif reduction == "sum":
        loss = np.sum(losses)
    else:
        loss = losses

    return float(loss) if reduction != "none" else loss, grad


def negative_binomial_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    theta: float = 1.0,
    log_input: bool = True,
    reduction: str = "mean"
) -> Tuple[float, np.ndarray]:
    """
    Negative Binomial loss for overdispersed count data.

    RNA-seq count data often shows more variance than Poisson
    (overdispersion). Negative binomial handles this by adding
    a dispersion parameter.

    Args:
        predictions: Predicted mean (or log-mean if log_input=True)
        targets: Observed counts
        theta: Dispersion parameter (higher = more dispersed)
        log_input: If True, predictions are log(mean)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Tuple of (loss value, gradient)
    """
    eps = 1e-8

    if log_input:
        mu = np.exp(np.clip(predictions, -20, 20))
    else:
        mu = np.maximum(predictions, eps)

    # Negative binomial NLL
    # -log P(y|mu,theta) = -log(Gamma(y+r)/Gamma(y+1)/Gamma(r))
    #                      - r*log(r/(r+mu)) - y*log(mu/(r+mu))
    # where r = 1/theta

    r = 1.0 / theta

    losses = (
        -r * np.log(r / (r + mu) + eps) -
        targets * np.log(mu / (r + mu) + eps)
    )

    # Simplified gradient
    if log_input:
        grad = mu * (mu + targets) / (mu + r) - targets
    else:
        grad = (mu - targets * r / (r + mu)) / (r + mu)

    if reduction == "mean":
        loss = np.mean(losses)
        grad = grad / predictions.size
    elif reduction == "sum":
        loss = np.sum(losses)
    else:
        loss = losses

    return float(loss) if reduction != "none" else loss, grad


def cosine_similarity_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    reduction: str = "mean"
) -> Tuple[float, np.ndarray]:
    """
    Cosine similarity loss.

    Useful for comparing expression profiles or sequence embeddings.
    Loss = 1 - cosine_similarity (so minimizing means maximizing similarity)

    Args:
        predictions: Predicted vectors of shape (batch, dim)
        targets: Target vectors of shape (batch, dim)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Tuple of (loss value, gradient)
    """
    eps = 1e-8

    # Normalize
    pred_norm = np.sqrt(np.sum(predictions ** 2, axis=1, keepdims=True) + eps)
    target_norm = np.sqrt(np.sum(targets ** 2, axis=1, keepdims=True) + eps)

    pred_normalized = predictions / pred_norm
    target_normalized = targets / target_norm

    # Cosine similarity
    cos_sim = np.sum(pred_normalized * target_normalized, axis=1)

    # Loss = 1 - similarity
    losses = 1 - cos_sim

    # Gradient
    grad = -(target_normalized - cos_sim[:, np.newaxis] * pred_normalized) / pred_norm

    if reduction == "mean":
        loss = np.mean(losses)
        grad = grad / predictions.shape[0]
    elif reduction == "sum":
        loss = np.sum(losses)
    else:
        loss = losses

    return float(loss) if reduction != "none" else loss, grad


def pearson_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    reduction: str = "mean"
) -> Tuple[float, np.ndarray]:
    """
    Pearson correlation loss.

    Minimizes 1 - Pearson correlation. Useful for gene expression
    prediction where we care about relative ordering.

    Args:
        predictions: Predicted values (batch,) or (batch, features)
        targets: Target values
        reduction: 'mean' or 'sum' (over batch)

    Returns:
        Tuple of (loss value, gradient)
    """
    if predictions.ndim == 1:
        predictions = predictions[np.newaxis, :]
        targets = targets[np.newaxis, :]

    batch_size, n = predictions.shape
    eps = 1e-8

    # Center
    pred_mean = np.mean(predictions, axis=1, keepdims=True)
    target_mean = np.mean(targets, axis=1, keepdims=True)

    pred_centered = predictions - pred_mean
    target_centered = targets - target_mean

    # Covariance and stds
    cov = np.sum(pred_centered * target_centered, axis=1)
    pred_std = np.sqrt(np.sum(pred_centered ** 2, axis=1) + eps)
    target_std = np.sqrt(np.sum(target_centered ** 2, axis=1) + eps)

    # Pearson correlation
    corr = cov / (pred_std * target_std)

    # Loss = 1 - correlation
    losses = 1 - corr

    # Gradient (simplified)
    grad = -(target_centered - corr[:, np.newaxis] * pred_centered) / (
        pred_std[:, np.newaxis] * target_std[:, np.newaxis] + eps
    )

    if reduction == "mean":
        loss = np.mean(losses)
        grad = grad / batch_size
    elif reduction == "sum":
        loss = np.sum(losses)
    else:
        loss = losses

    return float(loss) if reduction != "none" else loss, grad.squeeze()

"""
Classification loss functions.

These losses are used for tasks like:
- Sequence classification (promoter vs non-promoter)
- Binding site prediction
- Variant effect prediction
"""

import numpy as np
from typing import Tuple, Optional


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def cross_entropy_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    reduction: str = "mean"
) -> Tuple[float, np.ndarray]:
    """
    Cross-entropy loss for multi-class classification.

    Args:
        logits: Raw model outputs of shape (batch, num_classes)
        targets: Integer class labels of shape (batch,) or
                 one-hot encoded of shape (batch, num_classes)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Tuple of (loss value, gradient w.r.t. logits)

    Example:
        >>> logits = np.array([[2.0, 1.0, 0.1]])
        >>> targets = np.array([0])  # Class 0
        >>> loss, grad = cross_entropy_loss(logits, targets)
    """
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]

    # Convert integer targets to one-hot if needed
    if targets.ndim == 1:
        one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
        one_hot[np.arange(batch_size), targets] = 1.0
    else:
        one_hot = targets

    # Compute softmax probabilities
    probs = softmax(logits)

    # Compute loss
    eps = 1e-7
    log_probs = np.log(probs + eps)
    losses = -np.sum(one_hot * log_probs, axis=1)

    # Compute gradient: softmax - one_hot
    grad = probs - one_hot

    if reduction == "mean":
        loss = np.mean(losses)
        grad = grad / batch_size
    elif reduction == "sum":
        loss = np.sum(losses)
    else:
        loss = losses

    return float(loss) if reduction != "none" else loss, grad


def binary_cross_entropy(
    predictions: np.ndarray,
    targets: np.ndarray,
    from_logits: bool = True,
    reduction: str = "mean",
    pos_weight: Optional[float] = None
) -> Tuple[float, np.ndarray]:
    """
    Binary cross-entropy loss.

    Commonly used for binary classification tasks like
    peak calling or binding site prediction.

    Args:
        predictions: Model outputs of shape (batch,) or (batch, 1)
        targets: Binary labels of shape (batch,) or (batch, 1)
        from_logits: If True, predictions are raw logits
        reduction: 'mean', 'sum', or 'none'
        pos_weight: Weight for positive class (for imbalanced data)

    Returns:
        Tuple of (loss value, gradient)
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    if from_logits:
        # Use numerically stable version
        # Loss = max(x, 0) - x*t + log(1 + exp(-|x|))
        pos_loss = np.maximum(predictions, 0) - predictions * targets
        neg_loss = np.log1p(np.exp(-np.abs(predictions)))
        losses = pos_loss + neg_loss

        # Gradient
        probs = 1 / (1 + np.exp(-predictions))
        grad = probs - targets
    else:
        # Predictions are already probabilities
        eps = 1e-7
        probs = np.clip(predictions, eps, 1 - eps)
        losses = -targets * np.log(probs) - (1 - targets) * np.log(1 - probs)
        grad = (probs - targets) / (probs * (1 - probs) + eps)

    # Apply positive class weight
    if pos_weight is not None:
        weight = targets * pos_weight + (1 - targets)
        losses = losses * weight
        grad = grad * weight

    if reduction == "mean":
        loss = np.mean(losses)
        grad = grad / len(predictions)
    elif reduction == "sum":
        loss = np.sum(losses)
    else:
        loss = losses

    return float(loss) if reduction != "none" else loss, grad


def focal_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
    reduction: str = "mean"
) -> Tuple[float, np.ndarray]:
    """
    Focal loss for handling class imbalance.

    Focal loss down-weights easy examples and focuses on hard ones.
    Particularly useful for imbalanced genomics datasets.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: Raw model outputs
        targets: Binary labels
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Class weight for positive class
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Tuple of (loss value, gradient)
    """
    logits = logits.flatten()
    targets = targets.flatten()

    # Sigmoid
    probs = 1 / (1 + np.exp(-logits))

    # p_t = p if y=1 else 1-p
    p_t = targets * probs + (1 - targets) * (1 - probs)

    # Focal weight
    focal_weight = (1 - p_t) ** gamma

    # Cross entropy
    eps = 1e-7
    ce = -targets * np.log(probs + eps) - (1 - targets) * np.log(1 - probs + eps)

    # Focal loss
    losses = focal_weight * ce

    # Apply alpha weighting
    if alpha is not None:
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
        losses = alpha_t * losses

    # Gradient (approximation)
    grad = focal_weight * (probs - targets)
    grad = grad - gamma * (1 - p_t) ** (gamma - 1) * p_t * ce * (2 * targets - 1)

    if alpha is not None:
        grad = alpha_t * grad

    if reduction == "mean":
        loss = np.mean(losses)
        grad = grad / len(logits)
    elif reduction == "sum":
        loss = np.sum(losses)
    else:
        loss = losses

    return float(loss) if reduction != "none" else loss, grad


def label_smoothing_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    smoothing: float = 0.1,
    reduction: str = "mean"
) -> Tuple[float, np.ndarray]:
    """
    Cross-entropy with label smoothing.

    Label smoothing helps prevent overconfident predictions
    and can improve generalization.

    Args:
        logits: Raw model outputs of shape (batch, num_classes)
        targets: Integer class labels
        smoothing: Amount of smoothing (0 = no smoothing, 1 = uniform)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Tuple of (loss value, gradient)
    """
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]

    # Create smoothed targets
    if targets.ndim == 1:
        one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
        one_hot[np.arange(batch_size), targets] = 1.0
    else:
        one_hot = targets

    smooth_targets = one_hot * (1 - smoothing) + smoothing / num_classes

    # Compute softmax and loss
    probs = softmax(logits)
    eps = 1e-7
    log_probs = np.log(probs + eps)
    losses = -np.sum(smooth_targets * log_probs, axis=1)

    # Gradient
    grad = probs - smooth_targets

    if reduction == "mean":
        loss = np.mean(losses)
        grad = grad / batch_size
    elif reduction == "sum":
        loss = np.sum(losses)
    else:
        loss = losses

    return float(loss) if reduction != "none" else loss, grad


def multi_label_cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
    reduction: str = "mean"
) -> Tuple[float, np.ndarray]:
    """
    Multi-label binary cross-entropy loss.

    For tasks where each sample can have multiple labels,
    like predicting multiple transcription factors binding.

    Args:
        logits: Raw model outputs of shape (batch, num_labels)
        targets: Binary labels of shape (batch, num_labels)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Tuple of (loss value, gradient)
    """
    # Apply sigmoid independently for each label
    probs = 1 / (1 + np.exp(-logits))

    eps = 1e-7
    losses = (
        -targets * np.log(probs + eps) -
        (1 - targets) * np.log(1 - probs + eps)
    )

    # Sum over labels
    losses = np.sum(losses, axis=1)

    # Gradient
    grad = probs - targets

    if reduction == "mean":
        loss = np.mean(losses)
        grad = grad / logits.shape[0]
    elif reduction == "sum":
        loss = np.sum(losses)
    else:
        loss = losses

    return float(loss) if reduction != "none" else loss, grad

"""
Evaluation metrics for genomics machine learning.

Common metrics used to evaluate models for tasks like:
- Classification (binding site prediction, variant classification)
- Regression (expression prediction, signal prediction)
"""

import numpy as np
from typing import Optional, Tuple


def accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Classification accuracy.

    Args:
        predictions: Predicted probabilities or logits
        targets: True labels (0 or 1 for binary, class indices for multi-class)
        threshold: Threshold for binary classification

    Returns:
        Accuracy as a fraction
    """
    if predictions.ndim == 1 or predictions.shape[1] == 1:
        # Binary classification
        preds = (predictions.flatten() >= threshold).astype(int)
        return float(np.mean(preds == targets.flatten()))
    else:
        # Multi-class
        preds = np.argmax(predictions, axis=1)
        return float(np.mean(preds == targets))


def precision(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Precision (positive predictive value).

    Precision = TP / (TP + FP)

    Args:
        predictions: Predicted probabilities
        targets: True binary labels

    Returns:
        Precision score
    """
    preds = (predictions.flatten() >= threshold).astype(int)
    targets = targets.flatten()

    tp = np.sum((preds == 1) & (targets == 1))
    fp = np.sum((preds == 1) & (targets == 0))

    if tp + fp == 0:
        return 0.0

    return float(tp / (tp + fp))


def recall(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Recall (sensitivity, true positive rate).

    Recall = TP / (TP + FN)

    Args:
        predictions: Predicted probabilities
        targets: True binary labels

    Returns:
        Recall score
    """
    preds = (predictions.flatten() >= threshold).astype(int)
    targets = targets.flatten()

    tp = np.sum((preds == 1) & (targets == 1))
    fn = np.sum((preds == 0) & (targets == 1))

    if tp + fn == 0:
        return 0.0

    return float(tp / (tp + fn))


def f1_score(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    F1 score (harmonic mean of precision and recall).

    Args:
        predictions: Predicted probabilities
        targets: True binary labels

    Returns:
        F1 score
    """
    p = precision(predictions, targets, threshold)
    r = recall(predictions, targets, threshold)

    if p + r == 0:
        return 0.0

    return float(2 * p * r / (p + r))


def confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Compute confusion matrix.

    Returns:
        2x2 matrix: [[TN, FP], [FN, TP]]
    """
    preds = (predictions.flatten() >= threshold).astype(int)
    targets = targets.flatten()

    tn = np.sum((preds == 0) & (targets == 0))
    fp = np.sum((preds == 1) & (targets == 0))
    fn = np.sum((preds == 0) & (targets == 1))
    tp = np.sum((preds == 1) & (targets == 1))

    return np.array([[tn, fp], [fn, tp]])


def auroc(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_thresholds: int = 1000
) -> float:
    """
    Area Under the Receiver Operating Characteristic curve.

    AUROC measures the ability to distinguish between classes.
    0.5 = random, 1.0 = perfect discrimination.

    Args:
        predictions: Predicted probabilities
        targets: True binary labels
        num_thresholds: Number of thresholds for approximation

    Returns:
        AUROC score
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Sort by predictions
    sorted_indices = np.argsort(predictions)[::-1]
    predictions = predictions[sorted_indices]
    targets = targets[sorted_indices]

    # Compute TPR and FPR at each threshold
    thresholds = np.linspace(0, 1, num_thresholds)

    tprs = []
    fprs = []

    total_pos = np.sum(targets == 1)
    total_neg = np.sum(targets == 0)

    if total_pos == 0 or total_neg == 0:
        return 0.5

    for thresh in thresholds:
        preds = (predictions >= thresh).astype(int)
        tp = np.sum((preds == 1) & (targets == 1))
        fp = np.sum((preds == 1) & (targets == 0))

        tprs.append(tp / total_pos)
        fprs.append(fp / total_neg)

    # Compute AUC using trapezoidal rule
    fprs = np.array(fprs)
    tprs = np.array(tprs)

    # Sort by FPR
    sorted_idx = np.argsort(fprs)
    fprs = fprs[sorted_idx]
    tprs = tprs[sorted_idx]

    auc = np.trapz(tprs, fprs)

    return float(auc)


def aupr(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_thresholds: int = 1000
) -> float:
    """
    Area Under the Precision-Recall curve.

    AUPR is particularly useful for imbalanced datasets
    common in genomics (e.g., rare variants, sparse binding).

    Args:
        predictions: Predicted probabilities
        targets: True binary labels
        num_thresholds: Number of thresholds for approximation

    Returns:
        AUPR score
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    thresholds = np.linspace(1, 0, num_thresholds)

    precisions = []
    recalls = []

    for thresh in thresholds:
        p = precision(predictions, targets, thresh)
        r = recall(predictions, targets, thresh)
        precisions.append(p)
        recalls.append(r)

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # Sort by recall
    sorted_idx = np.argsort(recalls)
    recalls = recalls[sorted_idx]
    precisions = precisions[sorted_idx]

    # Compute AUC
    auc = np.trapz(precisions, recalls)

    return float(auc)


def pearson_correlation(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Pearson correlation coefficient.

    Measures linear relationship between predictions and targets.
    Common metric for expression prediction.

    Args:
        predictions: Predicted values
        targets: True values

    Returns:
        Pearson correlation coefficient (-1 to 1)
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    pred_mean = np.mean(predictions)
    target_mean = np.mean(targets)

    pred_centered = predictions - pred_mean
    target_centered = targets - target_mean

    numerator = np.sum(pred_centered * target_centered)
    denominator = np.sqrt(
        np.sum(pred_centered ** 2) * np.sum(target_centered ** 2)
    )

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def spearman_correlation(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Spearman rank correlation coefficient.

    Measures monotonic relationship between predictions and targets.
    More robust to outliers than Pearson.

    Args:
        predictions: Predicted values
        targets: True values

    Returns:
        Spearman correlation coefficient (-1 to 1)
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Convert to ranks
    def rank_data(x):
        sorted_idx = np.argsort(x)
        ranks = np.empty_like(sorted_idx, dtype=float)
        ranks[sorted_idx] = np.arange(len(x)) + 1
        return ranks

    pred_ranks = rank_data(predictions)
    target_ranks = rank_data(targets)

    # Pearson on ranks = Spearman
    return pearson_correlation(pred_ranks, target_ranks)


def r_squared(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    R-squared (coefficient of determination).

    Measures proportion of variance explained by the model.

    Args:
        predictions: Predicted values
        targets: True values

    Returns:
        R-squared value (can be negative for poor models)
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1 - ss_res / ss_tot)


def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((predictions - targets) ** 2))


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(predictions - targets)))


def top_k_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    k: int = 5
) -> float:
    """
    Top-k accuracy for multi-class classification.

    Correct if true class is in top k predictions.

    Args:
        predictions: Predicted logits/probabilities (batch, num_classes)
        targets: True class indices

    Returns:
        Top-k accuracy
    """
    targets = targets.flatten()
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:]

    correct = 0
    for i, target in enumerate(targets):
        if target in top_k_preds[i]:
            correct += 1

    return float(correct / len(targets))


def per_class_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: Optional[int] = None
) -> dict:
    """
    Compute metrics per class for multi-class classification.

    Args:
        predictions: Predicted class probabilities (batch, num_classes)
        targets: True class indices

    Returns:
        Dictionary with per-class precision, recall, f1
    """
    if num_classes is None:
        num_classes = predictions.shape[1]

    pred_classes = np.argmax(predictions, axis=1)
    targets = targets.flatten()

    metrics = {}

    for c in range(num_classes):
        binary_preds = (pred_classes == c).astype(float)
        binary_targets = (targets == c).astype(float)

        metrics[f"class_{c}"] = {
            "precision": precision(binary_preds, binary_targets, threshold=0.5),
            "recall": recall(binary_preds, binary_targets, threshold=0.5),
            "f1": f1_score(binary_preds, binary_targets, threshold=0.5),
        }

    return metrics

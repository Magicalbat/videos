#!/usr/bin/env python3
"""
Example: Promoter Classification with GenomL

This example demonstrates how to use GenomL to build a simple
neural network for classifying DNA sequences as promoters or
non-promoters.

Promoters are regions upstream of genes where transcription
is initiated. Identifying promoters is a classic problem in
computational biology.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from genoml.sequence import one_hot_encode, batch_encode
from genoml.layers import Conv1D, GlobalMaxPool1D, Dense, Dropout
from genoml.losses import binary_cross_entropy, accuracy, auroc, f1_score


def generate_synthetic_data(n_samples=1000, seq_length=200):
    """
    Generate synthetic promoter and non-promoter sequences.

    Real promoters have specific motifs like TATA box (TATAAA)
    near position -30 and GC-rich regions.
    """
    sequences = []
    labels = []

    for _ in range(n_samples // 2):
        # Generate promoter-like sequence
        seq = list(np.random.choice(['A', 'C', 'G', 'T'], seq_length))

        # Add TATA box motif around position 170 (-30 from end)
        tata_pos = 170
        seq[tata_pos:tata_pos+6] = list('TATAAA')

        # Add GC-rich region upstream
        for i in range(50, 100):
            if np.random.random() > 0.3:
                seq[i] = np.random.choice(['G', 'C'])

        sequences.append(''.join(seq))
        labels.append(1)

    for _ in range(n_samples // 2):
        # Generate non-promoter sequence (random)
        seq = ''.join(np.random.choice(['A', 'C', 'G', 'T'], seq_length))
        sequences.append(seq)
        labels.append(0)

    return sequences, np.array(labels)


class PromoterClassifier:
    """
    Simple CNN for promoter classification.

    Architecture:
    - Conv1D (learns motif patterns)
    - GlobalMaxPool (finds strongest motif match)
    - Dense layers for classification
    """

    def __init__(self, seq_length=200, n_filters=32, kernel_size=8):
        self.seq_length = seq_length

        # Convolutional layer to detect motifs
        self.conv1 = Conv1D(
            in_channels=4,  # One-hot DNA
            out_channels=n_filters,
            kernel_size=kernel_size,
            activation='relu'
        )

        # Global pooling
        self.pool = GlobalMaxPool1D()

        # Dropout for regularization
        self.dropout = Dropout(p=0.3)

        # Dense layers
        self.dense1 = Dense(n_filters, 16, activation='relu')
        self.dense2 = Dense(16, 1, activation='sigmoid')

        self.layers = [self.conv1, self.pool, self.dropout, self.dense1, self.dense2]

    def forward(self, x, training=True):
        """Forward pass through the network."""
        # x: (batch, seq_length, 4) -> transpose to (batch, 4, seq_length)
        x = np.transpose(x, (0, 2, 1))

        # Conv -> Pool
        x = self.conv1.forward(x)
        x = self.pool.forward(x)

        # Dropout (only during training)
        if training:
            x = self.dropout.forward(x)

        # Dense layers
        x = self.dense1.forward(x)
        x = self.dense2.forward(x)

        return x.flatten()

    def backward(self, grad):
        """Backward pass for gradient computation."""
        grad = grad.reshape(-1, 1)

        grad = self.dense2.backward(grad)
        grad = self.dense1.backward(grad)
        grad = self.dropout.backward(grad)
        grad = self.pool.backward(grad)
        grad = self.conv1.backward(grad)

        return grad

    def update_weights(self, learning_rate):
        """Update weights using gradients."""
        # Conv layer
        self.conv1.weights -= learning_rate * self.conv1.weights_grad
        self.conv1.bias -= learning_rate * self.conv1.bias_grad

        # Dense layers
        self.dense1.weights -= learning_rate * self.dense1.weights_grad
        self.dense1.bias -= learning_rate * self.dense1.bias_grad

        self.dense2.weights -= learning_rate * self.dense2.weights_grad
        self.dense2.bias -= learning_rate * self.dense2.bias_grad


def train_epoch(model, X, y, batch_size=32, learning_rate=0.001):
    """Train for one epoch."""
    n_samples = len(X)
    indices = np.random.permutation(n_samples)

    total_loss = 0
    n_batches = 0

    for i in range(0, n_samples, batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]

        # Forward pass
        predictions = model.forward(X_batch, training=True)

        # Compute loss and gradient
        loss, grad = binary_cross_entropy(predictions, y_batch, from_logits=False)

        # Backward pass
        model.backward(grad)

        # Update weights
        model.update_weights(learning_rate)

        total_loss += loss
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, X, y):
    """Evaluate model on data."""
    predictions = model.forward(X, training=False)

    loss, _ = binary_cross_entropy(predictions, y, from_logits=False)
    acc = accuracy(predictions, y)
    auc = auroc(predictions, y)
    f1 = f1_score(predictions, y)

    return {
        'loss': loss,
        'accuracy': acc,
        'auroc': auc,
        'f1': f1
    }


def main():
    print("=" * 60)
    print("Promoter Classification with GenomL")
    print("=" * 60)

    # Generate data
    print("\nGenerating synthetic data...")
    sequences, labels = generate_synthetic_data(n_samples=1000)

    # Encode sequences
    print("Encoding sequences...")
    X = batch_encode(sequences)  # (n_samples, seq_length, 4)
    y = labels

    # Train/test split
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Create model
    print("\nCreating model...")
    model = PromoterClassifier(
        seq_length=200,
        n_filters=32,
        kernel_size=8
    )

    # Training
    print("\nTraining...")
    n_epochs = 20

    for epoch in range(n_epochs):
        train_loss = train_epoch(
            model, X_train, y_train,
            batch_size=32,
            learning_rate=0.001
        )

        if (epoch + 1) % 5 == 0:
            metrics = evaluate(model, X_test, y_test)
            print(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f}, "
                  f"Test Acc={metrics['accuracy']:.4f}, "
                  f"AUROC={metrics['auroc']:.4f}")

    # Final evaluation
    print("\nFinal Evaluation:")
    print("-" * 40)
    final_metrics = evaluate(model, X_test, y_test)
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Show learned motifs
    print("\nLearned Motif Filters (top 5 by weight magnitude):")
    print("-" * 40)

    filter_norms = np.linalg.norm(model.conv1.weights, axis=(1, 2))
    top_filters = np.argsort(filter_norms)[-5:]

    nuc_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    for f_idx in top_filters:
        # Convert filter to consensus
        consensus = []
        for pos in range(model.conv1.kernel_size):
            best_nuc = np.argmax(model.conv1.weights[f_idx, :, pos])
            consensus.append(nuc_map[best_nuc])
        print(f"  Filter {f_idx}: {''.join(consensus)}")


if __name__ == "__main__":
    main()

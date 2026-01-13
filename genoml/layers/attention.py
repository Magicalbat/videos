"""
Attention mechanisms for genomic sequence analysis.

Attention allows models to focus on relevant parts of sequences,
which is particularly useful for:
- Long-range dependencies in regulatory sequences
- Identifying important positions in binding sites
- Learning sequence context effects
"""

import numpy as np
from typing import Optional, Tuple


class PositionalEncoding:
    """
    Sinusoidal positional encoding for sequence positions.

    Adds position information to sequence embeddings, enabling
    the model to understand positional context.

    Args:
        d_model: Dimension of the model embeddings
        max_len: Maximum sequence length
        dropout: Dropout rate (0 to disable)
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 10000,
        dropout: float = 0.0
    ):
        self.d_model = d_model
        self.dropout = dropout

        # Compute positional encodings
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input.

        Args:
            x: Input of shape (batch, seq_len, d_model)

        Returns:
            Output with positional encoding added
        """
        seq_len = x.shape[1]
        output = x + self.pe[:seq_len]

        if self.dropout > 0:
            mask = (np.random.random(output.shape) > self.dropout).astype(np.float32)
            output = output * mask / (1 - self.dropout)
            self._cache["dropout_mask"] = mask

        return output

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class SelfAttention:
    """
    Self-attention layer for sequence analysis.

    Computes attention weights between all positions in a sequence,
    allowing the model to capture long-range dependencies.

    Args:
        d_model: Dimension of input/output
        d_k: Dimension of keys and queries (default: d_model)
        d_v: Dimension of values (default: d_model)
        scale: Whether to scale attention scores
    """

    def __init__(
        self,
        d_model: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        scale: bool = True
    ):
        self.d_model = d_model
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model
        self.scale = scale

        # Initialize weight matrices
        scale_factor = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, self.d_k).astype(np.float32) * scale_factor
        self.W_k = np.random.randn(d_model, self.d_k).astype(np.float32) * scale_factor
        self.W_v = np.random.randn(d_model, self.d_v).astype(np.float32) * scale_factor
        self.W_o = np.random.randn(self.d_v, d_model).astype(np.float32) * scale_factor

        # Gradients
        self.W_q_grad = None
        self.W_k_grad = None
        self.W_v_grad = None
        self.W_o_grad = None

        self._cache = {}

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Compute queries, keys, values
        Q = np.dot(x, self.W_q)  # (batch, seq_len, d_k)
        K = np.dot(x, self.W_k)  # (batch, seq_len, d_k)
        V = np.dot(x, self.W_v)  # (batch, seq_len, d_v)

        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1))  # (batch, seq_len, seq_len)

        if self.scale:
            scores = scores / np.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask * (-1e9)

        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Apply attention to values
        context = np.matmul(attention, V)  # (batch, seq_len, d_v)

        # Output projection
        output = np.dot(context, self.W_o)

        # Cache for backward pass
        self._cache = {
            "input": x,
            "Q": Q, "K": K, "V": V,
            "attention": attention,
            "context": context,
        }

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            grad_output: Gradient of shape (batch, seq_len, d_model)

        Returns:
            Gradient w.r.t. input
        """
        x = self._cache["input"]
        Q = self._cache["Q"]
        K = self._cache["K"]
        V = self._cache["V"]
        attention = self._cache["attention"]
        context = self._cache["context"]

        batch_size, seq_len, _ = x.shape

        # Gradient w.r.t. W_o
        self.W_o_grad = np.sum(
            np.matmul(context.transpose(0, 2, 1), grad_output),
            axis=0
        )

        # Gradient w.r.t. context
        grad_context = np.dot(grad_output, self.W_o.T)

        # Gradient w.r.t. attention and V
        grad_attention = np.matmul(grad_context, V.transpose(0, 2, 1))
        grad_V = np.matmul(attention.transpose(0, 2, 1), grad_context)

        # Gradient through softmax
        grad_scores = attention * (
            grad_attention - np.sum(grad_attention * attention, axis=-1, keepdims=True)
        )

        if self.scale:
            grad_scores = grad_scores / np.sqrt(self.d_k)

        # Gradient w.r.t. Q and K
        grad_Q = np.matmul(grad_scores, K)
        grad_K = np.matmul(grad_scores.transpose(0, 2, 1), Q)

        # Gradient w.r.t. weight matrices
        self.W_q_grad = np.sum(np.matmul(x.transpose(0, 2, 1), grad_Q), axis=0)
        self.W_k_grad = np.sum(np.matmul(x.transpose(0, 2, 1), grad_K), axis=0)
        self.W_v_grad = np.sum(np.matmul(x.transpose(0, 2, 1), grad_V), axis=0)

        # Gradient w.r.t. input
        grad_input = (
            np.dot(grad_Q, self.W_q.T) +
            np.dot(grad_K, self.W_k.T) +
            np.dot(grad_V, self.W_v.T)
        )

        return grad_input

    def get_attention_weights(self) -> np.ndarray:
        """Return the attention weights from the last forward pass."""
        return self._cache.get("attention", None)

    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        return self.forward(x, mask)


class MultiHeadAttention:
    """
    Multi-head attention layer.

    Uses multiple attention heads to capture different types
    of relationships in the sequence.

    Args:
        d_model: Dimension of input/output
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout

        # Initialize projections
        scale_factor = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * scale_factor
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * scale_factor
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * scale_factor
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * scale_factor

        self._cache = {}

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split into multiple heads."""
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq_len, d_k)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """Merge heads back together."""
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)  # (batch, seq_len, heads, d_k)
        return x.reshape(batch_size, seq_len, self.d_model)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # Split into heads
        Q = self._split_heads(Q)  # (batch, heads, seq_len, d_k)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask * (-1e9)

        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Apply dropout
        if self.dropout > 0:
            drop_mask = (np.random.random(attention.shape) > self.dropout).astype(np.float32)
            attention = attention * drop_mask / (1 - self.dropout)

        # Apply attention to values
        context = np.matmul(attention, V)

        # Merge heads and project
        context = self._merge_heads(context)
        output = np.dot(context, self.W_o)

        self._cache = {
            "input": x,
            "Q": Q, "K": K, "V": V,
            "attention": attention,
            "context": context,
        }

        return output

    def get_attention_weights(self) -> np.ndarray:
        """Return attention weights from all heads."""
        return self._cache.get("attention", None)

    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        return self.forward(x, mask)


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create a causal (autoregressive) attention mask.

    Prevents attending to future positions.

    Args:
        seq_len: Sequence length

    Returns:
        Mask of shape (1, 1, seq_len, seq_len)
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask.reshape(1, 1, seq_len, seq_len)


def create_padding_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
    """
    Create a padding mask for variable-length sequences.

    Args:
        lengths: Array of sequence lengths (batch_size,)
        max_len: Maximum sequence length

    Returns:
        Mask of shape (batch_size, 1, 1, max_len)
    """
    batch_size = len(lengths)
    mask = np.zeros((batch_size, max_len), dtype=np.float32)

    for i, length in enumerate(lengths):
        mask[i, length:] = 1.0

    return mask.reshape(batch_size, 1, 1, max_len)

"""
Convolutional layers for genomic sequence processing.

1D convolutions are essential for motif detection in DNA/RNA sequences.
These layers can learn sequence patterns like transcription factor binding sites.
"""

import numpy as np
from typing import Optional, Tuple, Callable


class Conv1D:
    """
    1D Convolutional layer for sequence data.

    Designed for processing one-hot encoded genomic sequences.
    Learns filters that can detect sequence motifs.

    Args:
        in_channels: Number of input channels (4 for one-hot DNA)
        out_channels: Number of output filters
        kernel_size: Size of convolutional kernel (motif length)
        stride: Stride of convolution
        padding: Padding mode ('valid', 'same', or int)
        activation: Activation function ('relu', 'sigmoid', 'tanh', None)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str = "valid",
        activation: Optional[str] = "relu"
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation

        # Initialize weights using He initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size))
        self.weights = np.random.randn(
            out_channels, in_channels, kernel_size
        ).astype(np.float32) * scale
        self.bias = np.zeros(out_channels, dtype=np.float32)

        # For gradient computation
        self.weights_grad = None
        self.bias_grad = None
        self._cache = {}

    def _apply_padding(self, x: np.ndarray) -> np.ndarray:
        """Apply padding to input."""
        if self.padding == "valid":
            return x
        elif self.padding == "same":
            pad_total = self.kernel_size - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return np.pad(
                x,
                ((0, 0), (0, 0), (pad_left, pad_right)),
                mode="constant"
            )
        elif isinstance(self.padding, int):
            return np.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding)),
                mode="constant"
            )
        else:
            raise ValueError(f"Unknown padding mode: {self.padding}")

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation is None:
            return x
        elif self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == "tanh":
            return np.tanh(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, seq_length)

        Returns:
            Output tensor of shape (batch, out_channels, new_length)
        """
        self._cache["input"] = x

        x = self._apply_padding(x)
        batch_size, in_ch, seq_len = x.shape

        # Calculate output length
        out_len = (seq_len - self.kernel_size) // self.stride + 1

        # Perform convolution using im2col-style operation
        output = np.zeros(
            (batch_size, self.out_channels, out_len),
            dtype=np.float32
        )

        for i in range(out_len):
            start = i * self.stride
            end = start + self.kernel_size
            # Extract patch: (batch, in_channels, kernel_size)
            patch = x[:, :, start:end]
            # Convolve: (batch, out_channels)
            for f in range(self.out_channels):
                output[:, f, i] = np.sum(
                    patch * self.weights[f], axis=(1, 2)
                ) + self.bias[f]

        self._cache["pre_activation"] = output.copy()
        output = self._activate(output)
        self._cache["output"] = output

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            grad_output: Gradient w.r.t. output

        Returns:
            Gradient w.r.t. input
        """
        pre_act = self._cache["pre_activation"]
        x = self._cache["input"]

        # Gradient through activation
        if self.activation == "relu":
            grad_output = grad_output * (pre_act > 0)
        elif self.activation == "sigmoid":
            sig = 1 / (1 + np.exp(-np.clip(pre_act, -500, 500)))
            grad_output = grad_output * sig * (1 - sig)
        elif self.activation == "tanh":
            grad_output = grad_output * (1 - np.tanh(pre_act) ** 2)

        x = self._apply_padding(x)
        batch_size, in_ch, seq_len = x.shape
        _, out_ch, out_len = grad_output.shape

        # Gradient w.r.t. weights and bias
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)

        for i in range(out_len):
            start = i * self.stride
            end = start + self.kernel_size
            patch = x[:, :, start:end]

            for f in range(self.out_channels):
                self.weights_grad[f] += np.sum(
                    patch * grad_output[:, f:f+1, i:i+1],
                    axis=0
                )
                self.bias_grad[f] += np.sum(grad_output[:, f, i])

        # Gradient w.r.t. input
        grad_input = np.zeros_like(x)
        for i in range(out_len):
            start = i * self.stride
            end = start + self.kernel_size
            for f in range(self.out_channels):
                grad_input[:, :, start:end] += (
                    self.weights[f] * grad_output[:, f:f+1, i:i+1]
                )

        # Remove padding from gradient
        if self.padding == "same":
            pad_total = self.kernel_size - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            if pad_right > 0:
                grad_input = grad_input[:, :, pad_left:-pad_right]
            else:
                grad_input = grad_input[:, :, pad_left:]

        return grad_input

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class MaxPool1D:
    """
    1D Max Pooling layer.

    Useful for reducing sequence length while preserving
    the strongest signals (e.g., strongest motif matches).
    """

    def __init__(self, pool_size: int = 2, stride: Optional[int] = None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, channels, length)

        Returns:
            Pooled output
        """
        batch, channels, length = x.shape
        out_len = (length - self.pool_size) // self.stride + 1

        output = np.zeros((batch, channels, out_len), dtype=np.float32)
        indices = np.zeros((batch, channels, out_len), dtype=np.int32)

        for i in range(out_len):
            start = i * self.stride
            end = start + self.pool_size
            window = x[:, :, start:end]
            output[:, :, i] = np.max(window, axis=2)
            indices[:, :, i] = start + np.argmax(window, axis=2)

        self._cache["input_shape"] = x.shape
        self._cache["indices"] = indices

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        input_shape = self._cache["input_shape"]
        indices = self._cache["indices"]

        grad_input = np.zeros(input_shape, dtype=np.float32)
        batch, channels, out_len = grad_output.shape

        for b in range(batch):
            for c in range(channels):
                for i in range(out_len):
                    idx = indices[b, c, i]
                    grad_input[b, c, idx] += grad_output[b, c, i]

        return grad_input

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class GlobalMaxPool1D:
    """
    Global max pooling over the sequence dimension.

    Takes the maximum value across all positions for each channel.
    Useful as the final layer before classification.
    """

    def __init__(self):
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, channels, length)

        Returns:
            Output of shape (batch, channels)
        """
        self._cache["input"] = x
        self._cache["indices"] = np.argmax(x, axis=2)
        return np.max(x, axis=2)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        x = self._cache["input"]
        indices = self._cache["indices"]

        grad_input = np.zeros_like(x)
        batch, channels = grad_output.shape

        for b in range(batch):
            for c in range(channels):
                idx = indices[b, c]
                grad_input[b, c, idx] = grad_output[b, c]

        return grad_input

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class GlobalAvgPool1D:
    """
    Global average pooling over the sequence dimension.

    Takes the mean value across all positions for each channel.
    """

    def __init__(self):
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, channels, length)

        Returns:
            Output of shape (batch, channels)
        """
        self._cache["length"] = x.shape[2]
        return np.mean(x, axis=2)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        length = self._cache["length"]
        # Expand gradient and divide by length
        return np.expand_dims(grad_output, 2) / length * np.ones((1, 1, length))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

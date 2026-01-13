"""
Dense (fully connected) layers and normalization for genomics models.

These are standard neural network layers adapted for use
in genomics pipelines.
"""

import numpy as np
from typing import Optional


class Dense:
    """
    Fully connected (dense) layer.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        activation: Activation function ('relu', 'sigmoid', 'tanh', None)
        use_bias: Whether to include bias term
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[str] = None,
        use_bias: bool = True
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.use_bias = use_bias

        # He initialization
        scale = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(in_features, out_features).astype(np.float32) * scale

        if use_bias:
            self.bias = np.zeros(out_features, dtype=np.float32)
        else:
            self.bias = None

        self.weights_grad = None
        self.bias_grad = None
        self._cache = {}

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
        elif self.activation == "softmax":
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input of shape (..., in_features)

        Returns:
            Output of shape (..., out_features)
        """
        self._cache["input"] = x

        output = np.dot(x, self.weights)
        if self.use_bias:
            output = output + self.bias

        self._cache["pre_activation"] = output.copy()
        output = self._activate(output)

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
        elif self.activation == "softmax":
            # Softmax gradient is more complex, handled separately
            soft = np.exp(pre_act - np.max(pre_act, axis=-1, keepdims=True))
            soft = soft / np.sum(soft, axis=-1, keepdims=True)
            # Simplified: assume cross-entropy loss follows
            pass

        # Flatten for matrix operations if needed
        original_shape = x.shape
        if x.ndim > 2:
            x_flat = x.reshape(-1, self.in_features)
            grad_flat = grad_output.reshape(-1, self.out_features)
        else:
            x_flat = x
            grad_flat = grad_output

        # Gradient w.r.t. weights and bias
        self.weights_grad = np.dot(x_flat.T, grad_flat)
        if self.use_bias:
            self.bias_grad = np.sum(grad_flat, axis=0)

        # Gradient w.r.t. input
        grad_input = np.dot(grad_flat, self.weights.T)

        if x.ndim > 2:
            grad_input = grad_input.reshape(original_shape)

        return grad_input

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class BatchNorm1D:
    """
    Batch Normalization for 1D data.

    Normalizes activations to zero mean and unit variance,
    then applies learnable scale and shift.

    Args:
        num_features: Number of features/channels
        eps: Small constant for numerical stability
        momentum: Momentum for running statistics
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)

        # Running statistics
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

        self.training = True
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, features) or (batch, features, length)

        Returns:
            Normalized output
        """
        if self.training:
            if x.ndim == 2:
                mean = np.mean(x, axis=0)
                var = np.var(x, axis=0)
            else:  # (batch, features, length)
                mean = np.mean(x, axis=(0, 2))
                var = np.var(x, axis=(0, 2))

            # Update running statistics
            self.running_mean = (
                (1 - self.momentum) * self.running_mean +
                self.momentum * mean
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var +
                self.momentum * var
            )
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        if x.ndim == 2:
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            output = self.gamma * x_norm + self.beta
        else:
            x_norm = (x - mean.reshape(1, -1, 1)) / np.sqrt(var.reshape(1, -1, 1) + self.eps)
            output = self.gamma.reshape(1, -1, 1) * x_norm + self.beta.reshape(1, -1, 1)

        self._cache = {
            "x_norm": x_norm,
            "mean": mean,
            "var": var,
            "input": x,
        }

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        x_norm = self._cache["x_norm"]
        var = self._cache["var"]
        x = self._cache["input"]

        if x.ndim == 2:
            batch_size = x.shape[0]

            # Gradient w.r.t. gamma and beta
            self.gamma_grad = np.sum(grad_output * x_norm, axis=0)
            self.beta_grad = np.sum(grad_output, axis=0)

            # Gradient w.r.t. input
            std_inv = 1.0 / np.sqrt(var + self.eps)
            dx_norm = grad_output * self.gamma
            dvar = np.sum(dx_norm * (x - self._cache["mean"]) * -0.5 * std_inv**3, axis=0)
            dmean = np.sum(dx_norm * -std_inv, axis=0)

            grad_input = (
                dx_norm * std_inv +
                dvar * 2 * (x - self._cache["mean"]) / batch_size +
                dmean / batch_size
            )
        else:
            # Handle 3D case
            batch_size, _, length = x.shape
            n = batch_size * length

            self.gamma_grad = np.sum(grad_output * x_norm, axis=(0, 2))
            self.beta_grad = np.sum(grad_output, axis=(0, 2))

            std_inv = 1.0 / np.sqrt(var.reshape(1, -1, 1) + self.eps)
            dx_norm = grad_output * self.gamma.reshape(1, -1, 1)

            grad_input = dx_norm * std_inv

        return grad_input

    def train(self):
        """Set to training mode."""
        self.training = True

    def eval(self):
        """Set to evaluation mode."""
        self.training = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class LayerNorm:
    """
    Layer Normalization.

    Normalizes across features for each sample independently.
    Often preferred over BatchNorm for sequence models.

    Args:
        normalized_shape: Shape of the normalized dimensions
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5
    ):
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = np.ones(normalized_shape, dtype=np.float32)
        self.beta = np.zeros(normalized_shape, dtype=np.float32)

        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input of shape (..., normalized_shape)

        Returns:
            Normalized output
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        x_norm = (x - mean) / np.sqrt(var + self.eps)
        output = self.gamma * x_norm + self.beta

        self._cache = {
            "x_norm": x_norm,
            "mean": mean,
            "var": var,
            "input": x,
        }

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        x_norm = self._cache["x_norm"]
        var = self._cache["var"]
        x = self._cache["input"]
        mean = self._cache["mean"]

        n = self.normalized_shape

        # Gradient w.r.t. gamma and beta
        self.gamma_grad = np.sum(
            (grad_output * x_norm).reshape(-1, n),
            axis=0
        )
        self.beta_grad = np.sum(
            grad_output.reshape(-1, n),
            axis=0
        )

        # Gradient w.r.t. input
        std_inv = 1.0 / np.sqrt(var + self.eps)
        dx_norm = grad_output * self.gamma

        grad_input = (
            (1.0 / n) * std_inv * (
                n * dx_norm -
                np.sum(dx_norm, axis=-1, keepdims=True) -
                x_norm * np.sum(dx_norm * x_norm, axis=-1, keepdims=True)
            )
        )

        return grad_input

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Dropout:
    """
    Dropout layer for regularization.

    Randomly zeros elements during training to prevent overfitting.

    Args:
        p: Dropout probability
    """

    def __init__(self, p: float = 0.5):
        self.p = p
        self.training = True
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        if not self.training or self.p == 0:
            return x

        mask = (np.random.random(x.shape) > self.p).astype(np.float32)
        self._cache["mask"] = mask

        return x * mask / (1 - self.p)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        if not self.training or self.p == 0:
            return grad_output

        mask = self._cache["mask"]
        return grad_output * mask / (1 - self.p)

    def train(self):
        """Set to training mode."""
        self.training = True

    def eval(self):
        """Set to evaluation mode."""
        self.training = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

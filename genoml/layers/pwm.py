"""
Position Weight Matrix (PWM) layers for motif analysis.

PWMs are fundamental to computational biology, used for
representing and detecting sequence motifs like transcription
factor binding sites. This module provides differentiable
PWM operations for use in neural networks.
"""

import numpy as np
from typing import List, Optional, Tuple, Union


def create_pwm_from_sequences(
    sequences: List[str],
    pseudocount: float = 0.01
) -> np.ndarray:
    """
    Create a Position Weight Matrix from aligned sequences.

    Args:
        sequences: List of aligned sequences (same length)
        pseudocount: Small value to avoid log(0)

    Returns:
        PWM of shape (length, 4) with log-odds scores
    """
    if not sequences:
        raise ValueError("No sequences provided")

    length = len(sequences[0])
    for seq in sequences:
        if len(seq) != length:
            raise ValueError("All sequences must have same length")

    # Count nucleotide frequencies
    nuc_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    counts = np.zeros((length, 4), dtype=np.float32) + pseudocount

    for seq in sequences:
        for i, nuc in enumerate(seq.upper()):
            if nuc in nuc_to_idx:
                counts[i, nuc_to_idx[nuc]] += 1

    # Convert to frequencies
    freq = counts / counts.sum(axis=1, keepdims=True)

    # Background frequency (assume uniform)
    background = 0.25

    # Convert to log-odds (PWM)
    pwm = np.log2(freq / background)

    return pwm


class PWMScanner:
    """
    Scans sequences for matches to a Position Weight Matrix.

    This is a differentiable layer that can be used in neural networks
    for motif detection tasks.

    Args:
        pwm: Position Weight Matrix of shape (motif_length, 4)
        threshold: Score threshold for calling a match
    """

    def __init__(
        self,
        pwm: np.ndarray,
        threshold: Optional[float] = None
    ):
        self.pwm = pwm.astype(np.float32)
        self.motif_length = pwm.shape[0]
        self.threshold = threshold
        self._cache = {}

    @classmethod
    def from_sequences(cls, sequences: List[str], **kwargs):
        """Create PWMScanner from aligned sequences."""
        pwm = create_pwm_from_sequences(sequences)
        return cls(pwm, **kwargs)

    def score_sequence(self, one_hot: np.ndarray) -> np.ndarray:
        """
        Score a one-hot encoded sequence.

        Args:
            one_hot: One-hot encoded sequence (seq_len, 4)

        Returns:
            Array of scores for each position
        """
        seq_len = one_hot.shape[0]
        num_positions = seq_len - self.motif_length + 1

        if num_positions <= 0:
            return np.array([])

        scores = np.zeros(num_positions, dtype=np.float32)

        for i in range(num_positions):
            window = one_hot[i:i + self.motif_length]
            scores[i] = np.sum(window * self.pwm)

        return scores

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for batch of sequences.

        Args:
            x: Input of shape (batch, seq_len, 4) or (batch, 4, seq_len)

        Returns:
            Scores of shape (batch, num_positions)
        """
        # Handle channel-first format
        if x.ndim == 3 and x.shape[1] == 4:
            x = np.transpose(x, (0, 2, 1))

        self._cache["input"] = x

        batch_size = x.shape[0]
        seq_len = x.shape[1]
        num_positions = seq_len - self.motif_length + 1

        scores = np.zeros((batch_size, num_positions), dtype=np.float32)

        for b in range(batch_size):
            scores[b] = self.score_sequence(x[b])

        return scores

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            grad_output: Gradient of shape (batch, num_positions)

        Returns:
            Gradient w.r.t. input
        """
        x = self._cache["input"]
        batch_size, seq_len, _ = x.shape
        num_positions = grad_output.shape[1]

        grad_input = np.zeros_like(x)

        for b in range(batch_size):
            for i in range(num_positions):
                grad_input[b, i:i + self.motif_length] += (
                    self.pwm * grad_output[b, i]
                )

        return grad_input

    def find_matches(
        self,
        sequence: Union[str, np.ndarray],
        threshold: Optional[float] = None
    ) -> List[Tuple[int, float]]:
        """
        Find all positions matching the motif above threshold.

        Args:
            sequence: DNA sequence string or one-hot encoded
            threshold: Score threshold (uses instance threshold if None)

        Returns:
            List of (position, score) tuples
        """
        if threshold is None:
            threshold = self.threshold or 0.0

        if isinstance(sequence, str):
            from genoml.sequence import one_hot_encode
            one_hot = one_hot_encode(sequence)
        else:
            one_hot = sequence

        scores = self.score_sequence(one_hot)
        matches = []

        for i, score in enumerate(scores):
            if score >= threshold:
                matches.append((i, float(score)))

        return matches

    def max_score(self) -> float:
        """Calculate maximum possible score for this PWM."""
        return float(np.sum(np.max(self.pwm, axis=1)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class MotifLayer:
    """
    Learnable motif layer using multiple PWM-like filters.

    Unlike PWMScanner with fixed PWMs, this layer learns
    the motif patterns from data through backpropagation.

    Args:
        num_motifs: Number of motifs to learn
        motif_length: Length of each motif
        input_channels: Number of input channels (4 for DNA)
    """

    def __init__(
        self,
        num_motifs: int,
        motif_length: int,
        input_channels: int = 4
    ):
        self.num_motifs = num_motifs
        self.motif_length = motif_length
        self.input_channels = input_channels

        # Initialize filters (similar to PWMs but learnable)
        # Use small random values centered around 0
        self.filters = np.random.randn(
            num_motifs, motif_length, input_channels
        ).astype(np.float32) * 0.1

        self.filters_grad = None
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, seq_len, 4)

        Returns:
            Motif scores of shape (batch, num_motifs, num_positions)
        """
        self._cache["input"] = x

        batch_size, seq_len, channels = x.shape
        num_positions = seq_len - self.motif_length + 1

        output = np.zeros(
            (batch_size, self.num_motifs, num_positions),
            dtype=np.float32
        )

        for i in range(num_positions):
            window = x[:, i:i + self.motif_length, :]  # (batch, motif_len, 4)
            for m in range(self.num_motifs):
                # Score = sum(window * filter)
                output[:, m, i] = np.sum(
                    window * self.filters[m],
                    axis=(1, 2)
                )

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            grad_output: Gradient of shape (batch, num_motifs, num_positions)

        Returns:
            Gradient w.r.t. input
        """
        x = self._cache["input"]
        batch_size, seq_len, channels = x.shape
        num_positions = grad_output.shape[2]

        # Gradient w.r.t. filters
        self.filters_grad = np.zeros_like(self.filters)

        # Gradient w.r.t. input
        grad_input = np.zeros_like(x)

        for i in range(num_positions):
            window = x[:, i:i + self.motif_length, :]

            for m in range(self.num_motifs):
                # Gradient w.r.t. filter
                self.filters_grad[m] += np.sum(
                    window * grad_output[:, m:m+1, i:i+1],
                    axis=0
                ).reshape(self.motif_length, channels)

                # Gradient w.r.t. input
                grad_input[:, i:i + self.motif_length, :] += (
                    self.filters[m] * grad_output[:, m:m+1, i:i+1]
                )

        return grad_input

    def get_pwms(self) -> List[np.ndarray]:
        """
        Convert learned filters to PWM format for visualization.

        Returns:
            List of PWM arrays (after softmax normalization)
        """
        pwms = []
        for f in self.filters:
            # Apply softmax to get probability-like values
            exp_f = np.exp(f - np.max(f, axis=1, keepdims=True))
            pwm = exp_f / exp_f.sum(axis=1, keepdims=True)
            pwms.append(pwm)
        return pwms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


def pwm_to_consensus(pwm: np.ndarray) -> str:
    """
    Convert PWM to consensus sequence.

    Args:
        pwm: Position Weight Matrix (length, 4)

    Returns:
        Consensus sequence string
    """
    idx_to_nuc = {0: "A", 1: "C", 2: "G", 3: "T"}
    consensus = []

    for position in pwm:
        max_idx = np.argmax(position)
        consensus.append(idx_to_nuc[max_idx])

    return "".join(consensus)


def information_content(pwm: np.ndarray) -> np.ndarray:
    """
    Calculate information content per position in PWM.

    Information content measures how conserved each position is.
    Maximum is 2 bits for DNA (log2(4)).

    Args:
        pwm: PWM in probability format (length, 4)

    Returns:
        Information content per position
    """
    # Convert log-odds back to probabilities if needed
    if np.any(pwm < 0):
        # Assume it's log-odds, convert to probability
        prob = np.power(2, pwm) * 0.25
        prob = prob / prob.sum(axis=1, keepdims=True)
    else:
        prob = pwm / pwm.sum(axis=1, keepdims=True)

    # Shannon entropy
    entropy = -np.sum(prob * np.log2(prob + 1e-10), axis=1)

    # Information content = max entropy - entropy
    max_entropy = 2.0  # log2(4) for DNA
    ic = max_entropy - entropy

    return ic

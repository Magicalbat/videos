"""
Sequence encoding functions for genomic data.

Supports DNA and RNA sequences with various encoding schemes
commonly used in computational biology and machine learning.
"""

import numpy as np
from typing import List, Optional, Union, Dict
from itertools import product

# Standard vocabulary mappings
DNA_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3}
RNA_VOCAB = {"A": 0, "C": 1, "G": 2, "U": 3}
NUCLEOTIDE_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}  # Combined

# Extended IUPAC codes for ambiguous bases
IUPAC_DNA = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "AG", "Y": "CT", "S": "GC", "W": "AT",
    "K": "GT", "M": "AC", "B": "CGT", "D": "AGT",
    "H": "ACT", "V": "ACG", "N": "ACGT",
}


def one_hot_encode(
    sequence: str,
    vocab: Optional[Dict[str, int]] = None,
    handle_unknown: str = "zero"
) -> np.ndarray:
    """
    One-hot encode a DNA/RNA sequence.

    Args:
        sequence: DNA/RNA sequence string (e.g., "ACGT")
        vocab: Mapping from nucleotide to index. Defaults to DNA_VOCAB.
        handle_unknown: How to handle unknown characters:
            - "zero": Return all-zero vector (default)
            - "uniform": Return uniform distribution (0.25 each)
            - "error": Raise ValueError

    Returns:
        numpy array of shape (len(sequence), vocab_size)
        where vocab_size is 4 for standard nucleotides

    Example:
        >>> one_hot_encode("ACG")
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.]])
    """
    if vocab is None:
        vocab = NUCLEOTIDE_VOCAB

    vocab_size = max(vocab.values()) + 1
    seq_len = len(sequence)
    encoding = np.zeros((seq_len, vocab_size), dtype=np.float32)

    sequence = sequence.upper()

    for i, nuc in enumerate(sequence):
        if nuc in vocab:
            encoding[i, vocab[nuc]] = 1.0
        elif handle_unknown == "zero":
            pass  # Already zeros
        elif handle_unknown == "uniform":
            encoding[i, :] = 1.0 / vocab_size
        elif handle_unknown == "error":
            raise ValueError(f"Unknown nucleotide '{nuc}' at position {i}")
        else:
            raise ValueError(f"Invalid handle_unknown mode: {handle_unknown}")

    return encoding


def one_hot_decode(
    encoding: np.ndarray,
    vocab: Optional[Dict[str, int]] = None,
    threshold: float = 0.5
) -> str:
    """
    Decode a one-hot encoded sequence back to string.

    Args:
        encoding: One-hot encoded array of shape (seq_len, vocab_size)
        vocab: Mapping from nucleotide to index. Defaults to DNA_VOCAB.
        threshold: Minimum probability to consider a nucleotide (for soft encodings)

    Returns:
        Decoded sequence string
    """
    if vocab is None:
        vocab = DNA_VOCAB

    # Invert the vocabulary
    idx_to_nuc = {v: k for k, v in vocab.items()}

    sequence = []
    for position in encoding:
        max_idx = np.argmax(position)
        if position[max_idx] >= threshold:
            sequence.append(idx_to_nuc.get(max_idx, "N"))
        else:
            sequence.append("N")

    return "".join(sequence)


def generate_kmers(k: int, alphabet: str = "ACGT") -> List[str]:
    """Generate all possible k-mers from an alphabet."""
    return ["".join(kmer) for kmer in product(alphabet, repeat=k)]


def kmer_encode(
    sequence: str,
    k: int = 3,
    stride: int = 1,
    alphabet: str = "ACGT"
) -> np.ndarray:
    """
    Encode a sequence as a series of k-mer indices.

    Args:
        sequence: DNA/RNA sequence string
        k: Length of k-mers (default: 3 for codons)
        stride: Step size between k-mers (default: 1)
        alphabet: Nucleotide alphabet to use

    Returns:
        numpy array of k-mer indices, shape (num_kmers,)
        where num_kmers = (len(sequence) - k) // stride + 1

    Example:
        >>> kmer_encode("ACGT", k=2)  # Encodes: AC, CG, GT
        array([1, 6, 11])
    """
    sequence = sequence.upper()

    # Build k-mer to index mapping
    all_kmers = generate_kmers(k, alphabet)
    kmer_to_idx = {kmer: i for i, kmer in enumerate(all_kmers)}

    # Extract k-mers
    num_kmers = (len(sequence) - k) // stride + 1
    indices = np.zeros(num_kmers, dtype=np.int32)

    for i in range(num_kmers):
        start = i * stride
        kmer = sequence[start:start + k]
        # Handle unknown k-mers (containing N or other chars)
        indices[i] = kmer_to_idx.get(kmer, -1)

    return indices


def kmer_frequencies(
    sequence: str,
    k: int = 3,
    normalize: bool = True,
    alphabet: str = "ACGT"
) -> np.ndarray:
    """
    Compute k-mer frequency vector for a sequence.

    This is useful for representing sequences of variable length
    as fixed-size feature vectors.

    Args:
        sequence: DNA/RNA sequence string
        k: Length of k-mers
        normalize: If True, return frequencies; if False, return counts
        alphabet: Nucleotide alphabet

    Returns:
        numpy array of shape (4^k,) containing k-mer frequencies/counts

    Example:
        >>> kmer_frequencies("ACGTACGT", k=2, normalize=False)
        # Returns counts for all 16 possible 2-mers
    """
    sequence = sequence.upper()

    all_kmers = generate_kmers(k, alphabet)
    num_kmers = len(all_kmers)
    kmer_to_idx = {kmer: i for i, kmer in enumerate(all_kmers)}

    counts = np.zeros(num_kmers, dtype=np.float32)

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in kmer_to_idx:
            counts[kmer_to_idx[kmer]] += 1

    if normalize and counts.sum() > 0:
        counts /= counts.sum()

    return counts


def pad_sequence(
    encoding: np.ndarray,
    max_length: int,
    pad_value: float = 0.0,
    truncate: str = "right",
    pad_side: str = "right"
) -> np.ndarray:
    """
    Pad or truncate an encoded sequence to a fixed length.

    Args:
        encoding: Encoded sequence, shape (seq_len, features)
        max_length: Target length
        pad_value: Value to use for padding
        truncate: Where to truncate if too long ("left" or "right")
        pad_side: Where to add padding if too short ("left" or "right")

    Returns:
        Padded/truncated array of shape (max_length, features)
    """
    seq_len = encoding.shape[0]

    if seq_len == max_length:
        return encoding

    if seq_len > max_length:
        # Truncate
        if truncate == "right":
            return encoding[:max_length]
        else:
            return encoding[seq_len - max_length:]

    # Pad
    pad_len = max_length - seq_len
    pad_shape = (pad_len,) + encoding.shape[1:]
    padding = np.full(pad_shape, pad_value, dtype=encoding.dtype)

    if pad_side == "right":
        return np.concatenate([encoding, padding], axis=0)
    else:
        return np.concatenate([padding, encoding], axis=0)


def batch_encode(
    sequences: List[str],
    max_length: Optional[int] = None,
    vocab: Optional[Dict[str, int]] = None,
    pad_value: float = 0.0
) -> np.ndarray:
    """
    One-hot encode a batch of sequences with padding.

    Args:
        sequences: List of DNA/RNA sequence strings
        max_length: Maximum sequence length (defaults to longest)
        vocab: Nucleotide vocabulary mapping
        pad_value: Value for padding shorter sequences

    Returns:
        numpy array of shape (batch_size, max_length, vocab_size)
    """
    if max_length is None:
        max_length = max(len(s) for s in sequences)

    if vocab is None:
        vocab = NUCLEOTIDE_VOCAB

    vocab_size = max(vocab.values()) + 1
    batch_size = len(sequences)

    batch = np.full(
        (batch_size, max_length, vocab_size),
        pad_value,
        dtype=np.float32
    )

    for i, seq in enumerate(sequences):
        encoded = one_hot_encode(seq, vocab)
        seq_len = min(len(seq), max_length)
        batch[i, :seq_len, :] = encoded[:seq_len]

    return batch


def positional_encoding(
    seq_length: int,
    d_model: int,
    max_len: int = 10000
) -> np.ndarray:
    """
    Generate sinusoidal positional encodings for sequence positions.

    Useful for transformer-based models on genomic sequences.

    Args:
        seq_length: Length of the sequence
        d_model: Dimension of the model embeddings
        max_len: Maximum sequence length for precomputation

    Returns:
        numpy array of shape (seq_length, d_model)
    """
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_model, 2) * (-np.log(max_len) / d_model)
    )

    pe = np.zeros((seq_length, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe

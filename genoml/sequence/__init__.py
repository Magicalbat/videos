"""
Sequence encoding and processing for genomic data.

This module provides functions for:
- One-hot encoding of DNA/RNA sequences
- K-mer extraction and encoding
- Sequence padding and batching
- Embedding-based representations
"""

from genoml.sequence.encoding import (
    one_hot_encode,
    one_hot_decode,
    kmer_encode,
    kmer_frequencies,
    pad_sequence,
    batch_encode,
    NUCLEOTIDE_VOCAB,
    DNA_VOCAB,
    RNA_VOCAB,
)

__all__ = [
    "one_hot_encode",
    "one_hot_decode",
    "kmer_encode",
    "kmer_frequencies",
    "pad_sequence",
    "batch_encode",
    "NUCLEOTIDE_VOCAB",
    "DNA_VOCAB",
    "RNA_VOCAB",
]

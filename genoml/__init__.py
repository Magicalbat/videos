"""
GenomL: A Machine Learning Library for Quantitative Biology

This package provides tools for:
- DNA/RNA sequence encoding and processing
- Genomic data I/O (FASTA, FASTQ)
- Differentiable layers for sequence analysis
- Loss functions and metrics for genomics tasks
- Utilities for sequence manipulation

Built on top of NumPy for numerical computation with optional
PyTorch integration for gradient-based optimization.
"""

__version__ = "0.1.0"
__author__ = "GenomL Contributors"

from genoml.sequence import (
    one_hot_encode,
    one_hot_decode,
    kmer_encode,
    kmer_frequencies,
    pad_sequence,
    batch_encode,
)

from genoml.io import (
    read_fasta,
    read_fastq,
    write_fasta,
    FastaRecord,
    FastqRecord,
)

from genoml.utils import (
    reverse_complement,
    gc_content,
    translate,
    hamming_distance,
    find_orfs,
)

__all__ = [
    # Sequence encoding
    "one_hot_encode",
    "one_hot_decode",
    "kmer_encode",
    "kmer_frequencies",
    "pad_sequence",
    "batch_encode",
    # I/O
    "read_fasta",
    "read_fastq",
    "write_fasta",
    "FastaRecord",
    "FastqRecord",
    # Utilities
    "reverse_complement",
    "gc_content",
    "translate",
    "hamming_distance",
    "find_orfs",
]

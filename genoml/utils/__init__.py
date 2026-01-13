"""
Sequence manipulation utilities for genomic data.

This module provides common operations for DNA/RNA sequences:
- Reverse complement
- GC content calculation
- Codon translation
- ORF finding
- Sequence comparison
"""

from genoml.utils.sequences import (
    reverse_complement,
    gc_content,
    translate,
    hamming_distance,
    find_orfs,
    CODON_TABLE,
    START_CODONS,
    STOP_CODONS,
)

from genoml.utils.alignment import (
    needleman_wunsch,
    smith_waterman,
    pairwise_identity,
)

__all__ = [
    "reverse_complement",
    "gc_content",
    "translate",
    "hamming_distance",
    "find_orfs",
    "CODON_TABLE",
    "START_CODONS",
    "STOP_CODONS",
    "needleman_wunsch",
    "smith_waterman",
    "pairwise_identity",
]

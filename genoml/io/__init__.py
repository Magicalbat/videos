"""
Genomic file I/O utilities.

This module provides functions for reading and writing common
genomic file formats:
- FASTA: Sequence storage format
- FASTQ: Sequence + quality scores (NGS data)
"""

from genoml.io.fasta import (
    read_fasta,
    write_fasta,
    FastaRecord,
    parse_fasta_string,
)

from genoml.io.fastq import (
    read_fastq,
    write_fastq,
    FastqRecord,
    quality_to_phred,
    phred_to_quality,
)

__all__ = [
    "read_fasta",
    "write_fasta",
    "FastaRecord",
    "parse_fasta_string",
    "read_fastq",
    "write_fastq",
    "FastqRecord",
    "quality_to_phred",
    "phred_to_quality",
]

"""
FASTA file format reader and writer.

FASTA is a text-based format for representing nucleotide or
protein sequences. Each record consists of:
- A header line starting with '>' followed by the sequence ID and description
- One or more lines of sequence data
"""

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union


@dataclass
class FastaRecord:
    """
    Represents a single FASTA record.

    Attributes:
        id: Sequence identifier (first word after '>')
        description: Full description line (everything after '>')
        sequence: The nucleotide/protein sequence
    """
    id: str
    description: str
    sequence: str

    def __len__(self) -> int:
        return len(self.sequence)

    def __str__(self) -> str:
        return f">{self.description}\n{self.sequence}"

    def to_fasta(self, line_width: int = 60) -> str:
        """Format as FASTA string with wrapped sequence lines."""
        lines = [f">{self.description}"]
        for i in range(0, len(self.sequence), line_width):
            lines.append(self.sequence[i:i + line_width])
        return "\n".join(lines)


def _open_file(filepath: Union[str, Path], mode: str = "rt"):
    """Open a file, handling gzip compression if needed."""
    filepath = Path(filepath)
    if filepath.suffix == ".gz":
        return gzip.open(filepath, mode)
    return open(filepath, mode)


def read_fasta(
    filepath: Union[str, Path],
    uppercase: bool = True
) -> Iterator[FastaRecord]:
    """
    Read sequences from a FASTA file.

    Supports both plain text and gzip-compressed files.

    Args:
        filepath: Path to FASTA file (.fasta, .fa, .fna, or .gz)
        uppercase: Convert sequences to uppercase

    Yields:
        FastaRecord objects

    Example:
        >>> for record in read_fasta("sequences.fasta"):
        ...     print(f"{record.id}: {len(record)} bp")
    """
    with _open_file(filepath, "rt") as f:
        yield from parse_fasta_string(f.read(), uppercase=uppercase)


def parse_fasta_string(
    content: str,
    uppercase: bool = True
) -> Iterator[FastaRecord]:
    """
    Parse FASTA format from a string.

    Args:
        content: FASTA formatted string
        uppercase: Convert sequences to uppercase

    Yields:
        FastaRecord objects
    """
    current_header = None
    current_sequence: List[str] = []

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith(">"):
            # Save previous record if exists
            if current_header is not None:
                seq = "".join(current_sequence)
                if uppercase:
                    seq = seq.upper()
                seq_id = current_header.split()[0] if current_header else ""
                yield FastaRecord(
                    id=seq_id,
                    description=current_header,
                    sequence=seq
                )

            # Start new record
            current_header = line[1:].strip()
            current_sequence = []
        else:
            current_sequence.append(line)

    # Don't forget the last record
    if current_header is not None:
        seq = "".join(current_sequence)
        if uppercase:
            seq = seq.upper()
        seq_id = current_header.split()[0] if current_header else ""
        yield FastaRecord(
            id=seq_id,
            description=current_header,
            sequence=seq
        )


def write_fasta(
    records: Union[FastaRecord, List[FastaRecord], Iterator[FastaRecord]],
    filepath: Union[str, Path],
    line_width: int = 60,
    compress: bool = False
) -> None:
    """
    Write sequences to a FASTA file.

    Args:
        records: Single record or iterable of FastaRecord objects
        filepath: Output file path
        line_width: Number of characters per sequence line
        compress: If True, write gzip-compressed file

    Example:
        >>> records = [FastaRecord("seq1", "seq1 example", "ACGT")]
        >>> write_fasta(records, "output.fasta")
    """
    if isinstance(records, FastaRecord):
        records = [records]

    filepath = Path(filepath)
    if compress and not filepath.suffix == ".gz":
        filepath = Path(str(filepath) + ".gz")

    mode = "wt"
    opener = gzip.open if compress or filepath.suffix == ".gz" else open

    with opener(filepath, mode) as f:
        for record in records:
            f.write(record.to_fasta(line_width) + "\n")


def load_fasta_dict(
    filepath: Union[str, Path],
    uppercase: bool = True
) -> dict:
    """
    Load FASTA file as dictionary mapping IDs to sequences.

    Args:
        filepath: Path to FASTA file
        uppercase: Convert sequences to uppercase

    Returns:
        Dictionary mapping sequence IDs to sequences
    """
    return {
        record.id: record.sequence
        for record in read_fasta(filepath, uppercase=uppercase)
    }

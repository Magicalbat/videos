"""
FASTQ file format reader and writer.

FASTQ is a text-based format for storing nucleotide sequences
along with quality scores. Each record consists of 4 lines:
1. Header line starting with '@' followed by sequence ID
2. Sequence line
3. '+' line (optionally followed by the ID again)
4. Quality line (ASCII-encoded Phred scores)
"""

import gzip
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union


# Phred quality score encoding offsets
PHRED33_OFFSET = 33  # Sanger/Illumina 1.8+
PHRED64_OFFSET = 64  # Illumina 1.3-1.7


@dataclass
class FastqRecord:
    """
    Represents a single FASTQ record.

    Attributes:
        id: Sequence identifier
        sequence: The nucleotide sequence
        quality: Quality string (ASCII-encoded)
        description: Optional description after the ID
    """
    id: str
    sequence: str
    quality: str
    description: str = ""

    def __len__(self) -> int:
        return len(self.sequence)

    def __str__(self) -> str:
        header = f"@{self.id}"
        if self.description:
            header += f" {self.description}"
        return f"{header}\n{self.sequence}\n+\n{self.quality}"

    def quality_scores(self, offset: int = PHRED33_OFFSET) -> np.ndarray:
        """
        Convert quality string to numeric Phred scores.

        Args:
            offset: ASCII offset (33 for Phred+33, 64 for Phred+64)

        Returns:
            numpy array of integer quality scores
        """
        return np.array([ord(c) - offset for c in self.quality], dtype=np.int32)

    def mean_quality(self, offset: int = PHRED33_OFFSET) -> float:
        """Calculate mean quality score."""
        scores = self.quality_scores(offset)
        return float(np.mean(scores))

    def error_probabilities(self, offset: int = PHRED33_OFFSET) -> np.ndarray:
        """
        Convert quality scores to error probabilities.

        P(error) = 10^(-Q/10)

        Returns:
            numpy array of error probabilities
        """
        scores = self.quality_scores(offset)
        return np.power(10, -scores / 10)

    def trim_quality(
        self,
        min_quality: int = 20,
        offset: int = PHRED33_OFFSET,
        window_size: int = 4
    ) -> "FastqRecord":
        """
        Trim low-quality bases from the 3' end.

        Uses a sliding window approach to find where quality drops.

        Args:
            min_quality: Minimum average quality in window
            offset: Phred offset
            window_size: Size of sliding window

        Returns:
            New FastqRecord with trimmed sequence
        """
        scores = self.quality_scores(offset)

        # Find trim position
        trim_pos = len(scores)
        for i in range(len(scores) - window_size, -1, -1):
            window_mean = np.mean(scores[i:i + window_size])
            if window_mean >= min_quality:
                trim_pos = i + window_size
                break
            trim_pos = i

        return FastqRecord(
            id=self.id,
            sequence=self.sequence[:trim_pos],
            quality=self.quality[:trim_pos],
            description=self.description
        )


def quality_to_phred(quality_string: str, offset: int = PHRED33_OFFSET) -> List[int]:
    """
    Convert quality string to Phred scores.

    Args:
        quality_string: ASCII-encoded quality string
        offset: Phred offset (33 or 64)

    Returns:
        List of integer Phred scores
    """
    return [ord(c) - offset for c in quality_string]


def phred_to_quality(phred_scores: List[int], offset: int = PHRED33_OFFSET) -> str:
    """
    Convert Phred scores to quality string.

    Args:
        phred_scores: List of integer Phred scores
        offset: Phred offset (33 or 64)

    Returns:
        ASCII-encoded quality string
    """
    return "".join(chr(score + offset) for score in phred_scores)


def _open_file(filepath: Union[str, Path], mode: str = "rt"):
    """Open a file, handling gzip compression if needed."""
    filepath = Path(filepath)
    if filepath.suffix == ".gz":
        return gzip.open(filepath, mode)
    return open(filepath, mode)


def read_fastq(
    filepath: Union[str, Path],
    uppercase: bool = True
) -> Iterator[FastqRecord]:
    """
    Read sequences from a FASTQ file.

    Supports both plain text and gzip-compressed files.

    Args:
        filepath: Path to FASTQ file (.fastq, .fq, or .gz)
        uppercase: Convert sequences to uppercase

    Yields:
        FastqRecord objects

    Example:
        >>> for record in read_fastq("reads.fastq.gz"):
        ...     if record.mean_quality() > 20:
        ...         print(record.id)
    """
    with _open_file(filepath, "rt") as f:
        while True:
            # Read 4 lines at a time
            header = f.readline().strip()
            if not header:
                break

            sequence = f.readline().strip()
            plus_line = f.readline().strip()
            quality = f.readline().strip()

            if not header.startswith("@"):
                raise ValueError(f"Invalid FASTQ header: {header}")

            # Parse header
            header = header[1:]  # Remove @
            parts = header.split(None, 1)
            seq_id = parts[0]
            description = parts[1] if len(parts) > 1 else ""

            if uppercase:
                sequence = sequence.upper()

            yield FastqRecord(
                id=seq_id,
                sequence=sequence,
                quality=quality,
                description=description
            )


def write_fastq(
    records: Union[FastqRecord, List[FastqRecord], Iterator[FastqRecord]],
    filepath: Union[str, Path],
    compress: bool = False
) -> None:
    """
    Write sequences to a FASTQ file.

    Args:
        records: Single record or iterable of FastqRecord objects
        filepath: Output file path
        compress: If True, write gzip-compressed file

    Example:
        >>> records = [FastqRecord("read1", "ACGT", "IIII")]
        >>> write_fastq(records, "output.fastq")
    """
    if isinstance(records, FastqRecord):
        records = [records]

    filepath = Path(filepath)
    if compress and not filepath.suffix == ".gz":
        filepath = Path(str(filepath) + ".gz")

    opener = gzip.open if compress or filepath.suffix == ".gz" else open

    with opener(filepath, "wt") as f:
        for record in records:
            f.write(str(record) + "\n")


def filter_by_quality(
    records: Iterator[FastqRecord],
    min_mean_quality: float = 20.0,
    min_length: int = 0,
    offset: int = PHRED33_OFFSET
) -> Iterator[FastqRecord]:
    """
    Filter FASTQ records by quality and length.

    Args:
        records: Iterator of FastqRecord objects
        min_mean_quality: Minimum mean quality score
        min_length: Minimum sequence length
        offset: Phred offset for quality calculation

    Yields:
        FastqRecord objects passing the filters
    """
    for record in records:
        if len(record) < min_length:
            continue
        if record.mean_quality(offset) < min_mean_quality:
            continue
        yield record


def paired_end_reader(
    filepath1: Union[str, Path],
    filepath2: Union[str, Path],
    uppercase: bool = True
) -> Iterator[tuple]:
    """
    Read paired-end FASTQ files simultaneously.

    Args:
        filepath1: Path to R1 (forward) reads
        filepath2: Path to R2 (reverse) reads
        uppercase: Convert sequences to uppercase

    Yields:
        Tuples of (R1 record, R2 record)
    """
    r1_reader = read_fastq(filepath1, uppercase)
    r2_reader = read_fastq(filepath2, uppercase)

    for r1, r2 in zip(r1_reader, r2_reader):
        yield (r1, r2)

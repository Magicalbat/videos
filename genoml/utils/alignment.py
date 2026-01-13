"""
Sequence alignment algorithms.

Implements classic dynamic programming algorithms for
pairwise sequence alignment.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class AlignmentResult:
    """Result of a pairwise alignment."""
    aligned_seq1: str
    aligned_seq2: str
    score: float
    start1: int
    end1: int
    start2: int
    end2: int

    def __str__(self) -> str:
        """Pretty print the alignment."""
        lines = []
        # Create match line
        match_line = ""
        for c1, c2 in zip(self.aligned_seq1, self.aligned_seq2):
            if c1 == c2 and c1 != "-":
                match_line += "|"
            elif c1 == "-" or c2 == "-":
                match_line += " "
            else:
                match_line += "."

        # Split into chunks for display
        chunk_size = 60
        for i in range(0, len(self.aligned_seq1), chunk_size):
            lines.append(self.aligned_seq1[i:i + chunk_size])
            lines.append(match_line[i:i + chunk_size])
            lines.append(self.aligned_seq2[i:i + chunk_size])
            lines.append("")

        lines.append(f"Score: {self.score}")
        return "\n".join(lines)


# Default scoring matrices
DNA_MATCH_SCORE = 2
DNA_MISMATCH_SCORE = -1
GAP_OPEN_PENALTY = -5
GAP_EXTEND_PENALTY = -1


def needleman_wunsch(
    seq1: str,
    seq2: str,
    match_score: int = DNA_MATCH_SCORE,
    mismatch_score: int = DNA_MISMATCH_SCORE,
    gap_penalty: int = GAP_OPEN_PENALTY
) -> AlignmentResult:
    """
    Global alignment using Needleman-Wunsch algorithm.

    Args:
        seq1: First sequence
        seq2: Second sequence
        match_score: Score for matching bases
        mismatch_score: Score for mismatching bases
        gap_penalty: Penalty for gaps (linear gap model)

    Returns:
        AlignmentResult with aligned sequences and score

    Example:
        >>> result = needleman_wunsch("ACGT", "ACT")
        >>> print(result.aligned_seq1)
        ACGT
    """
    seq1 = seq1.upper()
    seq2 = seq2.upper()

    m, n = len(seq1), len(seq2)

    # Initialize scoring matrix
    score_matrix = np.zeros((m + 1, n + 1), dtype=np.float32)

    # Initialize first row and column
    for i in range(m + 1):
        score_matrix[i, 0] = i * gap_penalty
    for j in range(n + 1):
        score_matrix[0, j] = j * gap_penalty

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                diag_score = score_matrix[i - 1, j - 1] + match_score
            else:
                diag_score = score_matrix[i - 1, j - 1] + mismatch_score

            up_score = score_matrix[i - 1, j] + gap_penalty
            left_score = score_matrix[i, j - 1] + gap_penalty

            score_matrix[i, j] = max(diag_score, up_score, left_score)

    # Traceback
    aligned1, aligned2 = [], []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if seq1[i - 1] == seq2[j - 1]:
                current_score = match_score
            else:
                current_score = mismatch_score

            if score_matrix[i, j] == score_matrix[i - 1, j - 1] + current_score:
                aligned1.append(seq1[i - 1])
                aligned2.append(seq2[j - 1])
                i -= 1
                j -= 1
                continue

        if i > 0 and score_matrix[i, j] == score_matrix[i - 1, j] + gap_penalty:
            aligned1.append(seq1[i - 1])
            aligned2.append("-")
            i -= 1
        else:
            aligned1.append("-")
            aligned2.append(seq2[j - 1])
            j -= 1

    return AlignmentResult(
        aligned_seq1="".join(reversed(aligned1)),
        aligned_seq2="".join(reversed(aligned2)),
        score=float(score_matrix[m, n]),
        start1=0,
        end1=m,
        start2=0,
        end2=n
    )


def smith_waterman(
    seq1: str,
    seq2: str,
    match_score: int = DNA_MATCH_SCORE,
    mismatch_score: int = DNA_MISMATCH_SCORE,
    gap_penalty: int = GAP_OPEN_PENALTY
) -> AlignmentResult:
    """
    Local alignment using Smith-Waterman algorithm.

    Finds the best local alignment between two sequences.

    Args:
        seq1: First sequence
        seq2: Second sequence
        match_score: Score for matching bases
        mismatch_score: Score for mismatching bases
        gap_penalty: Penalty for gaps

    Returns:
        AlignmentResult with aligned sequences and score

    Example:
        >>> result = smith_waterman("AAACGT", "CGT")
        >>> print(result.aligned_seq2)
        CGT
    """
    seq1 = seq1.upper()
    seq2 = seq2.upper()

    m, n = len(seq1), len(seq2)

    # Initialize scoring matrix (all zeros for local alignment)
    score_matrix = np.zeros((m + 1, n + 1), dtype=np.float32)

    # Track maximum score position
    max_score = 0
    max_pos = (0, 0)

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                diag_score = score_matrix[i - 1, j - 1] + match_score
            else:
                diag_score = score_matrix[i - 1, j - 1] + mismatch_score

            up_score = score_matrix[i - 1, j] + gap_penalty
            left_score = score_matrix[i, j - 1] + gap_penalty

            # Local alignment: don't go below zero
            score_matrix[i, j] = max(0, diag_score, up_score, left_score)

            if score_matrix[i, j] > max_score:
                max_score = score_matrix[i, j]
                max_pos = (i, j)

    # Traceback from maximum score
    aligned1, aligned2 = [], []
    i, j = max_pos
    end1, end2 = i, j

    while i > 0 and j > 0 and score_matrix[i, j] > 0:
        if seq1[i - 1] == seq2[j - 1]:
            current_score = match_score
        else:
            current_score = mismatch_score

        if score_matrix[i, j] == score_matrix[i - 1, j - 1] + current_score:
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif score_matrix[i, j] == score_matrix[i - 1, j] + gap_penalty:
            aligned1.append(seq1[i - 1])
            aligned2.append("-")
            i -= 1
        else:
            aligned1.append("-")
            aligned2.append(seq2[j - 1])
            j -= 1

    start1, start2 = i, j

    return AlignmentResult(
        aligned_seq1="".join(reversed(aligned1)),
        aligned_seq2="".join(reversed(aligned2)),
        score=float(max_score),
        start1=start1,
        end1=end1,
        start2=start2,
        end2=end2
    )


def pairwise_identity(seq1: str, seq2: str) -> float:
    """
    Calculate pairwise sequence identity after alignment.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Identity as fraction (0 to 1)
    """
    result = needleman_wunsch(seq1, seq2)

    matches = 0
    total = 0

    for c1, c2 in zip(result.aligned_seq1, result.aligned_seq2):
        if c1 != "-" or c2 != "-":
            total += 1
            if c1 == c2:
                matches += 1

    if total == 0:
        return 0.0

    return matches / total


def multiple_sequence_alignment_score(
    sequences: list,
    gap_penalty: int = GAP_OPEN_PENALTY
) -> float:
    """
    Calculate sum-of-pairs score for aligned sequences.

    Args:
        sequences: List of aligned sequences (same length)
        gap_penalty: Penalty for gaps

    Returns:
        Sum of pairwise alignment scores
    """
    if not sequences:
        return 0.0

    n_seqs = len(sequences)
    seq_len = len(sequences[0])

    total_score = 0.0

    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            for k in range(seq_len):
                c1 = sequences[i][k]
                c2 = sequences[j][k]

                if c1 == "-" and c2 == "-":
                    continue
                elif c1 == "-" or c2 == "-":
                    total_score += gap_penalty
                elif c1 == c2:
                    total_score += DNA_MATCH_SCORE
                else:
                    total_score += DNA_MISMATCH_SCORE

    return total_score

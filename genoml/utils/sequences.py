"""
Core sequence manipulation utilities.

Functions for working with DNA and RNA sequences including
complementation, translation, and sequence analysis.
"""

import re
from typing import List, Tuple, Optional, Dict

# DNA complement mapping
DNA_COMPLEMENT = {
    "A": "T", "T": "A", "G": "C", "C": "G",
    "a": "t", "t": "a", "g": "c", "c": "g",
    "N": "N", "n": "n",
    # IUPAC ambiguity codes
    "R": "Y", "Y": "R", "S": "S", "W": "W",
    "K": "M", "M": "K", "B": "V", "V": "B",
    "D": "H", "H": "D",
}

# RNA complement mapping
RNA_COMPLEMENT = {
    "A": "U", "U": "A", "G": "C", "C": "G",
    "a": "u", "u": "a", "g": "c", "c": "g",
    "N": "N", "n": "n",
}

# Standard genetic code (DNA codons)
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

START_CODONS = {"ATG"}
STOP_CODONS = {"TAA", "TAG", "TGA"}


def reverse_complement(sequence: str, rna: bool = False) -> str:
    """
    Get the reverse complement of a DNA or RNA sequence.

    Args:
        sequence: DNA or RNA sequence string
        rna: If True, treat as RNA (use U instead of T)

    Returns:
        Reverse complement sequence

    Example:
        >>> reverse_complement("ACGT")
        'ACGT'
        >>> reverse_complement("AACG")
        'CGTT'
    """
    complement_map = RNA_COMPLEMENT if rna else DNA_COMPLEMENT

    # Handle unknown characters
    def get_complement(base: str) -> str:
        return complement_map.get(base, base)

    complemented = "".join(get_complement(base) for base in sequence)
    return complemented[::-1]


def gc_content(sequence: str) -> float:
    """
    Calculate the GC content (fraction) of a sequence.

    Args:
        sequence: DNA or RNA sequence

    Returns:
        GC content as a fraction between 0 and 1

    Example:
        >>> gc_content("ACGT")
        0.5
        >>> gc_content("AAAA")
        0.0
    """
    sequence = sequence.upper()
    gc_count = sequence.count("G") + sequence.count("C")
    total = len(sequence)

    if total == 0:
        return 0.0

    return gc_count / total


def gc_content_sliding(
    sequence: str,
    window_size: int = 100,
    step: int = 1
) -> List[float]:
    """
    Calculate GC content using a sliding window.

    Args:
        sequence: DNA sequence
        window_size: Size of the sliding window
        step: Step size between windows

    Returns:
        List of GC content values for each window
    """
    gc_values = []
    for i in range(0, len(sequence) - window_size + 1, step):
        window = sequence[i:i + window_size]
        gc_values.append(gc_content(window))
    return gc_values


def translate(
    sequence: str,
    codon_table: Optional[Dict[str, str]] = None,
    stop_symbol: str = "*",
    to_stop: bool = False
) -> str:
    """
    Translate a DNA sequence to protein.

    Args:
        sequence: DNA sequence (length should be multiple of 3)
        codon_table: Custom codon table (defaults to standard)
        stop_symbol: Symbol to use for stop codons
        to_stop: If True, stop translation at first stop codon

    Returns:
        Amino acid sequence

    Example:
        >>> translate("ATGGCC")
        'MA'
        >>> translate("ATGTGA", to_stop=True)
        'M'
    """
    if codon_table is None:
        codon_table = CODON_TABLE

    sequence = sequence.upper()
    # Convert RNA to DNA if needed
    sequence = sequence.replace("U", "T")

    protein = []
    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i + 3]
        aa = codon_table.get(codon, "X")  # X for unknown

        if aa == "*":
            if to_stop:
                break
            aa = stop_symbol

        protein.append(aa)

    return "".join(protein)


def hamming_distance(seq1: str, seq2: str) -> int:
    """
    Calculate Hamming distance between two sequences.

    Sequences must be of equal length.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Number of positions where sequences differ

    Example:
        >>> hamming_distance("ACGT", "ACGA")
        1
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")

    return sum(c1 != c2 for c1, c2 in zip(seq1.upper(), seq2.upper()))


def find_orfs(
    sequence: str,
    min_length: int = 100,
    start_codons: Optional[set] = None,
    stop_codons: Optional[set] = None,
    all_frames: bool = True
) -> List[Tuple[int, int, int, str]]:
    """
    Find Open Reading Frames (ORFs) in a DNA sequence.

    Args:
        sequence: DNA sequence
        min_length: Minimum ORF length in nucleotides
        start_codons: Set of start codons (default: ATG)
        stop_codons: Set of stop codons (default: TAA, TAG, TGA)
        all_frames: If True, search all 6 reading frames

    Returns:
        List of tuples (start, end, frame, protein_sequence)
        Frame is 0, 1, 2 for forward strand, -1, -2, -3 for reverse

    Example:
        >>> orfs = find_orfs("ATGAAATGA", min_length=3)
        >>> len(orfs) > 0
        True
    """
    if start_codons is None:
        start_codons = START_CODONS
    if stop_codons is None:
        stop_codons = STOP_CODONS

    sequence = sequence.upper()
    orfs = []

    # Frames to search
    if all_frames:
        frames = [0, 1, 2, -1, -2, -3]
    else:
        frames = [0, 1, 2]

    for frame in frames:
        if frame >= 0:
            seq = sequence
            offset = frame
            strand = 1
        else:
            seq = reverse_complement(sequence)
            offset = abs(frame) - 1
            strand = -1

        # Find start codons
        starts = []
        for i in range(offset, len(seq) - 2, 3):
            codon = seq[i:i + 3]
            if codon in start_codons:
                starts.append(i)

        # For each start, find the next stop codon
        for start in starts:
            for i in range(start, len(seq) - 2, 3):
                codon = seq[i:i + 3]
                if codon in stop_codons:
                    end = i + 3
                    orf_seq = seq[start:end]

                    if len(orf_seq) >= min_length:
                        protein = translate(orf_seq, to_stop=True)

                        # Convert coordinates back to original if reverse strand
                        if strand == -1:
                            orig_start = len(sequence) - end
                            orig_end = len(sequence) - start
                        else:
                            orig_start = start
                            orig_end = end

                        orfs.append((orig_start, orig_end, frame, protein))
                    break

    return sorted(orfs, key=lambda x: x[0])


def find_motif(
    sequence: str,
    motif: str,
    allow_overlap: bool = True
) -> List[int]:
    """
    Find all occurrences of a motif in a sequence.

    Args:
        sequence: DNA/RNA sequence
        motif: Pattern to search for (supports IUPAC codes as regex)
        allow_overlap: Whether to allow overlapping matches

    Returns:
        List of start positions (0-indexed)
    """
    sequence = sequence.upper()
    motif = motif.upper()

    # Convert IUPAC to regex
    iupac_regex = {
        "R": "[AG]", "Y": "[CT]", "S": "[GC]", "W": "[AT]",
        "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]",
        "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]",
    }

    pattern = ""
    for char in motif:
        if char in iupac_regex:
            pattern += iupac_regex[char]
        else:
            pattern += char

    positions = []
    start = 0
    while True:
        match = re.search(pattern, sequence[start:])
        if match is None:
            break

        pos = start + match.start()
        positions.append(pos)

        if allow_overlap:
            start = pos + 1
        else:
            start = pos + len(match.group())

    return positions


def sequence_entropy(sequence: str) -> float:
    """
    Calculate Shannon entropy of a sequence.

    Higher entropy indicates more sequence complexity/diversity.

    Args:
        sequence: DNA/RNA sequence

    Returns:
        Shannon entropy in bits
    """
    import math

    sequence = sequence.upper()
    length = len(sequence)

    if length == 0:
        return 0.0

    # Count nucleotide frequencies
    counts = {}
    for nuc in sequence:
        counts[nuc] = counts.get(nuc, 0) + 1

    # Calculate entropy
    entropy = 0.0
    for count in counts.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def melting_temperature(
    sequence: str,
    method: str = "wallace"
) -> float:
    """
    Estimate the melting temperature (Tm) of a DNA sequence.

    Args:
        sequence: DNA sequence (typically a primer, 10-30 bp)
        method: Calculation method:
            - "wallace": Simple 2(A+T) + 4(G+C) rule
            - "basic": 64.9 + 41*(G+C-16.4)/N rule

    Returns:
        Estimated melting temperature in Celsius
    """
    sequence = sequence.upper()
    length = len(sequence)

    gc_count = sequence.count("G") + sequence.count("C")
    at_count = sequence.count("A") + sequence.count("T")

    if method == "wallace":
        # Wallace rule (for short oligos < 14 bp)
        return 2 * at_count + 4 * gc_count

    elif method == "basic":
        # Basic formula
        if length == 0:
            return 0.0
        return 64.9 + 41 * (gc_count - 16.4) / length

    else:
        raise ValueError(f"Unknown method: {method}")

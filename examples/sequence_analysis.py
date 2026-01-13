#!/usr/bin/env python3
"""
Example: Sequence Analysis with GenomL

This example demonstrates the sequence processing and analysis
capabilities of GenomL:
- Reading FASTA files
- Sequence encoding (one-hot, k-mer)
- Computing sequence properties
- Finding motifs and ORFs
- Sequence alignment
"""

import sys
sys.path.insert(0, '..')

from genoml.sequence import (
    one_hot_encode,
    one_hot_decode,
    kmer_encode,
    kmer_frequencies,
    positional_encoding,
)
from genoml.utils import (
    reverse_complement,
    gc_content,
    translate,
    find_orfs,
    hamming_distance,
)
from genoml.utils.alignment import (
    needleman_wunsch,
    smith_waterman,
    pairwise_identity,
)
from genoml.utils.sequences import (
    find_motif,
    melting_temperature,
    sequence_entropy,
    gc_content_sliding,
)
from genoml.io import FastaRecord, write_fasta


def demo_encoding():
    """Demonstrate sequence encoding methods."""
    print("\n" + "=" * 60)
    print("SEQUENCE ENCODING")
    print("=" * 60)

    # Example sequence
    seq = "ACGTACGTNN"
    print(f"\nInput sequence: {seq}")

    # One-hot encoding
    print("\n1. One-hot encoding:")
    one_hot = one_hot_encode(seq)
    print(f"   Shape: {one_hot.shape}")
    print(f"   Encoding (first 5 positions):")
    print(f"   Position:  A    C    G    T")
    for i, nuc in enumerate(seq[:5]):
        print(f"   {nuc}:        {one_hot[i]}")

    # Decode back
    decoded = one_hot_decode(one_hot)
    print(f"\n   Decoded: {decoded}")

    # K-mer encoding
    print("\n2. K-mer encoding (k=3, codon-based):")
    kmer_idx = kmer_encode(seq[:9], k=3)
    print(f"   K-mer indices: {kmer_idx}")

    print("\n3. K-mer frequency vector (k=2):")
    freq = kmer_frequencies(seq, k=2, normalize=True)
    print(f"   Shape: {freq.shape} (16 possible 2-mers)")
    print(f"   Top 3 frequencies: {sorted(freq, reverse=True)[:3]}")


def demo_sequence_properties():
    """Demonstrate sequence property calculations."""
    print("\n" + "=" * 60)
    print("SEQUENCE PROPERTIES")
    print("=" * 60)

    seq = "ATGCATGCATGCTAGCTGATCGATCGATCGATCG"
    print(f"\nSequence: {seq}")
    print(f"Length: {len(seq)} bp")

    # GC content
    gc = gc_content(seq)
    print(f"\nGC Content: {gc:.2%}")

    # Sliding window GC
    gc_values = gc_content_sliding(seq, window_size=10, step=5)
    print(f"Sliding GC (window=10, step=5): {[f'{x:.2f}' for x in gc_values[:5]]}...")

    # Entropy
    entropy = sequence_entropy(seq)
    print(f"\nShannon Entropy: {entropy:.3f} bits")
    print("  (Max for DNA is 2.0 bits = equal ACGT)")

    # Melting temperature
    primer = "ATGCATGCATGCTAGC"
    tm = melting_temperature(primer, method="basic")
    print(f"\nMelting Temperature (Tm) for primer:")
    print(f"  Primer: {primer}")
    print(f"  Tm: {tm:.1f}°C")


def demo_reverse_complement():
    """Demonstrate reverse complement."""
    print("\n" + "=" * 60)
    print("REVERSE COMPLEMENT")
    print("=" * 60)

    seq = "ATGCGATCGA"
    rc = reverse_complement(seq)

    print(f"\nOriginal:   5'-{seq}-3'")
    print(f"Rev Comp:   3'-{rc[::-1]}-5'")
    print(f"            5'-{rc}-3'")

    # Verify palindrome detection
    palindrome = "GAATTC"  # EcoRI site
    rc_pal = reverse_complement(palindrome)
    print(f"\nPalindrome check (EcoRI site):")
    print(f"  Sequence: {palindrome}")
    print(f"  Rev Comp: {rc_pal}")
    print(f"  Is palindrome: {palindrome == rc_pal}")


def demo_translation():
    """Demonstrate DNA to protein translation."""
    print("\n" + "=" * 60)
    print("TRANSLATION")
    print("=" * 60)

    # Example gene sequence (GFP start)
    dna = "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGA"

    print(f"\nDNA sequence ({len(dna)} bp):")
    print(f"  {dna[:60]}...")

    protein = translate(dna, to_stop=True)
    print(f"\nProtein sequence ({len(protein)} aa):")
    print(f"  {protein}")

    # With stop codon
    protein_full = translate(dna, to_stop=False)
    print(f"\nWith stop codon: {protein_full[-5:]}")


def demo_orf_finding():
    """Demonstrate ORF finding."""
    print("\n" + "=" * 60)
    print("ORF FINDING")
    print("=" * 60)

    # Sequence with embedded ORFs
    seq = "NNNNATGAAACCCGGGTTTAAATGCCCAAAGGGTTTTGATCGATCGATG"

    print(f"\nSequence: {seq}")

    orfs = find_orfs(seq, min_length=9, all_frames=False)

    print(f"\nFound {len(orfs)} ORF(s):")
    for start, end, frame, protein in orfs:
        print(f"\n  Position: {start}-{end} (frame {frame})")
        print(f"  DNA: {seq[start:end]}")
        print(f"  Protein: {protein}")


def demo_motif_finding():
    """Demonstrate motif finding."""
    print("\n" + "=" * 60)
    print("MOTIF FINDING")
    print("=" * 60)

    seq = "ATGCTATAAATGCGCTATAAATATATATAGCGC"

    # Find TATA box
    print(f"\nSequence: {seq}")
    print("\nSearching for TATA box (TATAAA):")

    positions = find_motif(seq, "TATAAA")
    print(f"  Found at positions: {positions}")

    # Find with IUPAC ambiguity
    print("\nSearching for TATAWAW (W = A/T):")
    positions = find_motif(seq, "TATAWAW")
    print(f"  Found at positions: {positions}")


def demo_alignment():
    """Demonstrate sequence alignment."""
    print("\n" + "=" * 60)
    print("SEQUENCE ALIGNMENT")
    print("=" * 60)

    seq1 = "ACGTACGTACGT"
    seq2 = "ACGACGTAGT"

    print(f"\nSequence 1: {seq1}")
    print(f"Sequence 2: {seq2}")

    # Global alignment
    print("\n1. Global Alignment (Needleman-Wunsch):")
    result = needleman_wunsch(seq1, seq2)
    print(f"   Score: {result.score}")
    print(f"   Seq1: {result.aligned_seq1}")
    print(f"   Seq2: {result.aligned_seq2}")

    # Local alignment
    print("\n2. Local Alignment (Smith-Waterman):")
    seq1_local = "NNNNACGTACGTNNNN"
    seq2_local = "ACGTACGT"
    result = smith_waterman(seq1_local, seq2_local)
    print(f"   Score: {result.score}")
    print(f"   Seq1[{result.start1}:{result.end1}]: {result.aligned_seq1}")
    print(f"   Seq2[{result.start2}:{result.end2}]: {result.aligned_seq2}")

    # Pairwise identity
    identity = pairwise_identity(seq1, seq2)
    print(f"\n3. Pairwise Identity: {identity:.2%}")


def demo_hamming():
    """Demonstrate Hamming distance."""
    print("\n" + "=" * 60)
    print("HAMMING DISTANCE")
    print("=" * 60)

    seq1 = "ACGTACGT"
    seq2 = "ACGAACGA"

    print(f"\nSequence 1: {seq1}")
    print(f"Sequence 2: {seq2}")

    dist = hamming_distance(seq1, seq2)
    print(f"\nHamming Distance: {dist}")
    print(f"Identity: {1 - dist/len(seq1):.2%}")

    # Show mismatches
    print("\nMismatches:")
    for i, (a, b) in enumerate(zip(seq1, seq2)):
        if a != b:
            print(f"  Position {i}: {a} -> {b}")


def main():
    print("=" * 60)
    print("GenomL Sequence Analysis Demo")
    print("=" * 60)

    demo_encoding()
    demo_sequence_properties()
    demo_reverse_complement()
    demo_translation()
    demo_orf_finding()
    demo_motif_finding()
    demo_alignment()
    demo_hamming()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

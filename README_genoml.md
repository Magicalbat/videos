# GenomL - Machine Learning for Quantitative Biology

A Python library for applying machine learning to genomics and quantitative biology problems. Built on top of a from-scratch ML implementation in C, GenomL provides tools specifically designed for DNA/RNA sequence analysis.

## Features

### Sequence Encoding
- **One-hot encoding**: Convert DNA/RNA sequences to numerical format
- **K-mer encoding**: Tokenize sequences into k-mers
- **K-mer frequencies**: Fixed-length feature vectors for variable-length sequences
- **Positional encoding**: For transformer-based models

### Genomic Data I/O
- **FASTA reading/writing**: Handle sequence files with headers
- **FASTQ support**: Process NGS data with quality scores
- **Compression support**: Read/write gzip-compressed files

### Sequence Utilities
- **Reverse complement**: Get the reverse complement strand
- **GC content**: Calculate nucleotide composition
- **Translation**: Convert DNA to protein sequences
- **ORF finding**: Identify open reading frames
- **Motif finding**: Search for patterns with IUPAC codes
- **Sequence alignment**: Needleman-Wunsch and Smith-Waterman algorithms

### Differentiable Layers (NumPy-based)
- **Conv1D**: 1D convolutions for motif detection
- **PWM Scanner**: Position Weight Matrix operations
- **Attention**: Self-attention and multi-head attention
- **Dense, BatchNorm, Dropout**: Standard neural network layers
- Full backward pass support for gradient computation

### Loss Functions
- **Classification**: Cross-entropy, focal loss, label smoothing
- **Regression**: MSE, MAE, Huber, Poisson, Negative Binomial
- **Genomics-specific**: Pearson correlation loss

### Evaluation Metrics
- Classification: Accuracy, Precision, Recall, F1, AUROC, AUPR
- Regression: Pearson/Spearman correlation, R², MSE, MAE

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mlfromscratch.git
cd mlfromscratch

# Install in development mode
pip install -e .

# Or just install dependencies
pip install -r requirements.txt
```

## Quick Start

### Sequence Encoding

```python
from genoml.sequence import one_hot_encode, batch_encode

# Encode a single sequence
seq = "ACGTACGT"
encoded = one_hot_encode(seq)  # Shape: (8, 4)

# Batch encode with padding
sequences = ["ACGT", "ACGTACGT", "ACG"]
batch = batch_encode(sequences)  # Shape: (3, 8, 4)
```

### Reading FASTA Files

```python
from genoml.io import read_fasta

for record in read_fasta("sequences.fasta"):
    print(f"{record.id}: {len(record)} bp")
    print(f"GC content: {gc_content(record.sequence):.2%}")
```

### Sequence Analysis

```python
from genoml.utils import reverse_complement, translate, find_orfs

dna = "ATGAAACCCGGGTAA"

# Reverse complement
rc = reverse_complement(dna)

# Translate to protein
protein = translate(dna, to_stop=True)

# Find ORFs
orfs = find_orfs(dna, min_length=9)
```

### Building a Sequence Classifier

```python
from genoml.layers import Conv1D, GlobalMaxPool1D, Dense
from genoml.losses import binary_cross_entropy
from genoml.sequence import batch_encode

# Prepare data
X = batch_encode(sequences)  # (batch, length, 4)

# Build model
conv = Conv1D(in_channels=4, out_channels=32, kernel_size=8)
pool = GlobalMaxPool1D()
dense = Dense(32, 1, activation='sigmoid')

# Forward pass
x = conv(X.transpose(0, 2, 1))
x = pool(x)
pred = dense(x)

# Compute loss
loss, grad = binary_cross_entropy(pred, labels)
```

## Examples

See the `examples/` directory for complete examples:

- `promoter_classification.py`: Train a CNN to classify promoter sequences
- `sequence_analysis.py`: Demonstrate sequence processing utilities

## Project Structure

```
mlfromscratch/
├── genoml/
│   ├── __init__.py
│   ├── sequence/
│   │   ├── __init__.py
│   │   └── encoding.py       # One-hot, k-mer encoding
│   ├── io/
│   │   ├── __init__.py
│   │   ├── fasta.py          # FASTA reader/writer
│   │   └── fastq.py          # FASTQ reader/writer
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── sequences.py      # Sequence operations
│   │   └── alignment.py      # Alignment algorithms
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── conv.py           # Convolutional layers
│   │   ├── pwm.py            # PWM layers
│   │   ├── attention.py      # Attention mechanisms
│   │   └── dense.py          # Dense layers
│   └── losses/
│       ├── __init__.py
│       ├── classification.py # Classification losses
│       ├── regression.py     # Regression losses
│       └── metrics.py        # Evaluation metrics
├── examples/
│   ├── promoter_classification.py
│   └── sequence_analysis.py
├── machine-learning/          # Original C ML library
├── requirements.txt
├── setup.py
└── README_genoml.md
```

## Original C Library

This project builds on a from-scratch machine learning library implemented in C, which includes:

- Matrix operations with manual memory management
- Computation graph-based automatic differentiation
- Basic neural network training (demonstrated with MNIST)
- Arena-based memory allocation

The Python `genoml` package extends this with biology-specific features while maintaining a similar philosophy of implementing things from first principles.

## License

MIT License

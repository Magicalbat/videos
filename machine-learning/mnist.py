"""
Load MNIST and export .mat files for the C program.
Uses only NumPy and the standard library.
"""

import gzip
import urllib.request
import os
import numpy as np

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist"
# Fallback: http://yann.lecun.com/exdb/mnist/ (If official MNIST is not working.)

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download(path, url):
    if not os.path.exists(path):
        print(f"Downloading {path} ...")
        urllib.request.urlretrieve(url, path)
    return path


def load_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28 * 28)  # (N, 784)


def load_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)


def main():
    for name, filename in FILES.items():
        download(filename, f"{BASE}/{filename}")

    train_images = load_images(FILES["train_images"]).astype(np.float32) / 255.0
    train_labels = load_labels(FILES["train_labels"]).astype(np.float32)
    test_images = load_images(FILES["test_images"]).astype(np.float32) / 255.0
    test_labels = load_labels(FILES["test_labels"]).astype(np.float32)

    train_images.tofile("train_images.mat")
    train_labels.tofile("train_labels.mat")
    test_images.tofile("test_images.mat")
    test_labels.tofile("test_labels.mat")

    print(train_images.shape)
    print(train_labels.shape)
    print(test_images.shape)
    print(test_labels.shape)


if __name__ == "__main__":
    main()

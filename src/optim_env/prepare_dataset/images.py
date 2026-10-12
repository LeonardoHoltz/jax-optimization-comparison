"""Download MNIST or Fashion-MNIST into the mnist.npz layout MNISTDataModule reads.

    pixi run python -m optim_env.prepare_dataset.images --name mnist --root data
    pixi run python -m optim_env.prepare_dataset.images --name fashion_mnist --root data

Every file is checked against a pinned SHA-256, so a changed mirror fails loudly.
"""

import argparse
import gzip
import hashlib
import io
import urllib.request
from pathlib import Path

import numpy as np

MIRROR = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
SHA256 = {
    "mnist.npz": "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",
    # Fashion-MNIST, as Keras serves it (same files as zalandoresearch/fashion-mnist).
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}


def fetch(filename) -> bytes:
    with urllib.request.urlopen(MIRROR + filename, timeout=120) as r:
        data = r.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256[filename]:
        raise ValueError(f"{filename}: sha256 {digest} does not match the pinned {SHA256[filename]}")
    return data


def _idx(data: bytes, offset: int) -> np.ndarray:
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8, offset=offset)


def download(name, root) -> Path:
    out = Path(root) / name / "mnist.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if name == "mnist":
        out.write_bytes(fetch("mnist.npz"))
        return out
    if name != "fashion_mnist":
        raise ValueError(f"Unknown image dataset '{name}'")

    # idx format: 16-byte header for images, 8-byte header for labels.
    arrays = {}
    for split, prefix in (("train", "train"), ("test", "t10k")):
        arrays[f"x_{split}"] = _idx(fetch(f"{prefix}-images-idx3-ubyte.gz"), 16).reshape(-1, 28, 28)
        arrays[f"y_{split}"] = _idx(fetch(f"{prefix}-labels-idx1-ubyte.gz"), 8)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    out.write_bytes(buf.getvalue())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, choices=["mnist", "fashion_mnist"])
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    print(f"Wrote {download(args.name, args.root)}")


if __name__ == "__main__":
    main()

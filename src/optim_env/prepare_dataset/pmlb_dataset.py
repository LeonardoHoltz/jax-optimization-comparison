import argparse
from pathlib import Path

import numpy as np
from pmlb import fetch_data


def download(name, root):
    ds_dir = Path(root) / name
    ds_dir.mkdir(parents=True, exist_ok=True)

    X, y = fetch_data(dataset_name=name, return_X_y=True, dropna=True)
    X = X.astype(np.float32)

    np.save(ds_dir / "X.npy", X)
    np.save(ds_dir / "y.npy", y)
    return ds_dir


def main():
    parser = argparse.ArgumentParser(description="Download a PMLB dataset for offline use")
    parser.add_argument("--name", required=True, help="PMLB dataset name, e.g. mushroom")
    parser.add_argument("--root", required=True, help="Value of the dataset config's `root` field")
    args = parser.parse_args()

    ds_dir = download(args.name, args.root)
    print(f"Downloaded PMLB dataset '{args.name}' to {ds_dir}")


if __name__ == "__main__":
    main()

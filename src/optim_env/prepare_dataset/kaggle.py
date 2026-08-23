import argparse
from pathlib import Path

import kagglehub


def _find_csv(root_dir):
    csv_files = sorted(root_dir.rglob("*.csv"))
    if not csv_files:
        return None
    return csv_files[0]


def download(dataset_slug, name, root):
    ds_dir = Path(root) / name
    ds_dir.mkdir(parents=True, exist_ok=True)

    downloaded_dir = Path(kagglehub.dataset_download(dataset_slug))
    csv_path = _find_csv(downloaded_dir)
    if csv_path is None:
        raise FileNotFoundError(f"No CSV file was found in the downloaded Kaggle dataset '{dataset_slug}'")

    destination_path = ds_dir / csv_path.name
    destination_path.write_bytes(csv_path.read_bytes())
    return destination_path


def main():
    parser = argparse.ArgumentParser(description="Download a Kaggle dataset for offline use")
    parser.add_argument("--dataset", required=True, help="Kaggle dataset slug, e.g. uciml/adult-census-income")
    parser.add_argument("--name", required=True, help="Local dataset name, e.g. adult_income")
    parser.add_argument("--root", required=True, help="Value of the dataset config's `root` field")
    args = parser.parse_args()

    csv_path = download(args.dataset, args.name, args.root)
    print(f"Downloaded Kaggle dataset '{args.dataset}' to {csv_path}")


if __name__ == "__main__":
    main()

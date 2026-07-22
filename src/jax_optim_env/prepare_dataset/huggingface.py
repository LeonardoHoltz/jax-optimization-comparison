import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def download(repo_id, name, root, repo_type="dataset"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=root,
        allow_patterns=[f"{name}/*"],
    )
    return root / name


def main():
    parser = argparse.ArgumentParser(description="Download a dataset (or one of its subfolders) from the Hugging Face Hub")
    parser.add_argument("--repo-id", required=True, help="e.g. rototoHF/tabm-data")
    parser.add_argument("--name", required=True, help="Subfolder within the repo to download, e.g. microsoft")
    parser.add_argument("--root", required=True, help="Value of the dataset config's `root` field")
    parser.add_argument("--repo-type", default="dataset")
    args = parser.parse_args()

    ds_dir = download(args.repo_id, args.name, args.root, args.repo_type)
    print(f"Downloaded {args.repo_id}/{args.name} to {ds_dir}")


if __name__ == "__main__":
    main()

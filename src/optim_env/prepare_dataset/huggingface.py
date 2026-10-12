"""Download TabM benchmark datasets from their Hub mirror, at a pinned revision.

    pixi run python -m optim_env.prepare_dataset.huggingface --name california adult --root data/tabm

The mirror is one 1 GB data.tar with a folder per dataset; only the named folders are
extracted to <root>/<name>/.
"""

import argparse
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "rototoHF/tabm-data"
REVISION = "c2d6629df1192e714e2b35ac4c6df925d938adb0"


def download(names, root, cache=None):
    tar_path = hf_hub_download(REPO, "data.tar", repo_type="dataset", revision=REVISION, cache_dir=cache)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if m.name.split("/")[0] in names]
        missing = set(names) - {m.name.split("/")[0] for m in members}
        if missing:
            raise ValueError(f"Not in {REPO}: {sorted(missing)}")
        # Python 3.11.0 has no tar extraction filters, so check paths ourselves.
        for m in members:
            if not (m.isfile() or m.isdir()) or m.name.startswith("/") or ".." in m.name.split("/"):
                raise ValueError(f"Refusing unsafe tar entry {m.name!r}")
        tar.extractall(root, members=members)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", nargs="+", required=True, help="Dataset folders, e.g. california adult")
    parser.add_argument("--root", required=True, help="Value of the dataset config's `root` field")
    args = parser.parse_args()
    download(args.name, args.root)
    print(f"Extracted {', '.join(args.name)} to {args.root}")


if __name__ == "__main__":
    main()

"""Download and tokenize AG News or SST-2 into fixed train/val/test arrays.

    pixi run python -m optim_env.prepare_dataset.text --name ag_news sst2 --root data/text

Writes <root>/<name>/: {split}_ids.npy (int32 token ids, 0 = padding, 1 = unknown),
{split}_y.npy, vocab.txt, glove100.npy (one 100-d GloVe row per vocab entry, zeros when
GloVe lacks the word) and meta.json (sources, revisions, sizes).

Splits:
- AG News: the official 120K train / 7.6K test; validation is a fixed 10% of train.
- SST-2: GLUE train (phrases + sentences, as Kim 2014 trained), GLUE validation (872),
  and the 1,821-sentence labeled test set from SetFit/sst2. GLUE's own test has no labels.
"""

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# Pinned Hub revisions, so re-running produces identical files.
SOURCES = {
    "ag_news": ("fancyzhx/ag_news", "eb185aade064a813bc0b7f42de02595523103ca4"),
    "glue": ("nyu-mll/glue", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"),
    "setfit_sst2": ("SetFit/sst2", "00ea8ccb7a54b4e3780a3e51aa3f80361ff849c0"),
    "glove": ("stanfordnlp/glove", "1db2080b2d94def6e5b0386a523102f9d8849e9d"),
}
SETTINGS = {
    "ag_news": {"max_len": 100, "max_vocab": 50_000},
    "sst2": {"max_len": 64, "max_vocab": None},
}
VAL_FRACTION = 0.1
SPLIT_SEED = 0
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def _get(source, filename, cache):
    repo, revision = SOURCES[source]
    repo_type = "model" if source == "glove" else "dataset"
    return hf_hub_download(repo, filename, repo_type=repo_type, revision=revision, cache_dir=cache)


def load_splits(name, cache) -> dict[str, pd.DataFrame]:
    """{split: DataFrame(text, label)}"""
    if name == "ag_news":
        train = pd.read_parquet(_get("ag_news", "data/train-00000-of-00001.parquet", cache))
        test = pd.read_parquet(_get("ag_news", "data/test-00000-of-00001.parquet", cache))
        order = np.random.default_rng(SPLIT_SEED).permutation(len(train))
        n_val = int(len(train) * VAL_FRACTION)
        val, train = train.iloc[order[:n_val]], train.iloc[order[n_val:]]
        return {"train": train, "val": val, "test": test}
    if name == "sst2":
        train = pd.read_parquet(_get("glue", "sst2/train-00000-of-00001.parquet", cache))
        val = pd.read_parquet(_get("glue", "sst2/validation-00000-of-00001.parquet", cache))
        test = pd.read_json(_get("setfit_sst2", "test.jsonl", cache), lines=True)
        rename = {"sentence": "text"}
        return {"train": train.rename(columns=rename), "val": val.rename(columns=rename), "test": test}
    raise ValueError(f"Unknown text dataset '{name}'")


def build_vocab(texts, max_vocab) -> list[str]:
    counts = Counter(tok for t in texts for tok in tokenize(t))
    # Ties broken alphabetically so the vocabulary doesn't depend on input order.
    words = [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return ["<pad>", "<unk>"] + words[:max_vocab]


def encode(texts, index, max_len) -> np.ndarray:
    ids = np.zeros((len(texts), max_len), dtype=np.int32)
    for i, t in enumerate(texts):
        toks = [index.get(tok, 1) for tok in tokenize(t)][:max_len]
        ids[i, :len(toks)] = toks
    return ids


def glove_matrix(vocab, cache) -> tuple[np.ndarray, float]:
    """100-d GloVe 6B rows for the vocabulary, read straight from the 862 MB zip."""
    path = Path(_get("glove", "glove.6B.zip", cache))
    index = {w: i for i, w in enumerate(vocab)}
    matrix = np.zeros((len(vocab), 100), dtype=np.float32)
    found = 0
    with zipfile.ZipFile(path) as z, z.open("glove.6B.100d.txt") as f:
        for line in f:
            word, *values = line.decode("utf-8").rstrip().split(" ")
            i = index.get(word)
            if i is not None and i > 1:
                matrix[i] = np.asarray(values, dtype=np.float32)
                found += 1
    return matrix, found / (len(vocab) - 2)


def prepare(name, root, cache) -> Path:
    out = Path(root) / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = SETTINGS[name]
    splits = load_splits(name, cache)
    vocab = build_vocab(splits["train"]["text"], cfg["max_vocab"])
    index = {w: i for i, w in enumerate(vocab)}

    meta = {"sources": SOURCES, "split_seed": SPLIT_SEED, "max_len": cfg["max_len"],
            "vocab_size": len(vocab), "num_classes": int(splits["train"]["label"].nunique())}
    for split, df in splits.items():
        np.save(out / f"{split}_ids.npy", encode(df["text"].tolist(), index, cfg["max_len"]))
        np.save(out / f"{split}_y.npy", df["label"].to_numpy(dtype=np.int32))
        meta[f"{split}_size"] = len(df)
    (out / "vocab.txt").write_text("\n".join(vocab) + "\n")

    matrix, coverage = glove_matrix(vocab, cache)
    np.save(out / "glove100.npy", matrix)
    meta["glove_coverage"] = round(coverage, 4)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, nargs="+", choices=list(SETTINGS))
    parser.add_argument("--root", default="data/text")
    parser.add_argument("--cache", default="data/text/_hf_cache", help="Where raw downloads are kept")
    args = parser.parse_args()
    for name in args.name:
        out = prepare(name, args.root, args.cache)
        print(json.dumps(json.loads((out / "meta.json").read_text()), indent=2))
    # Only the per-vocab slice is kept; drop the 862 MB zip.
    Path(_get("glove", "glove.6B.zip", args.cache)).resolve().unlink()


if __name__ == "__main__":
    main()

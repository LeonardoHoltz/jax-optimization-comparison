"""Download the TabM paper's published MLP results and save them as reference targets.

For each dataset this records the mean/std test score over the 15 published seeds and the
tuned MLP (blocks, width, dropout, AdamW lr and weight decay), which the "paper"
architecture configs copy.

    pixi run python -m optim_env.prepare_dataset.tabm_references
"""

import argparse
import json
import statistics
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/yandex-research/tabm/{commit}/paper/exp/mlp/{dataset}/0-evaluation/{seed}/report.json"
# Pinned so the targets can't change under us.
COMMIT = "28e47ae301c92ec37787dde1ce923a0793f405b4"
DATASETS = ["california", "adult", "higgs-small", "churn", "house"]
N_SEEDS = 15
OUT = Path(__file__).resolve().parents[1] / "configs" / "references" / "tabm_mlp.json"


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def summarize(dataset, commit):
    reports = [fetch(RAW.format(commit=commit, dataset=dataset, seed=s)) for s in range(N_SEEDS)]
    regression = reports[0]["prediction_type"] is None or "rmse" in reports[0]["metrics"]["test"]
    key = "rmse" if regression else "accuracy"
    scores = [r["metrics"]["test"][key] for r in reports]
    cfg = reports[0]["config"]
    backbone = cfg["model"]["backbone"]
    return {
        "metric": key,
        "mean": statistics.mean(scores),
        "std": statistics.stdev(scores),
        "n_seeds": len(scores),
        "n_parameters": reports[0]["n_parameters"],
        "n_blocks": backbone["n_blocks"],
        "d_block": backbone["d_block"],
        "dropout": backbone.get("dropout", 0.0),
        "lr": cfg["optimizer"]["lr"],
        "weight_decay": cfg["optimizer"]["weight_decay"],
        "batch_size": cfg["batch_size"],
        "patience": cfg["patience"],
        "num_policy": cfg["data"].get("num_policy"),
        "source": RAW.format(commit=commit, dataset=dataset, seed="<seed>"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commit", default=COMMIT)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    refs = {d: summarize(d, args.commit) for d in DATASETS}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(refs, indent=2) + "\n")
    for d, r in refs.items():
        print(f"{d:12s} {r['metric']}={r['mean']:.4f}±{r['std']:.4f}  MLP {r['n_blocks']}x{r['d_block']} drop={r['dropout']:.3f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

"""Score finished runs (JAX or PyTorch) on their dataset's test split. Each run is rebuilt
from its own .hydra/config.yaml. Writes <campaign>.csv (one row per run) and
<campaign>_summary.csv (mean/std over seeds) to outputs/reports/.

Usage:
    pixi run python src/optim_env/tools/test.py                       # every run in outputs/runs
    pixi run python src/optim_env/tools/test.py 'outputs/runs/california/jax_NE/*/*/seed_*' \\
        --campaign california_ne_hpo
"""

import argparse
import csv
import functools
import glob
import logging
from pathlib import Path

from omegaconf import OmegaConf

from optim_env.analysis.loading import load_dataset, load_run, predict
from optim_env.paths import REPORTS
from optim_env.resolvers import register_custom_resolvers

register_custom_resolvers()


@functools.lru_cache(maxsize=2)
def _dataset(dataset_yaml: str):
    """One load per dataset config: reloading MNIST for each of ~200 runs grew memory
    past 15 GB. main() sorts runs by dataset and seed so two entries suffice (a text
    task's token and GloVe configs share a dataset folder)."""
    return load_dataset(OmegaConf.create({"dataset": OmegaConf.create(dataset_yaml)}))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("test")

CSV_FIELDS = ["exp_name", "framework", "method", "algorithm", "variant", "trainer",
              "optimizer", "dataset", "seed", "n_test", "mse", "rmse", "accuracy"]


def evaluate_run(run_dir: Path) -> dict | None:
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        log.warning(f"Skipping {run_dir}: missing .hydra/config.yaml")
        return None

    cfg = OmegaConf.load(config_path)
    is_torch = "torch_env" in cfg.trainer._target_
    checkpoint_path = run_dir / ("best_model.pt" if is_torch else "best_model.msgpack")
    if not checkpoint_path.exists():
        log.warning(f"Skipping {run_dir}: missing {checkpoint_path.name}")
        return None

    # Broad on purpose: old run directories may point at code that has since moved.
    # Skip them instead of failing the whole report.
    try:
        dataset = _dataset(OmegaConf.to_yaml(cfg.dataset, resolve=True))
        test_ds = dataset.get_test_ds()

        X_test, y_test = test_ds.X, test_ds.y
        preds = predict(load_run(run_dir), X_test)

        is_classification = bool(cfg.dataset.num_classes and cfg.dataset.num_classes > 0)
        row = {
            "exp_name": cfg.exp_name,
            "framework": cfg.get("framework", ""),
            "method": cfg.get("method", ""),
            "algorithm": cfg.get("algorithm", cfg.optimizer._target_.rsplit(".", 1)[-1]),
            "variant": cfg.get("variant", ""),
            "trainer": cfg.trainer._target_.rsplit(".", 1)[-1],
            "optimizer": cfg.optimizer._target_,
            "dataset": cfg.dataset.name,
            "seed": cfg.seed,
            "n_test": X_test.shape[0],
        }
        if is_classification:
            row["accuracy"] = float((preds.argmax(axis=-1) == y_test).mean())
        else:
            if preds.ndim > y_test.ndim:
                preds = preds.reshape(y_test.shape)
            # Targets may be standardized for training; report error in the target's own
            # units (e.g. California's $100k) so numbers compare with published results.
            y_scale = float(getattr(dataset, "_y_std", [1.0])[0])
            test_mse = float(((preds - y_test) ** 2).mean()) * y_scale**2
            row["mse"] = test_mse
            row["rmse"] = test_mse**0.5
    except Exception as e:
        log.warning(f"Skipping {run_dir}: {type(e).__name__}: {e}")
        return None

    log.info(f"{row['exp_name']}: {row}")
    return row


def _write_summary(rows: list[dict], summary_path: Path):
    """Aggregate the per-run rows across seeds: one line per
    (dataset, framework, method, algorithm, variant) with mean/std/n_seeds of each
    metric, so a campaign run with SEEDS="0,1,2" reports variance rather than a single
    point estimate."""
    import pandas as pd

    df = pd.DataFrame(rows)
    group_cols = ["dataset", "framework", "method", "algorithm", "variant"]
    metric_cols = [c for c in ("rmse", "mse", "accuracy") if c in df.columns and df[c].notna().any()]

    summary = df.groupby(group_cols, dropna=False).agg(
        n_seeds=("seed", "nunique"),
        **{f"{m}_{stat}": (m, stat) for m in metric_cols for stat in ("mean", "std")},
    ).reset_index()
    summary.to_csv(summary_path, index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "run_dirs_glob", nargs="*", default=["outputs/runs/*/*/*/*/seed_*"],
        help="One or more glob patterns for run directories. Defaults to every run in the "
             "outputs/runs tree; narrow it e.g. 'outputs/runs/california/jax_NE/*/*/seed_*'.",
    )
    parser.add_argument("--campaign", default="test_results",
                        help="Report name; writes outputs/reports/<campaign>.csv plus a _summary.csv aggregated over seeds.")
    parser.add_argument("--output-dir", default=str(REPORTS))
    args = parser.parse_args()

    # outputs/runs/<dataset>/.../seed_<n>: grouped by dataset and seed for _dataset's cache.
    run_dirs = sorted({Path(p) for pattern in args.run_dirs_glob for p in glob.glob(pattern) if Path(p).is_dir()},
                      key=lambda d: (d.parts[-5:-4], d.name, d))
    if not run_dirs:
        raise SystemExit(f"No run directories matched: {args.run_dirs_glob}")

    rows = [row for run_dir in run_dirs if (row := evaluate_run(run_dir)) is not None]
    if not rows:
        raise SystemExit("No runs could be evaluated (missing configs/checkpoints/test splits).")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.campaign}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Wrote {len(rows)} per-run results to {csv_path}")

    summary_path = out_dir / f"{args.campaign}_summary.csv"
    summary = _write_summary(rows, summary_path)
    log.info(f"Wrote {len(summary)} aggregated rows to {summary_path}")


if __name__ == "__main__":
    main()

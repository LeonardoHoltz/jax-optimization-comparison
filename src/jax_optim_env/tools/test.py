"""Evaluate trained checkpoints on their dataset's held-out TEST split (not train/val),
reconstructing each run from the exact Hydra config it was trained with (saved at
outputs/<exp_name>/.hydra/config.yaml), and aggregate the results across multiple
runs/optimizers into a single CSV under outputs/tests/.

Usage:
    pixi run python src/jax_optim_env/tools/test.py "outputs/*california*"
    pixi run python src/jax_optim_env/tools/test.py "outputs/*california*" --csv-name california_test_results.csv
"""

import argparse
import csv
import glob
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
from flax.training import train_state
from lerna.utils import instantiate
from omegaconf import OmegaConf

from jax_optim_env.resolvers import register_custom_resolvers
from jax_optim_env.utils.checkpointing import load_checkpoint
from jax_optim_env.utils.metrics import accuracy, mse

register_custom_resolvers()

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("test")

CSV_FIELDS = ["exp_name", "trainer", "optimizer", "dataset", "seed", "n_test", "mse", "rmse", "accuracy"]


def _load_trained_params(cfg, model, dummy_x, checkpoint_path):
    """Reconstructs a template matching how each trainer actually saves its checkpoint,
    then deserializes into it. SGDTrainer saves the full TrainState (params + opt_state);
    ESTrainer saves just the model params directly (see their respective `fit()`)."""
    trainer_target = cfg.trainer._target_
    rng = jax.random.PRNGKey(0)  # only used to build a matching pytree shape/structure

    if trainer_target.endswith("SGDTrainer"):
        template = train_state.TrainState.create(
            apply_fn=model.apply,
            params=model.init(rng, dummy_x),
            tx=instantiate(cfg.optimizer),
        )
        state = load_checkpoint(template, checkpoint_path)
        return state.params
    elif trainer_target.endswith("ESTrainer"):
        template = model.init(rng, dummy_x)
        return load_checkpoint(template, checkpoint_path)
    else:
        raise ValueError(f"Unsupported trainer for evaluation: {trainer_target}")


def evaluate_run(run_dir: Path) -> dict | None:
    config_path = run_dir / ".hydra" / "config.yaml"
    checkpoint_path = run_dir / "best_model.msgpack"
    if not config_path.exists() or not checkpoint_path.exists():
        log.warning(f"Skipping {run_dir}: missing .hydra/config.yaml or best_model.msgpack")
        return None

    cfg = OmegaConf.load(config_path)

    try:
        dataset = instantiate(cfg.dataset)
        test_ds = dataset.get_test_ds()
    except NotImplementedError as e:
        log.warning(f"Skipping {run_dir}: {e}")
        return None

    X_test, y_test = jnp.asarray(test_ds.X), jnp.asarray(test_ds.y)

    model = instantiate(cfg.model)
    dummy_x = jnp.zeros((1, X_test.shape[1]))
    params = _load_trained_params(cfg, model, dummy_x, checkpoint_path)

    preds = model.apply(params, X_test, deterministic=True)

    is_classification = bool(cfg.dataset.num_classes and cfg.dataset.num_classes > 0)
    row = {
        "exp_name": cfg.exp_name,
        "trainer": cfg.trainer._target_.rsplit(".", 1)[-1],
        "optimizer": cfg.optimizer._target_,
        "dataset": cfg.dataset.name,
        "seed": cfg.seed,
        "n_test": X_test.shape[0],
    }
    if is_classification:
        row["accuracy"] = float(accuracy(preds, y_test))
    else:
        if preds.ndim > y_test.ndim:
            preds = preds.reshape(y_test.shape)
        test_mse = float(mse(preds, y_test))
        row["mse"] = test_mse
        row["rmse"] = test_mse**0.5

    log.info(f"{row['exp_name']}: {row}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs_glob", help="Glob pattern for experiment output directories, e.g. 'outputs/*california*'")
    parser.add_argument("--csv-name", default="test_results.csv")
    parser.add_argument("--output-dir", default="outputs/tests")
    args = parser.parse_args()

    run_dirs = sorted(Path(p) for p in glob.glob(args.run_dirs_glob) if Path(p).is_dir())
    if not run_dirs:
        raise SystemExit(f"No experiment directories matched: {args.run_dirs_glob}")

    rows = [row for run_dir in run_dirs if (row := evaluate_run(run_dir)) is not None]
    if not rows:
        raise SystemExit("No runs could be evaluated (missing configs/checkpoints/test splits).")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / args.csv_name

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"Wrote {len(rows)} results to {csv_path}")


if __name__ == "__main__":
    main()

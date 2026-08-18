"""Evaluate trained checkpoints (JAX or PyTorch) on their dataset's held-out TEST split
(not train/val), reconstructing each run from the exact Hydra config it was trained
with (saved at outputs/<exp_name>/.hydra/config.yaml), and aggregate the results across
multiple runs/optimizers/frameworks into a single CSV under outputs/tests/.

Usage:
    pixi run python src/jax_optim_env/tools/test.py "outputs/*california*"
    pixi run python src/jax_optim_env/tools/test.py "outputs/*california*" --csv-name california_test_results.csv
"""

import argparse
import csv
import glob
import logging
from pathlib import Path

from lerna.utils import instantiate
from omegaconf import OmegaConf

from jax_optim_env.resolvers import register_custom_resolvers

register_custom_resolvers()

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("test")

CSV_FIELDS = ["exp_name", "trainer", "optimizer", "dataset", "seed", "n_test", "mse", "rmse", "accuracy"]


def _predict_jax(cfg, model, X_test):
    import jax
    import jax.numpy as jnp
    from flax.training import train_state

    from jax_optim_env.utils.checkpointing import load_checkpoint

    checkpoint_path = Path(cfg.save_path) / "best_model.msgpack"
    rng = jax.random.PRNGKey(0)  # only used to build a matching pytree shape/structure
    dummy_x = jnp.zeros((1, X_test.shape[1]))

    # Reconstructs a template matching how each trainer actually saves its checkpoint,
    # then deserializes into it. SGDTrainer saves the full TrainState (params +
    # opt_state); ESTrainer saves just the model params directly (see their fit()).
    trainer_target = cfg.trainer._target_
    if trainer_target.endswith("SGDTrainer"):
        template = train_state.TrainState.create(
            apply_fn=model.apply,
            params=model.init(rng, dummy_x),
            tx=instantiate(cfg.optimizer),
        )
        params = load_checkpoint(template, checkpoint_path).params
    elif trainer_target.endswith("ESTrainer"):
        params = load_checkpoint(model.init(rng, dummy_x), checkpoint_path)
    else:
        raise ValueError(f"Unsupported trainer for evaluation: {trainer_target}")

    return model.apply(params, jnp.asarray(X_test), deterministic=True)


def _predict_torch(cfg, model, X_test):
    import torch

    from jax_optim_env.torch_env.utils.checkpointing import load_checkpoint

    checkpoint_path = Path(cfg.save_path) / "best_model.pt"
    model.load_state_dict(load_checkpoint(checkpoint_path))
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X_test)).numpy()


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

    try:
        dataset = instantiate(cfg.dataset)
        test_ds = dataset.get_test_ds()
    except NotImplementedError as e:
        log.warning(f"Skipping {run_dir}: {e}")
        return None

    X_test, y_test = test_ds.X, test_ds.y
    model = instantiate(cfg.model)
    preds = _predict_torch(cfg, model, X_test) if is_torch else _predict_jax(cfg, model, X_test)

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
        row["accuracy"] = float((preds.argmax(axis=-1) == y_test).mean())
    else:
        if preds.ndim > y_test.ndim:
            preds = preds.reshape(y_test.shape)
        test_mse = float(((preds - y_test) ** 2).mean())
        row["mse"] = test_mse
        row["rmse"] = test_mse**0.5

    log.info(f"{row['exp_name']}: {row}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "run_dirs_glob", nargs="+",
        help="One or more glob patterns for experiment output directories, e.g. 'outputs/*california*' 'outputs/*adult*'",
    )
    parser.add_argument("--csv-name", default="test_results.csv")
    parser.add_argument("--output-dir", default="outputs/tests")
    args = parser.parse_args()

    run_dirs = sorted({Path(p) for pattern in args.run_dirs_glob for p in glob.glob(pattern) if Path(p).is_dir()})
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

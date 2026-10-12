"""Hyperparameter search with Optuna (TPE), for gradient (SGD) and neuroevolution (NE) runs.

Each trial runs tools/train.py as its own subprocess, so JAX state never leaks between
trials. Trials stream their validation loss back, which lets the median pruner stop
trials that fall behind. After the search, the best settings are retrained at the full
budget once per --final-seeds seed, as normal runs under outputs/runs/.

The study lives in <study dir>/study.db: re-running the same command resumes it, and
final runs that already finished are skipped. Delete the directory to start over.

Search space, picked by the experiment config's `method`:
    NE:  optimizer.population_size, init_std_multiplier, and es_learning_rate for
         algorithms with an internal optimizer (Open_ES, PGPE, SNES, ARS)
    SGD: learning_rate, weight_decay, and model.dropout when the model has dropout

Usage:
    pixi run python src/optim_env/tools/hpo.py --optimizer jax/sep_cma_es
    pixi run python src/optim_env/tools/hpo.py --config-name study_sgd --optimizer jax/adamw_clip \\
        --dataset tabm_adult_quantile --model jax/mlp_tabm_adult --hpo-budget 100 --final-budget 1000
"""

import argparse
import inspect
import json
import logging
import re
import subprocess
import sys
from collections import deque
from pathlib import Path

import optuna
from lerna.utils import get_class
from omegaconf import OmegaConf

from optim_env.paths import hpo_dir, run_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("hpo")
optuna.logging.set_verbosity(optuna.logging.WARNING)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
class TrialCrashed(Exception):
    """A trial whose run exited with an error. Optuna marks it FAIL (not pruned), so a
    resumed study runs a replacement instead of counting it as done."""


# One line per evaluation, from either trainer.
VAL_LINE = re.compile(r"(?:Generation|Epoch) (\d+) \| val loss=([\d.eE+-]+)")


def _optimizer_target(optimizer_arg: str) -> str:
    return OmegaConf.load(CONFIGS / "optimizer" / f"{optimizer_arg}.yaml")._target_


def _algorithm_label(optimizer_arg: str) -> str:
    """Same label run_layout.yaml derives from _target_, e.g. "jax/particle_swarm" -> "PSO"."""
    return _optimizer_target(optimizer_arg).rsplit(".", 1)[-1]


def _accepts_internal_optimizer(optimizer_arg: str) -> bool:
    strategy_cls = get_class(_optimizer_target(optimizer_arg))
    return "optimizer" in inspect.signature(strategy_cls.__init__).parameters


def search_space(args, method: str):
    """Returns sample(trial) -> {override key: value}."""
    if method == "NE":
        tunes_lr = _accepts_internal_optimizer(args.optimizer)

        def sample(trial):
            params = {
                "optimizer.population_size": trial.suggest_int("population_size", 16, args.population_size_max, step=2),
                "init_std_multiplier": trial.suggest_float("init_std_multiplier", 0.1, 3.0, log=True),
            }
            if tunes_lr:
                params["es_learning_rate"] = trial.suggest_float("es_learning_rate", 1e-4, 1e-1, log=True)
            return params
        return sample

    has_dropout = "dropout" in OmegaConf.load(CONFIGS / "model" / f"{args.model}.yaml")

    def sample(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", args.lr_min, args.lr_max, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        }
        if has_dropout:
            params["model.dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
        return params
    return sample


def _run(args, overrides: list[str], out_dir: Path, trial: optuna.Trial = None, seed: int = None) -> dict | None:
    cmd = [
        sys.executable, "src/optim_env/tools/train.py",
        "--config-name", args.config_name,
        f"dataset={args.dataset}",
        f"model={args.model}",
        f"optimizer={args.optimizer}",
        f"seed={args.seed if seed is None else seed}",
        f"save_path={out_dir}",
        f"hydra.run.dir={out_dir}",
        *overrides,
        *args.set,  # last, so a --set wins over anything above
    ]
    # stderr folded into stdout, so one reader can't deadlock on a full pipe.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = deque(maxlen=40)
    best_so_far = float("inf")
    pruned = False
    for line in proc.stdout:
        tail.append(line)
        m = VAL_LINE.search(line) if trial is not None else None
        if m is None:
            continue
        # Report the running best: the same quantity the objective returns.
        best_so_far = min(best_so_far, float(m.group(2)))
        trial.report(best_so_far, int(m.group(1)))
        if trial.should_prune():
            pruned = True
            proc.kill()
            break
    proc.wait()
    if pruned:
        log.info(f"Trial {trial.number} pruned at best_val_loss={best_so_far:.4f}")
        raise optuna.TrialPruned()

    metrics_path = out_dir / "metrics.json"
    if proc.returncode != 0 or not metrics_path.exists():
        log.warning(f"Run at {out_dir} failed (exit {proc.returncode}):\n{''.join(tail)}")
        return None
    return json.loads(metrics_path.read_text())


def _overrides(params: dict) -> list[str]:
    return [f"{k}={v}" for k, v in params.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="california_mlp_es_exp")
    parser.add_argument("--dataset", default="tabm_california")
    parser.add_argument("--model", default="jax/mlp_california")
    parser.add_argument("--optimizer", required=True, help="e.g. jax/sep_cma_es or jax/adamw_clip")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--population-size-max", type=int, default=96,
                        help="NE only. Fitness runs the whole population at once, so lower this if trials run out of GPU memory.")
    parser.add_argument("--lr-min", type=float, default=3e-5, help="SGD only")
    parser.add_argument("--lr-max", type=float, default=1e-3, help="SGD only")
    parser.add_argument("--hpo-budget", type=int, default=100,
                        help="Generations (NE) or max epochs (SGD) per trial. Keep it near --final-budget: "
                             "short trials favour optimizers that plateau early.")
    parser.add_argument("--final-budget", type=int, default=300, help="Generations or max epochs for the final runs")
    parser.add_argument("--seed", type=int, default=0, help="Seed of every search trial")
    parser.add_argument("--final-seeds", default=None, help="Comma-separated seeds for the final runs. Defaults to --seed.")
    parser.add_argument("--variant", default="hpo",
                        help="Label of the final runs and the study directory. Use a throwaway name "
                             "(e.g. smoketest) when testing, or real results get overwritten.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Hydra override for every trial and final run, repeatable "
                             "(settings that must match the untuned runs, e.g. fitness_batch_size=256)")
    args = parser.parse_args()

    exp_cfg = OmegaConf.load(CONFIGS / f"{args.config_name}.yaml")
    method = exp_cfg.method
    budget_key = "generations" if method == "NE" else "epochs"
    algorithm = _algorithm_label(args.optimizer)
    dataset_name = OmegaConf.load(CONFIGS / "dataset" / f"{args.dataset}.yaml").name
    study_dir = hpo_dir(dataset_name, exp_cfg.framework, method, algorithm)
    if args.variant != "hpo":
        study_dir = study_dir.parent / f"{study_dir.name}_{args.variant}"
    study_dir.mkdir(parents=True, exist_ok=True)

    sample = search_space(args, method)

    def objective(trial):
        params = sample(trial)
        overrides = [f"{budget_key}={args.hpo_budget}", f"variant={args.variant}_trial{trial.number}",
                     "save_checkpoint=false", *_overrides(params)]
        metrics = _run(args, overrides, study_dir / f"trial_{trial.number}", trial=trial)
        if metrics is None:
            raise TrialCrashed(f"trial {trial.number} run failed")
        log.info(f"Trial {trial.number}: {trial.params} -> best_val_loss={metrics['best_val_loss']:.4f}")
        return metrics["best_val_loss"]

    log.info(f"HPO for {args.optimizer} ({method}) on {args.dataset}/{args.model}, {args.n_trials} trials")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=args.hpo_budget // 4)
    study = optuna.create_study(
        direction="minimize", study_name=algorithm, pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        storage=f"sqlite:///{study_dir / 'study.db'}", load_if_exists=True,
    )
    finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    n_done = len(study.get_trials(deepcopy=False, states=finished))
    if n_done:
        log.info(f"Resuming study: {n_done} trials already finished")
    study.optimize(objective, n_trials=max(args.n_trials - n_done, 0), catch=(TrialCrashed,))

    study.trials_dataframe().to_csv(study_dir / "trials.csv", index=False)
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        log.error(f"All {len(study.trials)} trials failed; see their output above.")
        sys.exit(1)

    # Map Optuna's param names back to override keys by re-sampling the best trial.
    best = sample(optuna.trial.FixedTrial(study.best_params))
    (study_dir / "best_params.json").write_text(
        json.dumps({"best_value": study.best_value, "best_params": study.best_params, "overrides": best}, indent=2))
    log.info(f"Best trial: value={study.best_value:.4f} params={study.best_params}")

    final_overrides = [f"{budget_key}={args.final_budget}", f"variant={args.variant}", *_overrides(best)]
    final_seeds = [int(x) for x in args.final_seeds.split(",")] if args.final_seeds else [args.seed]
    failed = 0
    for seed in final_seeds:
        final_dir = run_dir(dataset_name, exp_cfg.framework, method, algorithm, args.variant, seed)
        if (final_dir / "metrics.json").exists():
            log.info(f"Final run seed {seed} already done, skipping ({final_dir})")
            continue
        log.info(f"Final run seed {seed} -> {final_dir}")
        metrics = _run(args, final_overrides, final_dir, seed=seed)
        if metrics is None:
            failed += 1
        else:
            log.info(f"Final run seed {seed}: best_val_loss={metrics['best_val_loss']:.4f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

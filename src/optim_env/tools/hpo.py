"""Hyperparameter optimization for JAX/evosax neuroevolution optimizers, via Optuna.

Each trial launches tools/train.py as an independent subprocess with sampled overrides
(so JAX/JIT state and RNGs never leak between trials), reads back that run's
outputs/hpo/<optimizer>/trial_<n>/metrics.json (written by ESTrainer.fit()), and reports
its best_val_loss to Optuna. After the search, retrains once at full generation budget
using the best hyperparameters found, once per --final-seeds seed, and saves those as
normal experiment outputs (variant "hpo") so they can be evaluated/compared like any other
run via tools/test.py.

The study is stored in <study dir>/study.db, so an interrupted search resumes where it
stopped when re-run with the same arguments. Delete study.db to start a fresh search.

Search space (see ESTrainer.prepare_model for what each knob actually does):
    optimizer.population_size  -- every evosax algorithm
    init_std_multiplier        -- every evosax algorithm
    es_learning_rate           -- only algorithms with an internal optimizer
                                   (Open_ES, PGPE, SNES, ARS); skipped otherwise

Usage:
    pixi run python src/optim_env/tools/hpo.py --optimizer jax/sep_cma_es
    pixi run python src/optim_env/tools/hpo.py --optimizer jax/open_es --n-trials 30 --hpo-generations 150
"""

import argparse
import inspect
import re
from collections import deque
import json
import logging
import subprocess
import sys
from pathlib import Path

import optuna
from lerna.utils import get_class
from omegaconf import OmegaConf

from optim_env.paths import hpo_dir, run_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("hpo")
optuna.logging.set_verbosity(optuna.logging.WARNING)


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _optimizer_target(optimizer_arg: str) -> str:
    """The evosax class path a `--optimizer jax/sep_cma_es` argument selects."""
    return OmegaConf.load(CONFIGS / "optimizer" / f"{optimizer_arg}.yaml")._target_


def _algorithm_label(optimizer_arg: str) -> str:
    """Same label configs/base/run_layout.yaml derives via
    ${basename:${optimizer._target_}}, e.g. "jax/particle_swarm" -> "PSO", so HPO
    artifacts and training runs agree on one name per algorithm."""
    return _optimizer_target(optimizer_arg).rsplit(".", 1)[-1]


def _accepts_internal_optimizer(optimizer_arg: str) -> bool:
    """Whether this evosax algorithm has an internal optax optimizer (see
    ESTrainer.prepare_model's `if "optimizer" in strategy_init_params` branch) -- if not,
    es_learning_rate would have no effect, so it's left out of the search space."""
    strategy_cls = get_class(_optimizer_target(optimizer_arg))
    return "optimizer" in inspect.signature(strategy_cls.__init__).parameters


# Trainers log one of these per evaluation; streaming them back is what lets Optuna
# prune a hopeless trial instead of paying for its full generation budget.
VAL_LINE = re.compile(r"(?:Generation|Epoch) (\d+) \| val loss=([\d.eE+-]+)")


def _run_trial(args, overrides: list[str], run_dir: Path, trial: optuna.Trial = None, seed: int = None) -> dict:
    cmd = [
        sys.executable, "src/optim_env/tools/train.py",
        "--config-name", args.config_name,
        f"dataset={args.dataset}",
        f"model={args.model}",
        f"optimizer={args.optimizer}",
        f"seed={args.seed if seed is None else seed}",
        f"save_path={run_dir}",
        f"hydra.run.dir={run_dir}",
        *overrides,
        # Applied last so a --set can override anything above it. These are the settings
        # that must match the baseline runs for the comparison to mean anything (e.g.
        # fitness_batch_size on image tasks), rather than hyperparameters being searched.
        *args.set,
    ]
    # stderr is folded into stdout so a single reader cannot deadlock on a full pipe.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    tail = deque(maxlen=40)
    best_so_far = float("inf")
    pruned = False
    for line in proc.stdout:
        tail.append(line)
        if trial is None:
            continue
        m = VAL_LINE.search(line)
        if m is None:
            continue
        # Report the running best, matching what the objective ultimately returns, so the
        # pruner compares trials on the same quantity it is optimising.
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

    metrics_path = run_dir / "metrics.json"
    if proc.returncode != 0 or not metrics_path.exists():
        log.warning(f"Trial run at {run_dir} failed (exit {proc.returncode}):\n{''.join(tail)}")
        return None
    with open(metrics_path) as f:
        return json.load(f)


def make_objective(args, tunes_lr: bool, study_dir: Path):
    def objective(trial: optuna.Trial) -> float:
        population_size = trial.suggest_int("population_size", 16, args.population_size_max, step=2)
        init_std_multiplier = trial.suggest_float("init_std_multiplier", 0.1, 3.0, log=True)
        overrides = [
            f"generations={args.hpo_generations}",
            f"variant={args.variant}_trial{trial.number}",
            f"optimizer.population_size={population_size}",
            f"init_std_multiplier={init_std_multiplier}",
        ]
        if tunes_lr:
            es_learning_rate = trial.suggest_float("es_learning_rate", 1e-4, 1e-1, log=True)
            overrides.append(f"es_learning_rate={es_learning_rate}")

        run_dir = study_dir / f"trial_{trial.number}"
        metrics = _run_trial(args, overrides, run_dir, trial=trial)
        if metrics is None:
            raise optuna.TrialPruned()

        log.info(f"Trial {trial.number}: {trial.params} -> best_val_loss={metrics['best_val_loss']:.4f}")
        return metrics["best_val_loss"]

    return objective


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="california_mlp_es_exp")
    parser.add_argument("--dataset", default="tabm_california")
    parser.add_argument("--model", default="jax/mlp_california")
    parser.add_argument("--optimizer", required=True, help="e.g. jax/sep_cma_es")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--population-size-max", type=int, default=96,
                         help="Upper bound for the population_size search dimension. Fitness is "
                              "vmapped over the whole population, so lower it if trials run out of GPU "
                              "memory (the MNIST campaigns use 64).")
    parser.add_argument("--hpo-generations", type=int, default=100, help="Generation budget per trial. Keep it close to --final-generations: "
                             "scoring trials at a much shorter budget favours optimizers that "
                             "plateau early over ones that keep improving. Pruning (see "
                             "_run_trial) is what makes a long budget affordable.")
    parser.add_argument("--final-generations", type=int, default=300, help="Generation budget for the final, best-hyperparameter confirmation run")
    parser.add_argument("--seed", type=int, default=0, help="Seed every search trial runs with")
    parser.add_argument("--final-seeds", default=None,
                        help="Comma-separated seeds for the final confirmation runs, e.g. 0,1,2, so "
                             "tuned results get the same seed spread as the baselines. Defaults to --seed.")
    parser.add_argument("--variant", default="hpo",
                        help="Variant label for the final confirmation runs and the study "
                             "directory. Re-running with the same variant resumes that study and "
                             "overwrites its final runs, so pass a throwaway name (e.g. --variant "
                             "smoketest) when testing the machinery.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Extra Hydra override passed through to every trial and to the "
                             "final run, repeatable (e.g. --set fitness_batch_size=256)")
    args = parser.parse_args()

    algorithm = _algorithm_label(args.optimizer)
    dataset_name = OmegaConf.load(CONFIGS / "dataset" / f"{args.dataset}.yaml").name
    exp_cfg = OmegaConf.load(CONFIGS / f"{args.config_name}.yaml")
    study_dir = hpo_dir(dataset_name, exp_cfg.framework, exp_cfg.method, algorithm)
    if args.variant != "hpo":
        study_dir = study_dir.parent / f"{study_dir.name}_{args.variant}"
    study_dir.mkdir(parents=True, exist_ok=True)

    tunes_lr = _accepts_internal_optimizer(args.optimizer)
    log.info(f"Starting HPO for optimizer={args.optimizer} ({args.n_trials} trials, es_learning_rate tuned={tunes_lr})")

    # Pruning only became possible once trials stream their intermediate validation loss
    # (see _run_trial). It is what makes a long per-trial budget affordable: a trial that is
    # clearly behind the others is killed rather than run to completion. The long budget in
    # turn is what keeps the search fair to optimizers that improve slowly but do not
    # plateau -- scoring at a short budget systematically favours early plateauers.
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=args.hpo_generations // 4,
    )
    # Persisted to disk after every trial, so a crash or Ctrl-C hours in loses at most the
    # trial that was running. Re-running continues up to --n-trials finished trials.
    study = optuna.create_study(
        direction="minimize", study_name=algorithm, pruner=pruner,
        storage=f"sqlite:///{study_dir / 'study.db'}", load_if_exists=True,
    )
    finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    n_done = len(study.get_trials(deepcopy=False, states=finished))
    if n_done:
        log.info(f"Resuming study: {n_done} trials already finished")
    study.optimize(make_objective(args, tunes_lr, study_dir), n_trials=max(args.n_trials - n_done, 0))

    study.trials_dataframe().to_csv(study_dir / "trials.csv", index=False)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        log.error(
            f"All {len(study.trials)} trials failed (see trial subprocess stderr above) -- "
            f"nothing to report. Try a smaller --population-size-max or --hpo-generations."
        )
        return

    with open(study_dir / "best_params.json", "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)
    log.info(f"Best trial: value={study.best_value:.4f} params={study.best_params}")

    # Final confirmation runs at full generation budget with the best hyperparameters found,
    # one per seed so the tuned rows of a report carry a seed spread like the baselines do.
    # They go in the normal runs/ tree (not under outputs/hpo/, which holds search trials)
    # as variant="hpo", right next to the "baseline" runs of the same algorithm, and
    # tools/test.py picks both up for comparison.
    final_overrides = [
        f"generations={args.final_generations}",
        f"variant={args.variant}",
        f"optimizer.population_size={study.best_params['population_size']}",
        f"init_std_multiplier={study.best_params['init_std_multiplier']}",
    ]
    if tunes_lr:
        final_overrides.append(f"es_learning_rate={study.best_params['es_learning_rate']}")

    final_seeds = [int(x) for x in args.final_seeds.split(",")] if args.final_seeds else [args.seed]
    for seed in final_seeds:
        final_dir = run_dir(dataset_name, exp_cfg.framework, exp_cfg.method, algorithm, args.variant, seed)
        log.info(f"Running final confirmation (seed {seed}) with best hyperparameters -> {final_dir}")
        final_metrics = _run_trial(args, final_overrides, final_dir, seed=seed)
        if final_metrics is not None:
            log.info(f"Final HPO-tuned run (seed {seed}): best_val_loss={final_metrics['best_val_loss']:.4f}")
        else:
            log.warning(f"Final confirmation run (seed {seed}) failed; see its output above.")

if __name__ == "__main__":
    main()

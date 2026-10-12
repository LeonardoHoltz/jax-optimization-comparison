"""Reproduction study driver: see src/optim_env/configs/study/reproduction.yaml.

    pixi run python scripts/campaigns/reproduction_study.py --pilot              # tiny budgets, all tasks
    pixi run python scripts/campaigns/reproduction_study.py --stage sgd          # gradient arms, all tasks
    pixi run python scripts/campaigns/reproduction_study.py --stage ne --tasks california adult

Safe to stop and re-run: finished runs (metrics.json present) are skipped, and HPO
studies resume from their study.db. Per-run output goes to
outputs/study/<study>/logs/; durations to outputs/study/<study>/timings.csv. After each
task it evaluates the test split and rebuilds outputs/reports/<study>/<study>.xlsx.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "src" / "optim_env" / "configs"
PY = sys.executable

# Children get the GPU with fixed kernel choices, so a rerun repeats its numbers bit for
# bit: autotuning picks kernels by timing, and Triton matmuls fail without autotuning.
# (--xla_gpu_deterministic_ops crashes LeNet's convolutions on this JAX version.)
DETERMINISTIC = "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
CHILD_ENV = {**os.environ, "HYDRA_FULL_ERROR": "1",
             "XLA_FLAGS": (os.environ.get("XLA_FLAGS", "") + " " + DETERMINISTIC).strip()}
# Embedding gradients are summed by GPU atomics in varying order; this flag fixes the
# order. It is ~4x slower and crashes LeNet's convolutions, so only gradient runs of
# embedding models get it (ES never computes those gradients).
EMBEDDING_MODELS = {"jax/fasttext", "jax/kim_cnn"}
SCATTER_ENV = {**CHILD_ENV, "XLA_FLAGS": CHILD_ENV["XLA_FLAGS"] + " --xla_gpu_deterministic_ops=true"}
os.environ["JAX_PLATFORMS"] = "cpu"


def label(optimizer: str) -> str:
    return OmegaConf.load(CONFIGS / "optimizer" / f"{optimizer}.yaml")._target_.rsplit(".", 1)[-1]


def dataset_name(dataset: str) -> str:
    return OmegaConf.load(CONFIGS / "dataset" / f"{dataset}.yaml").name


def n_params(dataset: str, model: str) -> int:
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    from lerna.utils import instantiate

    from optim_env.resolvers import register_custom_resolvers
    register_custom_resolvers()
    cfg = OmegaConf.create({"dataset": OmegaConf.load(CONFIGS / "dataset" / f"{dataset}.yaml"),
                            "model": OmegaConf.load(CONFIGS / "model" / f"{model}.yaml")})
    m = instantiate(cfg.model)
    params = m.init(jax.random.PRNGKey(0), jnp.zeros((1, cfg.dataset.num_features)))
    return int(ravel_pytree(params)[0].size)


class Study:
    def __init__(self, args):
        self.cfg = OmegaConf.load(CONFIGS / "study" / f"{args.study}.yaml")
        self.name = args.study + ("_pilot" if args.pilot else "")
        self.pilot = args.pilot
        self.prefix = "pilot-" if args.pilot else ""
        self.seeds = [0] if args.pilot else list(self.cfg.seeds)
        self.n_trials = 2 if args.pilot else self.cfg.n_trials
        self.dir = ROOT / "outputs" / "study" / self.name
        self.logs = self.dir / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.timings = self.dir / "timings.csv"
        self.reports = ROOT / "outputs" / "reports" / self.name

    def budget(self, value, pilot_value):
        return pilot_value if self.pilot else value

    def call(self, cmd: list[str], log_name: str, what: str, env=None) -> bool:
        log_path = self.logs / f"{log_name}.log"
        start = time.time()
        with open(log_path, "w") as log:
            code = subprocess.call(cmd, cwd=ROOT, env=env or CHILD_ENV, stdout=log, stderr=subprocess.STDOUT)
        seconds = time.time() - start
        new = not self.timings.exists()
        with open(self.timings, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["what", "seconds", "exit_code", "log"])
            w.writerow([what, round(seconds, 1), code, log_path.relative_to(ROOT)])
        status = "ok" if code == 0 else f"FAILED (exit {code}, see {log_path.relative_to(ROOT)})"
        print(f"[{time.strftime('%H:%M:%S')}] {what}: {status} in {seconds / 60:.1f} min", flush=True)
        return code == 0

    # ---- one arm --------------------------------------------------------------------
    def run_arm(self, task, t, arch, method, optimizer):
        spec = t[arch]
        regime = t.sgd if method == "SGD" else t.es
        config = "study_sgd" if method == "SGD" else "study_es"
        algo = label(optimizer)
        common = [f"dataset={spec.dataset}", f"model={spec.model}", f"optimizer={optimizer}", f"loss={t.loss}"]
        if method == "SGD":
            budget_key, budget = "epochs", self.budget(regime.epochs, 2)
            hpo_budget = budget
            fixed = [f"patience={regime.patience}"]
            untuned = [f"learning_rate={regime.learning_rate}", f"weight_decay={regime.weight_decay}"]
            hpo_extra = ["--lr-min", str(regime.lr_min), "--lr-max", str(regime.lr_max)]
        else:
            budget_key, budget = "generations", self.budget(regime.generations, 20)
            hpo_budget = self.budget(regime.hpo_generations, 20)
            fixed = [f"eval_every_gen={self.budget(regime.eval_every_gen, 5)}",
                     f"fitness_batch_size={regime.fitness_batch_size if regime.fitness_batch_size is not None else 'null'}"]
            untuned = [f"optimizer.population_size={regime.population_size}"] if "population_size" in regime else []
            hpo_extra = ["--population-size-max", str(regime.population_size_max)]

        tag = f"{task}.{arch}.{method}.{algo}"
        env = SCATTER_ENV if method == "SGD" and spec.model in EMBEDDING_MODELS else CHILD_ENV
        base_variant = f"{self.prefix}{arch}_baseline"
        for seed in self.seeds:
            out = ROOT / "outputs" / "runs" / dataset_name(spec.dataset) / f"jax_{method}" / algo / base_variant / f"seed_{seed}"
            if (out / "metrics.json").exists():
                continue
            self.call([PY, "src/optim_env/tools/train.py", "--config-name", config, *common, *fixed, *untuned,
                       f"{budget_key}={budget}", f"variant={base_variant}", f"seed={seed}"],
                      f"{tag}.baseline.seed{seed}", f"{tag} untuned seed {seed}", env)

        sets = [x for kv in [f"loss={t.loss}", *fixed] for x in ("--set", kv)]
        self.call([PY, "src/optim_env/tools/hpo.py", "--config-name", config, "--dataset", spec.dataset,
                   "--model", spec.model, "--optimizer", optimizer, "--n-trials", str(self.n_trials),
                   "--hpo-budget", str(hpo_budget), "--final-budget", str(budget),
                   "--seed", str(self.cfg.hpo_seed), "--final-seeds", ",".join(map(str, self.seeds)),
                   "--variant", f"{self.prefix}{arch}_hpo", *hpo_extra, *sets],
                  f"{tag}.hpo", f"{tag} HPO ({self.n_trials} trials) + {len(self.seeds)} seeds", env)

    def run_task(self, task, stages):
        t = self.cfg.tasks[task]
        for arch in ("paper", "simple"):
            if "sgd" in stages:
                self.run_arm(task, t, arch, "SGD", self.cfg.gradient_optimizer)
            if "ne" in stages:
                size = n_params(t[arch].dataset, t[arch].model)
                for opt in self.cfg.es_optimizers:
                    if opt.endswith("/cma_es") and size > self.cfg.cma_es_max_params:
                        self.note_skip(task, arch, opt, f"{size} parameters > {self.cfg.cma_es_max_params}: "
                                                         "full CMA-ES needs a D x D covariance matrix")
                        continue
                    skip = self.cfg.get("skip", {})
                    reason = skip.get(f"{task}/{arch}/{opt}") or skip.get(f"{task}/{arch}/*")
                    if reason:
                        self.note_skip(task, arch, opt, reason)
                        continue
                    self.run_arm(task, t, arch, "NE", opt)
        self.evaluate(task)

    def note_skip(self, task, arch, opt, reason):
        path = self.dir / "skipped.json"
        skipped = json.loads(path.read_text()) if path.exists() else {}
        skipped[f"{task}/{arch}/{label(opt)}"] = reason
        path.write_text(json.dumps(skipped, indent=2))

    def evaluate(self, task):
        t = self.cfg.tasks[task]
        names = sorted({dataset_name(t[a].dataset) for a in ("paper", "simple")})
        globs = [f"outputs/runs/{n}/jax_*/*/{self.prefix}{a}_*/seed_*" for n in names for a in ("paper", "simple")]
        self.reports.mkdir(parents=True, exist_ok=True)
        self.call([PY, "src/optim_env/tools/test.py", *globs, "--campaign", task, "--output-dir", str(self.reports)],
                  f"{task}.test", f"{task} test-set evaluation")
        self.call([PY, "src/optim_env/tools/study_report.py", "--study", self.name, "--config", str(CONFIGS / "study" / "reproduction.yaml")],
                  "report", "XLSX report")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", default="reproduction")
    parser.add_argument("--tasks", nargs="*", default=None, help="Default: every task in the study file")
    parser.add_argument("--stage", choices=["sgd", "ne", "all"], default="all")
    parser.add_argument("--pilot", action="store_true", help="1 seed, 2 trials, tiny budgets; separate outputs")
    parser.add_argument("--optimizers", nargs="*", default=None, help="Only these ES optimizers, e.g. jax/pgpe")
    args = parser.parse_args()

    study = Study(args)
    if args.optimizers:
        study.cfg.es_optimizers = args.optimizers
    tasks = args.tasks or list(study.cfg.tasks)
    stages = {"sgd", "ne"} if args.stage == "all" else {args.stage}
    subprocess.call(["bash", "-c", f"source scripts/lib/common.sh && write_manifest {study.name} "
                     f"stage={args.stage} tasks={','.join(tasks)} pilot={args.pilot}"], cwd=ROOT)
    for task in tasks:
        print(f"=== {task} ({', '.join(sorted(stages))})", flush=True)
        study.run_task(task, stages)


if __name__ == "__main__":
    main()

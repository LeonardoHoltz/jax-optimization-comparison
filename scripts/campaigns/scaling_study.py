"""Scaling study driver: see src/optim_env/configs/study/scaling_mnist.yaml.

    pixi run python scripts/campaigns/scaling_study.py plan --probe PROBE.csv  # run list + GPU hours
    pixi run python scripts/campaigns/scaling_study.py tune     # phase 1: noise x learning rate
    pixi run python scripts/campaigns/scaling_study.py sweep    # phase 2: width x population x rate
    pixi run python scripts/campaigns/scaling_study.py seeds    # phase 3: best rate, more seeds
    pixi run python scripts/campaigns/scaling_study.py adamw    # gradient reference
    ... --algorithms dust --widths 32 64    # a slice, e.g. one per GPU

Safe to stop and re-run: finished runs (metrics.json present) are skipped. Each phase reads
the previous phase's picks from outputs/study/<study>/tuned/. Runs go to
outputs/runs/mnist/jax_<SGD|NE>/<ALGO>/<variant>/seed_<n>/, variant scaling[tune]_w.._p.._n.._lr..
"""

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from reproduction_study import CHILD_ENV, CONFIGS, PY, ROOT, dataset_name, label  # noqa: E402

ENV = {**CHILD_ENV, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


def macs(w: int) -> int:
    """Multiply-adds per image of CNN [w, 2w, 4w] on 28x28 (XLA's own count misses convs)."""
    return 7056 * w * w + 7416 * w


class Scaling:
    def __init__(self, args):
        self.cfg = OmegaConf.load(CONFIGS / "study" / f"{args.study}.yaml")
        if args.pilot:  # checks the whole pipeline in minutes; separate run names and folders
            self.cfg.merge_with({"prefix": "pilot" + self.cfg.prefix, "widths": [1, 4], "populations": [8, 128],
                                 "max_generations": 300, "min_generations": 100, "extra_seeds": [1],
                                 "tune": {"population": 32, "generations": 200}})
            for a in self.cfg.algorithms.values():
                a.noise, a.learning_rates = a.noise[1:3], a.learning_rates[1:3]
        self.dir = ROOT / "outputs" / "study" / (args.study + ("_pilot" if args.pilot else ""))
        (self.dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.dir / "tuned").mkdir(exist_ok=True)
        self.algorithms = args.algorithms or list(self.cfg.algorithms)
        self.widths = args.widths or list(self.cfg.widths)
        self.dataset = dataset_name(self.cfg.dataset)
        self.dry = args.command == "plan"
        self.planned = []  # (phase, algorithm, width, population, generations), for plan

    # ---- run bookkeeping -----------------------------------------------------------------
    def generations(self, P, w):
        g = self.cfg.compute_budget_flops / (P * self.cfg.fitness_batch_size * 2 * macs(w))
        return int(min(self.cfg.max_generations, max(self.cfg.min_generations, g)))

    def chunk(self, P, w):
        return min(P, max(8, self.cfg.chunk_images_per_width // w))

    def run_dir(self, method, algo_label, variant, seed):
        return ROOT / "outputs" / "runs" / self.dataset / f"jax_{method}" / algo_label / variant / f"seed_{seed}"

    def best_val(self, run):
        path = run / "metrics.json"
        return json.loads(path.read_text())["best_val_loss"] if path.exists() else None

    def call(self, cmd, run, what, phase, algo, w, P, G):
        if self.dry:
            self.planned.append((phase, algo, w, P, G))
            return
        if (run / "metrics.json").exists():
            return
        log = self.dir / "logs" / f"{run.parent.name}.seed{run.name[5:]}.log"
        start = time.time()
        with open(log, "w") as f:
            code = subprocess.call(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
        seconds = time.time() - start
        timings = self.dir / "timings.csv"
        new = not timings.exists()
        with open(timings, "a", newline="") as f:
            if new:
                f.write("what,seconds,exit_code,log\n")
            csv.writer(f).writerow([what, round(seconds, 1), code, log.relative_to(ROOT)])
        status = "ok" if code == 0 else f"FAILED (exit {code}, see {log.relative_to(ROOT)})"
        print(f"[{time.strftime('%H:%M:%S')}] {what}: {status} in {seconds / 60:.1f} min", flush=True)

    # ---- one ES run ----------------------------------------------------------------------
    def es_run(self, phase, algo, w, P, noise, lr, seed, G):
        a = self.cfg.algorithms[algo]
        algo_label = label(a.optimizer)
        prefix = self.cfg.prefix + ("tune" if phase == "tune" else "")
        variant = f"{prefix}_w{w}_p{P}_n{noise:g}_lr{lr:g}"
        run = self.run_dir("NE", algo_label, variant, seed)
        cmd = [PY, "src/optim_env/tools/train.py", "--config-name", "study_es", f"dataset={self.cfg.dataset}",
               "model=jax/cnn_mnist", f"model.features=[{w},{2 * w},{4 * w}]", f"optimizer={a.optimizer}",
               f"loss={self.cfg.loss}", f"optimizer.population_size={P}", f"+population_chunk={self.chunk(P, w)}",
               f"fitness_batch_size={self.cfg.fitness_batch_size}", f"generations={G}",
               f"eval_every_gen={max(1, G // 200)}", "+eval_batch_size=1000", f"{a.noise_key}={noise}", f"es_learning_rate={lr}",
               f"variant={variant}", f"seed={seed}"]
        self.call(cmd, run, f"{algo} {variant} seed {seed}", phase, algo, w, P, G)
        return run

    def picked(self, name):
        path = self.dir / "tuned" / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def pick(self, name, value):
        if not self.dry:
            (self.dir / "tuned" / f"{name}.json").write_text(json.dumps(value))

    # ---- phases --------------------------------------------------------------------------
    def tune(self):
        P, G = self.cfg.tune.population, self.cfg.tune.generations
        for algo in self.algorithms:
            a = self.cfg.algorithms[algo]
            for w in self.widths:
                results = {}
                for noise in a.noise:
                    for lr in a.learning_rates:
                        run = self.es_run("tune", algo, w, P, noise, lr, self.cfg.sweep_seed, G)
                        results[(noise, lr)] = self.best_val(run)
                done = {k: v for k, v in results.items() if v is not None and math.isfinite(v)}
                if done and len(done) == len(results):
                    noise, lr = min(done, key=done.get)
                    self.pick(f"{algo}_w{w}_noise", {"noise": noise, "lr": lr, "val_loss": done[(noise, lr)]})

    def fits(self, algo, w, P):
        """False, and noted in skipped.json, when the population can't fit in memory."""
        cap = self.cfg.algorithms[algo].get("max_population_x_params")
        if cap is None or P * (90 * w * w + 376 * w + 10) <= cap:
            return True
        if not self.dry:
            path = self.dir / "skipped.json"
            skipped = json.loads(path.read_text()) if path.exists() else {}
            skipped[f"{algo}/w{w}/p{P}"] = f"population x parameters > {cap:g}: won't fit in GPU memory"
            path.write_text(json.dumps(skipped, indent=2))
        return False

    def rates_with_edges(self, algo, w, P, noise, seed):
        """Run every listed rate; while the best is at an end, add the next one out (x/÷ 10^0.5)."""
        rates = sorted(self.cfg.algorithms[algo].learning_rates)
        G = self.generations(P, w)
        results = {lr: self.best_val(self.es_run("sweep", algo, w, P, noise, lr, seed, G)) for lr in rates}
        for _ in range(2):
            done = {lr: v for lr, v in results.items() if v is not None and math.isfinite(v)}
            if self.dry or len(done) < len(results):
                break
            best = min(done, key=done.get)
            if best not in (rates[0], rates[-1]):
                break
            lr = float(f"{(best / 10 ** 0.5 if best == rates[0] else best * 10 ** 0.5):.3g}")
            rates = sorted(rates + [lr])
            results[lr] = self.best_val(self.es_run("sweep", algo, w, P, noise, lr, seed, G))
        return results

    def sweep(self):
        for algo in self.algorithms:
            for w in self.widths:
                tuned = self.picked(f"{algo}_w{w}_noise")
                if tuned is None and not self.dry:
                    print(f"skip {algo} w={w}: run the tune phase first", flush=True)
                    continue
                noise = tuned["noise"] if tuned else self.cfg.algorithms[algo].noise[0]
                for P in self.cfg.populations:
                    if not self.fits(algo, w, P):
                        continue
                    results = self.rates_with_edges(algo, w, P, noise, self.cfg.sweep_seed)
                    done = {lr: v for lr, v in results.items() if v is not None and math.isfinite(v)}
                    if done and len(done) == len(results):
                        best = min(done, key=done.get)
                        self.pick(f"{algo}_w{w}_p{P}_lr", {"noise": noise, "lr": best, "val_loss": done[best]})

    def seeds(self):
        for algo in self.algorithms:
            for w in self.widths:
                for P in self.cfg.populations:
                    if not self.fits(algo, w, P):
                        continue
                    tuned = self.picked(f"{algo}_w{w}_p{P}_lr")
                    if tuned is None and not self.dry:
                        continue
                    noise, lr = (tuned["noise"], tuned["lr"]) if tuned else (0, 0)
                    for seed in self.cfg.extra_seeds:
                        self.es_run("seeds", algo, w, P, noise, lr, seed, self.generations(P, w))

    def adamw(self):
        a = self.cfg.adamw
        for w in self.widths:
            for lr in a.learning_rates:
                for seed in a.seeds:
                    variant = f"scaling_w{w}_lr{lr:g}"
                    run = self.run_dir("SGD", "adamw_clipped", variant, seed)
                    cmd = [PY, "src/optim_env/tools/train.py", "--config-name", "study_sgd", f"dataset={self.cfg.dataset}",
                           "model=jax/cnn_mnist", f"model.features=[{w},{2 * w},{4 * w}]", f"loss={self.cfg.loss}",
                           f"learning_rate={lr}", f"epochs={a.epochs}", f"patience={a.patience}",
                           f"variant={variant}", f"seed={seed}"]
                    self.call(cmd, run, f"adamw {variant} seed {seed}", "adamw", "adamw", w, 0, a.epochs)

    def plan(self, probe):
        for phase in (self.tune, self.sweep, self.seeds):
            phase()
        # log(seconds per generation) = k + a log P + b log macs(w), fitted per algorithm on
        # the probe (median error ~15% on an RTX 3070; small cells are overhead-bound).
        fits = {}
        if probe:
            import numpy as np
            rows = [r for r in csv.reader(open(probe)) if r[5] == "ok"]
            for algo in self.algorithms:
                pts = np.array([(int(r[2]), macs(int(r[1])), float(r[4])) for r in rows if r[0] == algo])
                X = np.c_[np.ones(len(pts)), np.log(pts[:, 0]), np.log(pts[:, 1])]
                fits[algo] = np.linalg.lstsq(X, np.log(pts[:, 2]), rcond=None)[0]
        hours, counts = {}, {}
        for phase, algo, w, P, G in self.planned:
            counts[(phase, algo)] = counts.get((phase, algo), 0) + 1
            if algo in fits:
                k, a, b = fits[algo]
                s_gen = math.exp(k + a * math.log(P) + b * math.log(macs(w)))
                hours[(phase, algo)] = hours.get((phase, algo), 0) + (20 + G * s_gen) / 3600
        for key in sorted(counts):
            print(f"{key[0]:6s} {key[1]:8s} {counts[key]:4d} runs" + (f"  ~{hours[key]:6.1f} GPU h" if key in hours else ""))
        print(f"total {sum(counts.values())} runs" + (f", ~{sum(hours.values()):.1f} GPU h" if hours else "")
              + " (the sweep adds up to 2 rates per cell when the best sits at an end; not counted)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "tune", "sweep", "seeds", "adamw"])
    parser.add_argument("--study", default="scaling_mnist")
    parser.add_argument("--algorithms", nargs="*", default=None)
    parser.add_argument("--widths", nargs="*", type=int, default=None)
    parser.add_argument("--pilot", action="store_true", help="2 widths, 2 populations, short runs")
    parser.add_argument("--probe", default=None, help="plan: probe CSV (algo,w,P,chunk,s_per_gen,status,...)")
    args = parser.parse_args()
    study = Scaling(args)
    if args.command == "plan":
        study.plan(args.probe)
    else:
        getattr(study, args.command)()


if __name__ == "__main__":
    main()

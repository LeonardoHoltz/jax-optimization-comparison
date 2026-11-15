"""Fits for the scaling study (configs/study/scaling_mnist.yaml), from each run's validation
curve in metrics.json. Writes CSVs and plots to outputs/reports/<study>/.

    pixi run python src/optim_env/tools/scaling_analysis.py

1. envelope.csv: for each (algorithm, width, population, seed) and generation G, the lowest
   validation loss reached by G over all learning rates ("tuned at every G").
2. loss_vs_population.csv: per (algorithm, width), L(P) = E + A * P^-alpha at the last G.
3. critical_population.csv: per (algorithm, width, target loss L*), the generations G needed
   to reach L* at each P, fitted as G = G_min * (1 + P_crit / P) (McCandlish et al. 2018).
   If P_crit lines up with L* across widths, the useful population follows the loss; if it
   moves with width at the same L*, it follows model size.
4. frontier.csv: lowest loss per compute bin, with the width and population that reached it,
   and power-law fits L(C), N_opt(C), P_opt(C).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "campaigns"))
from scaling_study import macs  # noqa: E402



def params(w):
    return 90 * w * w + 376 * w + 10


def load_curves(dataset, prefix):
    variant = re.compile(prefix + r"_w(\d+)_p(\d+)_n([\d.e+-]+)_lr([\d.e+-]+)$")
    rows = []
    for m in (ROOT / "outputs" / "runs" / dataset).glob(f"jax_NE/*/{prefix}_w*/seed_*/metrics.json"):
        match = variant.match(m.parent.parent.name)
        curve = json.loads(m.read_text()).get("curve")
        if not match or not curve:
            continue
        w, P, noise, lr = int(match[1]), int(match[2]), float(match[3]), float(match[4])
        g, loss = np.array([c[0] for c in curve]), np.array([c[1] for c in curve], dtype=float)
        loss = np.fmin.accumulate(np.where(np.isfinite(loss), loss, np.inf))  # best checkpoint so far
        rows += [{"algorithm": m.parents[2].name, "width": w, "population": P, "noise": noise, "lr": lr,
                  "seed": int(m.parent.name[5:]), "G": gi + 1, "loss": li} for gi, li in zip(g, loss)]
    return pd.DataFrame(rows)


def envelope(curves, batch):
    keys = ["algorithm", "width", "population", "seed", "G"]
    env = curves.groupby(keys, as_index=False)["loss"].min()
    env["params"] = params(env.width)
    env["compute"] = env.population * env.G * batch * 2 * macs(env.width)
    return env[np.isfinite(env.loss)]


def power_offset(x, E, A, alpha):
    return E + A * np.power(x, -alpha)


def fit_power_offset(x, y):
    """E + A x^-alpha by least squares in log(y), E in [0, min y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    try:
        p, _ = curve_fit(lambda x, E, A, a: np.log(power_offset(x, E, A, a)), x, np.log(y),
                         p0=(0.5 * y.min(), y.max() - 0.5 * y.min(), 0.5),
                         bounds=([0, 0, 0], [y.min() * 0.999, np.inf, 5]), maxfev=20000)
        return dict(zip(("E", "A", "alpha"), p))
    except (RuntimeError, ValueError):
        return {"E": np.nan, "A": np.nan, "alpha": np.nan}


def loss_vs_population(env):
    last = env.loc[env.groupby(["algorithm", "width", "population", "seed"]).G.idxmax()]
    rows = []
    for (algo, w), g in last.groupby(["algorithm", "width"]):
        g = g.groupby("population").loss.mean()
        if len(g) >= 3:
            rows.append({"algorithm": algo, "width": w, "params": params(w), **fit_power_offset(g.index, g.values),
                         **{f"L(P={p})": v for p, v in g.items()}})
    return pd.DataFrame(rows)


def critical_population(env, n_targets=6):
    rows = []
    seed0 = env[env.seed == env.seed.min()]
    for (algo, w), g in seed0.groupby(["algorithm", "width"]):
        reached = g.groupby("population").loss.min()
        # Targets between the worst final loss over populations and the best: every
        # population that is fitted must reach them.
        lo, hi = reached.min(), reached.max()
        for target in np.geomspace(hi * 1.02, hi * 1.5, n_targets):
            steps = {P: c.G[c.loss <= target].min() for P, c in g.groupby("population") if (c.loss <= target).any()}
            if len(steps) < 3:
                continue
            P, G = np.array(list(steps)), np.array(list(steps.values()), float)
            slope, intercept = np.polyfit(1 / P, G, 1)  # G = G_min + G_min * P_crit / P
            rows.append({"algorithm": algo, "width": w, "params": params(w), "target_loss": target,
                         "G_min": intercept, "P_crit": slope / intercept if intercept > 0 else np.nan,
                         "n_populations": len(P), "best_loss": lo})
    return pd.DataFrame(rows)


def frontier(env, bins_per_decade=4):
    rows = []
    for algo, g in env.groupby("algorithm"):
        edges = np.arange(np.floor(np.log10(g.compute.min())), np.ceil(np.log10(g.compute.max())) + 1e-9,
                          1 / bins_per_decade)
        g = g.assign(bin=np.digitize(np.log10(g.compute), edges))
        best = g.loc[g.groupby("bin").loss.idxmin()]
        best = best.assign(loss=np.fmin.accumulate(best.loss.values))  # more compute never hurts
        rows.append(best.assign(algorithm=algo))
    return pd.concat(rows) if rows else pd.DataFrame()


def power_law(x, y):
    """y = k x^a in log-log, returns (a, k)."""
    a, logk = np.polyfit(np.log(x), np.log(y), 1)
    return a, np.exp(logk)


def plots(env, lvp, front, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for algo, g in env[env.seed == env.seed.min()].groupby("algorithm"):
        last = g.loc[g.groupby(["width", "population"]).G.idxmax()]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for w, c in last.groupby("width"):
            ax[0].plot(c.population, c.loss, "o-", label=f"w={w} ({params(w):,} params)")
        ax[0].set(xscale="log", yscale="log", xlabel="population P", ylabel="val loss (end of run)", title=algo)
        ax[0].legend(fontsize=7)
        f = front[front.algorithm == algo]
        ax[1].scatter(g.compute, g.loss, s=2, c=np.log2(g.population), alpha=0.3)
        ax[1].plot(f.compute, f.loss, "k-", label="frontier")
        ax[1].set(xscale="log", yscale="log", xlabel="compute (FLOPs)", ylabel="val loss", title="color: log2 P")
        fig.tight_layout()
        fig.savefig(out / f"{algo}_scaling.png", dpi=110)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", default="scaling_mnist")
    parser.add_argument("--pilot", action="store_true", help="the driver's --pilot runs")
    args = parser.parse_args()
    cfg = OmegaConf.load(ROOT / "src" / "optim_env" / "configs" / "study" / f"{args.study}.yaml")
    if args.pilot:
        cfg.prefix = "pilot" + cfg.prefix
    out = ROOT / "outputs" / "reports" / (args.study + ("_pilot" if args.pilot else ""))
    out.mkdir(parents=True, exist_ok=True)

    dataset = OmegaConf.load(ROOT / "src" / "optim_env" / "configs" / "dataset" / f"{cfg.dataset}.yaml").name
    curves = load_curves(dataset, cfg.prefix)
    if curves.empty:
        raise SystemExit(f"No scaling runs with curves under outputs/runs/{dataset}")
    env = envelope(curves, cfg.fitness_batch_size)
    env.to_csv(out / "envelope.csv", index=False)
    lvp = loss_vs_population(env)
    lvp.to_csv(out / "loss_vs_population.csv", index=False)
    critical_population(env).to_csv(out / "critical_population.csv", index=False)
    front = frontier(env)
    front.to_csv(out / "frontier.csv", index=False)
    for algo, f in front.groupby("algorithm"):
        if len(f) >= 4:
            print(f"{algo}: L(C) {fit_power_offset(f.compute, f.loss)} | "
                  f"N_opt ~ C^{power_law(f.compute, f.params)[0]:.2f} | P_opt ~ C^{power_law(f.compute, f.population)[0]:.2f}")
    plots(env, lvp, front, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

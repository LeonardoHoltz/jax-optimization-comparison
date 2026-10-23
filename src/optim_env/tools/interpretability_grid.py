"""Compare what trained image models (CNN or LeNet-5) look at: one row per run, one
column per class, rows sorted by test accuracy.

  --method features   feature visualization: the image that maximizes each class logit
                      (interpretability/activation_maximization.py). A red frame marks an
                      image the model gives to another class.
  --method saliency   saliency map of the first test image of each class, for its true
                      class (interpretability/saliency.py). The top row shows the images.
                      "on object" is the share of saliency that lands on the object's own
                      pixels, over the first 1,000 test images.

Usage:
    pixi run python src/optim_env/tools/interpretability_grid.py \\
        'outputs/runs/mnist/jax_*/*/paper_hpo/seed_0' --method saliency --out mnist_saliency.png
For inner layers of one model, see visualize_neurons.py.
"""

import argparse
import glob
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optim_env.analysis.loading import load_dataset, load_run, predict
from optim_env.interpretability.activation_maximization import layer_sizes, maximize_activation
from optim_env.interpretability.saliency import saliency
from optim_env.jax_env.models.cnn import CNN, LeNet5

CLASS_NAMES = {"fashion_mnist": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                                 "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]}
OBJECT = 0.1  # pixels brighter than this belong to the digit or garment


def display_name(run) -> str:
    """e.g. "Sep-CMA-ES", or "adamw-clipped (gradient)" for gradient runs."""
    name = run.cfg.algorithm.replace("_", "-")
    return f"{name} (gradient)" if run.cfg.method == "SGD" else name


def first_of_each_class(y, n_classes: int) -> np.ndarray:
    return np.array([np.flatnonzero(y == c)[0] for c in range(n_classes)])


def on_object_share(sal, X) -> float:
    """Share of total saliency on pixels brighter than OBJECT."""
    return float((sal * (X > OBJECT)).sum() / sal.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="Run directories or glob patterns")
    parser.add_argument("--method", choices=["features", "saliency"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    run_dirs = sorted({p for pattern in args.run_dirs for p in glob.glob(pattern)})
    if not run_dirs:
        parser.error(f"no run directories match {args.run_dirs}")
    runs = [load_run(d) for d in run_dirs]
    for run in runs:
        if not isinstance(run.model, (CNN, LeNet5)):
            parser.error(f"{run.run_dir}: {type(run.model).__name__} is not an image model")
    if len({run.cfg.dataset.name for run in runs}) > 1:
        parser.error("all runs must share one dataset")

    test = load_dataset(runs[0].cfg).get_test_ds()
    n_classes = runs[0].cfg.dataset.num_classes
    names = CLASS_NAMES.get(runs[0].cfg.dataset.name, [str(c) for c in range(n_classes)])
    idx = first_of_each_class(test.y, n_classes)
    accuracy = [float((predict(run, test.X).argmax(-1) == test.y).mean()) for run in runs]
    order = np.argsort(accuracy)[::-1]

    top_row = args.method == "saliency"
    n_rows = len(runs) + top_row
    fig, axes = plt.subplots(n_rows, n_classes, figsize=(1.05 * n_classes, 1.12 * n_rows + 0.4), squeeze=False)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for c in range(n_classes):
        axes[0, c].set_title(names[c], fontsize=9)
    if top_row:
        for c in range(n_classes):
            axes[0, c].imshow(test.X[idx[c]].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, 0].set_ylabel(f"input\n{(test.X[:1000] > OBJECT).mean():.0%} object", fontsize=8,
                              rotation=0, ha="right", va="center")

    for r, i in enumerate(order, start=top_row):
        run = runs[i]
        label = f"{display_name(run)}\n{accuracy[i]:.3f}"
        if args.method == "features":
            images, _ = maximize_activation(run.model, run.params, list(layer_sizes(run.params))[-1], list(range(n_classes)))
            predicted = run.model.apply(run.params, images.reshape(n_classes, -1)).argmax(-1)
            for c in range(n_classes):
                axes[r, c].imshow(images[c], cmap="gray", vmin=0, vmax=1)
                if int(predicted[c]) != c:
                    for spine in axes[r, c].spines.values():
                        spine.set_color("#c0392b")
                        spine.set_linewidth(2.5)
        else:
            maps = np.asarray(saliency(run.model, run.params, test.X[idx], test.y[idx])).reshape(-1, 28, 28)
            for c in range(n_classes):
                # Each map scaled to its own 99th percentile, so faint maps stay visible.
                axes[r, c].imshow(maps[c], cmap="magma", vmin=0, vmax=max(np.percentile(maps[c], 99), 1e-12))
            sample = test.X[:1000]
            share = on_object_share(np.asarray(saliency(run.model, run.params, sample, test.y[:1000])), sample)
            label += f"\n{share:.0%} on object"
        axes[r, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")

    fig.tight_layout(pad=0.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

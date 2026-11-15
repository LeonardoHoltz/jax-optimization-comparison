"""Compare what trained image models (CNN or LeNet-5) look at: one row per run, one
column per class, rows sorted by test accuracy.

  --method features   feature visualization: the image that maximizes each class logit
                      (interpretability/activation_maximization.py). A red frame marks an
                      image the model gives to another class. The top row is the class mean:
                      all training images of the class summed and scaled to [0, 1]. "match"
                      is the Pearson correlation between each drawing and its class mean,
                      averaged over classes (0: unrelated, 1: same picture up to brightness).
  --method saliency   signed saliency map of the first test image of each class, for its
                      true class (interpretability/saliency.py): red where brightening a
                      pixel raises the score, blue where it lowers it. The top row shows
                      the images.
                      "on object" is the share of saliency that lands on the object's own
                      pixels, over the first 1,000 test images.
  --method integrated_gradients
                      same layout, integrated gradients from a black image. Black pixels get
                      zero by construction, so instead of "on object" it reports "same sign":
                      the share of neighbouring object pixels whose maps agree in sign (50% is
                      a random checkerboard, higher means coherent patches).
  --method gradcam    same layout, Grad-CAM of the last conv layer (red only: clipped at zero),
                      one color scale per row, with the "on object" share.

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
from optim_env.interpretability.saliency import grad_cam, integrated_gradients, saliency
from optim_env.jax_env.models.cnn import CNN, LeNet5

CLASS_NAMES = {"fashion_mnist": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                                 "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]}
OBJECT = 0.1  # pixels brighter than this belong to the digit or garment
MAPS = {"saliency": saliency, "integrated_gradients": integrated_gradients, "gradcam": grad_cam}


def display_name(run) -> str:
    """e.g. "Sep-CMA-ES", "adamw-clipped (gradient)"; untuned runs say so, for grids
    that mix them with tuned ones."""
    name = run.cfg.algorithm.replace("_", "-")
    name += " (gradient)" if run.cfg.method == "SGD" else ""
    return name + ("\nuntuned" if run.cfg.variant.endswith("baseline") else "")


def first_of_each_class(y, n_classes: int) -> np.ndarray:
    return np.array([np.flatnonzero(y == c)[0] for c in range(n_classes)])


def on_object_share(sal, X) -> float:
    """Share of total saliency on pixels brighter than OBJECT."""
    return float((sal * (X > OBJECT)).sum() / sal.sum())


def same_sign_share(maps, X, side=28) -> float:
    """Share of side-by-side or stacked pixel pairs, both on the object and both nonzero,
    with one sign. Zeros (e.g. units switched off) are left out rather than counted as a flip."""
    m, on = maps.reshape(-1, side, side), X.reshape(-1, side, side) > OBJECT
    pairs = [(a, b, both & (a * b != 0)) for a, b, both in
             [(m[:, :, 1:], m[:, :, :-1], on[:, :, 1:] & on[:, :, :-1]), (m[:, 1:], m[:, :-1], on[:, 1:] & on[:, :-1])]]
    agree = sum(((a * b > 0) & both).sum() for a, b, both in pairs)
    return float(agree / sum(both.sum() for _, _, both in pairs))


def class_means(X, y, n_classes: int) -> np.ndarray:
    """(n_classes, features) sum of each class's images, scaled so its brightest pixel is 1."""
    sums = np.stack([X[y == c].sum(0) for c in range(n_classes)])
    return sums / sums.max(1, keepdims=True)


def match(images, means) -> float:
    """Mean over classes of the Pearson correlation between image c and mean c."""
    a, b = images.reshape(len(means), -1), means.reshape(len(means), -1)
    return float(np.mean([np.corrcoef(i, m)[0, 1] for i, m in zip(a, b)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="Run directories or glob patterns")
    parser.add_argument("--method", choices=["features", *MAPS], required=True)
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

    if args.method == "features":
        train = load_dataset(runs[0].cfg).get_train_ds()
        means = class_means(np.asarray(train.X), np.asarray(train.y), n_classes)
    n_rows = len(runs) + 1
    fig, axes = plt.subplots(n_rows, n_classes, figsize=(1.05 * n_classes, 1.12 * n_rows + 0.4), squeeze=False)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for c in range(n_classes):
        axes[0, c].set_title(names[c], fontsize=9)
    top = means if args.method == "features" else test.X[idx]
    for c in range(n_classes):
        axes[0, c].imshow(top[c].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_ylabel("class mean\n(train)" if args.method == "features"
                          else f"input\n{(test.X[:1000] > OBJECT).mean():.0%} object",
                          fontsize=8, rotation=0, ha="right", va="center")

    for r, i in enumerate(order, start=1):
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
            label += f"\n{match(np.asarray(images), means):.2f} match"
        else:
            attribute = MAPS[args.method]
            maps = np.asarray(attribute(run.model, run.params, test.X[idx], test.y[idx])).reshape(-1, 28, 28)
            for c in range(n_classes):
                # Each map scaled to its own 99th percentile of |value|, symmetric so that
                # white is zero; faint maps stay visible. Grad-CAM shares one scale per row:
                # its clip at zero leaves some maps almost empty, and a per-map scale would
                # blow their leftovers up into bright dots.
                scaled, q = (maps, 99.9) if args.method == "gradcam" else (maps[c], 99)
                top = max(np.percentile(np.abs(scaled), q), 1e-12)
                axes[r, c].imshow(maps[c], cmap="RdBu_r", vmin=-top, vmax=top)
            sample = test.X[:1000]
            sample_maps = np.asarray(attribute(run.model, run.params, sample, test.y[:1000]))
            if args.method == "integrated_gradients":
                label += f"\n{same_sign_share(sample_maps, sample):.0%} same sign"
            else:
                label += f"\n{on_object_share(np.abs(sample_maps), sample):.0%} on object"
        axes[r, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")

    fig.tight_layout(pad=0.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

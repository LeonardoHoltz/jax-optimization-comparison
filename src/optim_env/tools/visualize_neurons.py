"""Draw the image that most excites each unit of a trained image model (CNN or LeNet-5),
via interpretability/activation_maximization.py. Each panel shows the unit, the value
it reaches, and the class the model gives that image.

Usage:
    pixi run python src/optim_env/tools/visualize_neurons.py outputs/runs/mnist/jax_SGD/adamw_clipped/paper_baseline/seed_0
    pixi run python src/optim_env/tools/visualize_neurons.py RUN_DIR --layer Conv_1 --units 0 1 2 3
Default: the last layer (one image per class). The PNG goes to RUN_DIR/interpretability/.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optim_env.analysis.loading import load_run
from optim_env.interpretability.activation_maximization import layer_sizes, maximize_activation
from optim_env.jax_env.models.cnn import CNN, LeNet5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--layer", default=None, help="Default: the output layer")
    parser.add_argument("--units", type=int, nargs="*", default=None, help="Default: every unit of the layer")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--tv-weight", type=float, default=10.0, help="Higher gives smoother images")
    parser.add_argument("--brightness-weight", type=float, default=20.0, help="Higher gives darker, sparser images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run = load_run(args.run_dir)
    if not isinstance(run.model, (CNN, LeNet5)):
        parser.error(f"{type(run.model).__name__} is not an image model; this tool needs CNN or LeNet5")
    sizes = layer_sizes(run.params)
    layer = args.layer or list(sizes)[-1]
    if layer not in sizes:
        parser.error(f"unknown layer {layer}; this model has {sizes}")
    units = args.units if args.units is not None else list(range(sizes[layer]))

    images, reached = maximize_activation(run.model, run.params, layer, units, steps=args.steps,
                                          tv_weight=args.tv_weight, brightness_weight=args.brightness_weight,
                                          seed=args.seed)
    predicted = run.model.apply(run.params, images.reshape(len(units), -1)).argmax(-1)

    cols = min(len(units), 10)
    fig, axes = plt.subplots(math.ceil(len(units) / cols), cols, figsize=(1.6 * cols, 1.9 * math.ceil(len(units) / cols)),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, unit, image, value, cls in zip(axes.flat, units, images, reached, predicted):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"unit {unit}: {float(value):.1f}\nclass {int(cls)}", fontsize=8)
    fig.suptitle(f"{run.label} {layer}", fontsize=9)
    fig.tight_layout()

    out = args.run_dir / "interpretability" / f"max_activation_{layer}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

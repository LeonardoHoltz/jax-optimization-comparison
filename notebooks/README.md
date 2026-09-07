# Notebooks

Analysis and explainability on top of finished runs. Nothing here trains anything — the
notebooks read run directories produced by `scripts/campaigns/*`, so run a campaign first.

```bash
pixi run jupyter lab          # from the repo root
```

Every notebook starts by `chdir`-ing to the repo root, so repo-relative paths
(`outputs/...`, `data/...`) behave the same as they do for the CLI tools no matter where
Jupyter was launched.

## Section: explainability

| Notebook | Question |
|---|---|
| `01_activation_comparison.ipynb` | Do two models compute their answer the same way internally? Compares hidden-unit activations between any two runs. |

Planned additions: per-model feature attribution (integrated gradients / saliency),
attribution *differences* across optimizer families, and Optuna study analysis.

## Supporting code

The notebooks stay short because the machinery lives in `src/optim_env/analysis/`:

- **`loading.py`** — `load_run(run_dir)` rebuilds a trained model (either framework) from
  the Hydra config saved alongside its checkpoint. `tools/test.py` uses the same path, so
  reports and notebooks can't drift apart in how they load a model.
- **`activations.py`** — `hidden_activations(run, X)` returns post-ReLU activations per
  hidden layer. JAX goes through Flax's `capture_intermediates`; PyTorch uses forward
  hooks on the `nn.ReLU` modules.
- **`similarity.py`** — CKA, neuron matching, dead-unit fraction, effective rank.

### Why comparing activations needs those tools

Hidden units have no canonical identity. Two networks computing nearly the same function
can have their units in any order, and with ReLU you can scale a unit's incoming weights
by `a > 0` and its outgoing weights by `1/a` for free. So "unit *i* of A vs unit *i* of B"
is meaningless, and raw magnitudes aren't comparable either.

- **CKA** compares whole layers, invariant to rotation (hence permutation) and scaling.
- **Neuron matching** undoes the permutation explicitly, via optimal assignment on
  correlation, so individual units can be compared.

Interpret any similarity number against the **seed-to-seed** baseline: two runs of the
same optimizer differing only by seed set the floor for "different by accident". A
cross-optimizer score at that level is not evidence of a different representation.

## A note on committing notebooks

Outputs are stripped from the notebooks in this folder — stored plots from throwaway runs
are easy to mistake for real findings. To strip before committing:

```bash
pixi run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to notebook --inplace notebooks/*.ipynb
```

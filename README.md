# JAX Optimization Comparison (`optim-env`)

A research framework for comparing **gradient-based optimization** against **evolutionary
strategies (neuroevolution)** on the same model, data and budget — with a native PyTorch
path alongside the JAX one so results aren't an artifact of a single framework.

The reference task reproduces the MLP baseline on California Housing from Gorishniy et
al. (2021), *"Revisiting Deep Learning Models for Tabular Data"*, using that paper's own
tuned architecture and its exact train/val/test split.

## 🚀 Key Features

- **One model, many optimizers**: optax gradient methods, evosax evolutionary strategies
  and `torch.optim`, all selected by Hydra config against an identical architecture.
- **Two frameworks, shared data**: `jax_env/` and `torch_env/` hold the framework-specific
  models and trainers; datasets, splits and preprocessing are shared, so only the final
  collate step differs.
- **Hyperparameter optimization for neuroevolution**: Optuna/TPE search over ES
  hyperparameters, with each tuned result kept next to its untuned baseline.
- **Seed replication built in**: campaigns run multiple seeds and reports carry mean/std,
  so results come with a variance estimate rather than a single point.
- **Structured, traceable outputs**: every run lands in a predictable path and every
  campaign records a manifest with its git SHA and parameters.

## 📦 Installation

This project uses [Pixi](https://pixi.sh/) for environment management.

```bash
git clone <repo-url>
cd jax-optimization-comparison
pixi install
```

## 🧪 Types of Experiments

An experiment is defined by four choices — **framework** (`jax`/`torch`), **method**
(`SGD` gradient descent / `NE` neuroevolution), **algorithm**, and **dataset** — plus a
**variant** that distinguishes a plain run from a tuned or swept one.

### 1. Gradient-based training (`SGD`)

Backpropagation with a first-order optimizer. Available in both frameworks:

| Framework | Config | Algorithms |
|---|---|---|
| JAX (optax) | `california_mlp_sgd_exp` | `jax/adam`, `jax/adamw`, `jax/sgd` |
| PyTorch (`torch.optim`) | `california_mlp_torch_exp` | `torch/adam`, `torch/adamw`, `torch/sgd` |

Trained in mini-batches over epochs; dropout active during training, disabled for eval.

### 2. Neuroevolution (`NE`)

Gradient-free, population-based search with [evosax](https://github.com/RobertTLange/evosax).
Config `california_mlp_es_exp`. Each generation evaluates a whole population full-batch,
so there are generations rather than epochs, and no backward pass at all.

| Family | Algorithms |
|---|---|
| Distribution-based | `jax/sep_cma_es`, `jax/snes`, `jax/cr_fm_nes`, `jax/open_es`, `jax/pgpe`, `jax/ars` |
| Population-based | `jax/particle_swarm`, `jax/simple_ga`, `jax/differential_evolution` |

`jax/cma_es` (full covariance) also exists but is impractical on large models: it needs an
O(D²) covariance matrix, ~220GB for the 235K-parameter California MLP. Use `sep_cma_es`
(diagonal, O(D)) instead.

Evosax's defaults assume a generic search space with scale ~1.0, while neural network
weights live on a much smaller scale. `ESTrainer` corrects for this automatically by
deriving the initial spread from the model's own weights — see the extended comment in
`src/optim_env/jax_env/trainers/es_trainer.py` for what goes wrong otherwise.

### 3. Hyperparameter optimization (variant `hpo`)

Optuna with the TPE sampler minimizes validation loss over ES hyperparameters. Each trial
is an independent training subprocess, so JAX state never leaks between trials.

| Hyperparameter | Range | Applies to |
|---|---|---|
| `optimizer.population_size` | 16–96, step 2 | every algorithm |
| `init_std_multiplier` | 0.1–3.0, log | every algorithm |
| `es_learning_rate` | 1e-4–1e-1, log | only algorithms with an internal optimizer (Open-ES, PGPE, SNES, ARS) |

After the search it retrains at full budget with the best values, once per seed in
`--final-seeds` (the campaigns pass their `SEEDS`), saved as `variant=hpo` beside the
`baseline` runs of the same algorithm so the two compare directly. The search itself is
stored in `study.db`, so an interrupted search resumes when re-run with the same
arguments; delete `study.db` to start over.

ES runs save the candidate with the best validation loss (the search mean for
distribution-based algorithms, the best population member otherwise), the same
selection rule the SGD trainers use.

### 4. Sweeps (variant `<dimension>_<value>`)

Sweeping something that isn't part of the output path — architecture, population size —
encodes that dimension in `variant` so runs don't overwrite each other
(e.g. `pop_30`, `model_mlp_2_layers_5`).

### Datasets

| Config | Task | Notes |
|---|---|---|
| `tabm_california` | regression | The paper's exact 13209/3303/4128 split. **Default.** |
| `tabm_adult`, `tabm_microsoft` | classification / regression | Same TabM benchmark format |
| `pmlb_mushroom`, `pmlb_penguins` | classification | Smaller, quick to iterate on |
| `sklearn_california_housing` | regression | Raw sklearn fetch, random split (no held-out test) |

Only TabM datasets ship a genuine held-out test split, so they're the ones `test.py` can
evaluate; the others expose train/val only.

## 📜 Scripts

Campaigns are macro experiments: they train, evaluate and write a report plus a manifest.
All parameters are environment-variable overridable.

| Script | What it runs |
|---|---|
| `scripts/campaigns/california_optimizers.sh` | The headline comparison: gradient (optax), evolutionary (evosax) and PyTorch optimizers on the same MLP, across seeds |
| `scripts/campaigns/california_ne_hpo.sh` | Every ES optimizer with **and** without hyperparameter optimization |
| `scripts/campaigns/pmlb_seed_sweep.sh` | One configuration across seeds — the variance floor |
| `scripts/campaigns/pmlb_model_sweep.sh` | Architectures under a fixed optimizer |
| `scripts/campaigns/pmlb_population_sweep.sh` | Effect of ES population size |
| `scripts/tensorboard.sh` | Launch TensorBoard over `outputs/runs/` |

```bash
# Defaults: 3 seeds, 200 epochs / 300 generations
./scripts/campaigns/california_optimizers.sh

# Cheaper pass
SEEDS="0,1" GENERATIONS=100 ./scripts/campaigns/california_optimizers.sh

# HPO comparison; the HPO stage is expensive (one training run per trial, per optimizer)
N_TRIALS=20 ./scripts/campaigns/california_ne_hpo.sh
./scripts/campaigns/california_ne_hpo.sh "jax/sep_cma_es,jax/pgpe"   # subset
```

`scripts/lib/common.sh` holds the shared helpers (`section`, `write_manifest`).

### Underlying tools

Campaigns are thin wrappers over three entry points, all usable directly:

```bash
# Train a single run (any framework — dispatch is entirely config-driven)
pixi run python src/optim_env/tools/train.py --config-name california_mlp_es_exp optimizer=jax/pgpe
pixi run python src/optim_env/tools/train.py --config-name california_mlp_torch_exp optimizer=torch/adamw

# Sweep with --multirun
pixi run python src/optim_env/tools/train.py --config-name california_mlp_es_exp --multirun \
    optimizer=jax/snes,jax/pgpe seed=0,1,2

# Evaluate finished runs on the held-out test set -> outputs/reports/
pixi run python src/optim_env/tools/test.py --campaign my_report
pixi run python src/optim_env/tools/test.py 'outputs/runs/california/jax_NE/*/*/seed_*' --campaign ne_only

# Hyperparameter search for one ES optimizer
pixi run python src/optim_env/tools/hpo.py --optimizer jax/sep_cma_es --n-trials 20
```

## 📊 Outputs

Everything is written under `outputs/` (gitignored), separated by artifact kind.

```
outputs/
  runs/<dataset>/<framework>_<method>/<algorithm>/<variant>/seed_<n>/
      .hydra/config.yaml      exact config this run used (enough to rebuild the model)
      best_model.msgpack      trained weights — best_model.pt for PyTorch runs
      metrics.json            best validation loss, budget (ES runs)
      events.out.tfevents.*   TensorBoard scalars
      <exp_name>.log

  hpo/<dataset>/<framework>_<method>/<algorithm>/
      trials.csv              every trial: sampled hyperparameters, objective, duration
      best_params.json        winning configuration
      study.db                Optuna study, for resuming
      trial_<n>/              each trial's own run directory

  reports/
      <campaign>.csv          one row per run (per seed)
      <campaign>_summary.csv  aggregated over seeds: mean/std per algorithm + variant
      <campaign>_manifest.json  git SHA, timestamp, parameters the campaign ran with
```

Example paths:

```
outputs/runs/california/jax_NE/PSO/baseline/seed_0/       # untuned PSO
outputs/runs/california/jax_NE/PSO/hpo/seed_0/            # same algorithm, tuned
outputs/runs/california/torch_SGD/AdamW/baseline/seed_2/  # PyTorch, third seed
```

The layout is defined once in `src/optim_env/configs/base/run_layout.yaml` and mirrored
for tooling by `src/optim_env/paths.py` (kept in sync by `tests/test_paths.py`).

Search trials live under `hpo/`, never `runs/` — they are means, not results, and
shouldn't be picked up as if they were real runs.

**Reading a report.** `<campaign>.csv` gives per-seed detail; `<campaign>_summary.csv` is
what you usually want, one line per `(dataset, framework, method, algorithm, variant)`
with `rmse_mean` / `rmse_std` / `n_seeds`. RMSE is in the target's original units (the
model trains on standardized targets, and `test.py` scales the error back). Compare `baseline` against `hpo` rows of the
same algorithm, and read any difference against the seed-to-seed spread before treating
it as real.

## 📥 Downloading Datasets

PMLB and Hugging Face datasets must be downloaded once, using the `--name` and `--root`
values from the corresponding `dataset` config:

```bash
# PMLB
pixi run python -m optim_env.prepare_dataset.pmlb_dataset --name mushroom --root data/pmlb
pixi run python -m optim_env.prepare_dataset.pmlb_dataset --name penguins --root data/pmlb

# Hugging Face — the TabM benchmark datasets, mirrored at
# https://huggingface.co/datasets/rototoHF/tabm-data
pixi run python -m optim_env.prepare_dataset.huggingface --repo-id rototoHF/tabm-data --name california --root data/tabm
pixi run python -m optim_env.prepare_dataset.huggingface --repo-id rototoHF/tabm-data --name adult --root data/tabm
```

`optim_env.prepare_dataset.kaggle` is available for any future Kaggle-backed dataset
(`--dataset <kaggle-slug> --name <name> --root <root>`), though no current config uses it.
sklearn-backed datasets need no download — `scikit-learn` fetches them on first use.

## 📂 Project Structure

```
src/optim_env/
  jax_env/          JAX/Flax models and trainers (SGD + evolutionary)
  torch_env/        PyTorch models, trainer and utilities
  datasets/         DataModules shared by both frameworks (PMLB, TabM, sklearn)
  configs/          Hydra configs; model/optimizer/loss/trainer split into jax/ and torch/
    base/           run_layout.yaml — the shared output-path layout
  tools/            train.py, test.py, hpo.py entry points
  analysis/         Loading finished runs back for analysis
  prepare_dataset/  Dataset download CLIs
  paths.py          Canonical output locations for tooling
scripts/            Campaign scripts and shared shell helpers
tests/              Unit tests
```

## 🔍 Visualization

```bash
./scripts/tensorboard.sh          # or: pixi run tensorboard --logdir outputs/runs
```

## ✅ Tests

```bash
pixi run pytest tests/
```

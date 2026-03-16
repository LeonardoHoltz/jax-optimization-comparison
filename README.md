# JAX Optimization Comparison (`jax-optim-env`)

A research-oriented framework for benchmarking and comparing stochastic gradient-based optimization (SGD) with Evolutionary Strategies (ES) using the power of **JAX**.

This project provides a robust infrastructure for comparing traditional gradient-based methods with population-based and distribution-based evolution strategies across diverse datasets from the **PMLB** library.

## 🚀 Key Features

- **Dynamic Optimizer Instantiation**: Seamlessly switch between gradient-based and evolutionary strategies using Hydra-based configurations.
- **Unified Training Infrastructure**: Standardized trainers for both ES and SGD with consistent metrics and evaluation loops.
- **Automated Benchmarking**: Includes scripts to automate experiments across multiple datasets and algorithms.
- **High Performance**: Built on top of **JAX** and **Flax** for XLA-accelerated computing (CPU/GPU/TPU).
- **Flexible Data Handling**: Integrated with **PMLB** for easy access to standardized machine learning datasets with automated preprocessing and type-safe handling.
- **Visualization & Persistence**: Built-in support for TensorBoard logging and model checkpointing.

## 📦 Installation

This project uses [Pixi](https://pixi.sh/) for environment management.

```bash
# Clone the repository
git clone <repo-url>
cd jax-optimization-comparison

# Install dependencies and setup environment
pixi install
```

## 🏃 Usage

### Running the Baseline Optimization Comparison (BOC)
The BOC script runs several baseline training experiments.

```bash
./scripts/run_boc.sh [iterations]
```

### Individual Training Runs
It's possible to run specific experiments using the training tools:

```bash
# Train using SGD (Adam) on Mushroom
pixi run python src/jax_optim_env/tools/train.py --config-name sgd_exp dataset=pmlb_mushroom

# Train using CMA-ES on Penguins
pixi run python src/jax_optim_env/tools/train.py --config-name es_exp dataset=pmlb_penguins optimizer=cma_es
```

### Visualization
Monitor training progress with TensorBoard:

```bash
pixi run tensorboard --logdir outputs
```

## 📂 Project Structure

- `src/jax_optim_env/trainers/`: Core logic for ES and SGD training loops.
- `src/jax_optim_env/configs/`: Hydra configurations for models, datasets, and optimizers.
- `src/jax_optim_env/datasets/`: Dataset managers (PMLB integration).
- `src/jax_optim_env/tools/`: Entry point scripts for training.
- `scripts/`: Automation scripts (like `run_boc.sh`).
- `tests/`: Unit tests for metrics and checkpointing.

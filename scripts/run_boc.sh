#!/bin/bash
# Baseline Optimization Comparison (BOC)
# Runs 2 datasets (Mushroom, Penguins) across 3 optimizers (SGD, CMA-ES, SimpleGA)

GENS=${1:-100}
EVAL_FREQ=1
export HYDRA_FULL_ERROR=1

echo "Starting Baseline Optimization Comparison (BOC) with $GENS iterations..."

# 1. Mushroom Dataset
echo "=> Running Mushroom Matrix..."
pixi run python src/jax_optim_env/tools/train.py --config-name sgd_exp dataset=pmlb_mushroom epochs=$GENS eval_every=$EVAL_FREQ
pixi run python src/jax_optim_env/tools/train.py --config-name es_exp dataset=pmlb_mushroom generations=$GENS eval_every_gen=$EVAL_FREQ optimizer=cma_es
pixi run python src/jax_optim_env/tools/train.py --config-name es_exp dataset=pmlb_mushroom generations=$GENS eval_every_gen=$EVAL_FREQ optimizer=simple_ga

# 2. Penguins Dataset
echo "=> Running Penguins Matrix..."
pixi run python src/jax_optim_env/tools/train.py --config-name sgd_exp dataset=pmlb_penguins epochs=$GENS eval_every=$EVAL_FREQ
pixi run python src/jax_optim_env/tools/train.py --config-name es_exp dataset=pmlb_penguins generations=$GENS eval_every_gen=$EVAL_FREQ optimizer=cma_es
pixi run python src/jax_optim_env/tools/train.py --config-name es_exp dataset=pmlb_penguins generations=$GENS eval_every_gen=$EVAL_FREQ optimizer=simple_ga

echo "BOC Complete."

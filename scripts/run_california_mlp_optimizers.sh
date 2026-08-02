#!/bin/bash
# Reproduces (in JAX) the MLP baseline on California Housing from
# Gorishniy et al. (2021), "Revisiting Deep Learning Models for Tabular Data",
# training the same architecture (configs/model/mlp_california.yaml) with multiple
# optimizers: gradient-based (optax) and evolutionary strategies (evosax).
#
# Note: mlp_california has ~235K parameters. Full-covariance CMA-ES needs an O(D^2)
# covariance matrix (~220GB here), so the ES sweep uses sep_cma_es (diagonal-covariance
# CMA-ES, O(D)) instead of plain cma_es.

EPOCHS=${EPOCHS:-200}
GENERATIONS=${GENERATIONS:-300}
SEED=${SEED:-0}
GD_OPTIMIZERS=${1:-"adam,adamw,sgd"}
ES_OPTIMIZERS=${2:-"sep_cma_es,simple_ga,particle_swarm,open_es,pgpe,snes,ars,differential_evolution,cr_fm_nes"}
export HYDRA_FULL_ERROR=1

echo "Starting California Housing MLP optimizer comparison..."

echo "=> Running gradient-based (optax) trainings: $GD_OPTIMIZERS"
pixi run python src/jax_optim_env/tools/train.py --config-name california_mlp_sgd_exp --multirun \
    optimizer=$GD_OPTIMIZERS epochs=$EPOCHS seed=$SEED

echo "=> Running evolutionary-strategy (evosax) trainings: $ES_OPTIMIZERS"
pixi run python src/jax_optim_env/tools/train.py --config-name california_mlp_es_exp --multirun \
    optimizer=$ES_OPTIMIZERS generations=$GENERATIONS seed=$SEED

echo "California Housing MLP optimizer comparison complete. See outputs/SGD_*_california and outputs/NE_*_california."

#!/bin/bash
# Reproduces (in JAX) the MLP baseline on California Housing from
# Gorishniy et al. (2021), "Revisiting Deep Learning Models for Tabular Data",
# training the same architecture (configs/model/jax/mlp_california.yaml) with multiple
# optimizers: gradient-based (optax) and evolutionary strategies (evosax).
#
# Note: mlp_california has ~235K parameters. Full-covariance CMA-ES needs an O(D^2)
# covariance matrix (~220GB here), so the ES sweep uses sep_cma_es (diagonal-covariance
# CMA-ES, O(D)) instead of plain cma_es.

EPOCHS=${EPOCHS:-200}
GENERATIONS=${GENERATIONS:-300}
SEED=${SEED:-0}
GD_OPTIMIZERS=${1:-"jax/adam,jax/adamw,jax/sgd"}
ES_OPTIMIZERS=${2:-"jax/sep_cma_es,jax/simple_ga,jax/particle_swarm,jax/open_es,jax/pgpe,jax/snes,jax/ars,jax/differential_evolution,jax/cr_fm_nes"}
export HYDRA_FULL_ERROR=1

echo "Starting California Housing MLP optimizer comparison..."

echo "=> Running gradient-based (optax) trainings: $GD_OPTIMIZERS"
pixi run python src/jax_optim_env/tools/train.py --config-name california_mlp_sgd_exp --multirun \
    optimizer=$GD_OPTIMIZERS epochs=$EPOCHS seed=$SEED

echo "=> Running evolutionary-strategy (evosax) trainings: $ES_OPTIMIZERS"
pixi run python src/jax_optim_env/tools/train.py --config-name california_mlp_es_exp --multirun \
    optimizer=$ES_OPTIMIZERS generations=$GENERATIONS seed=$SEED

echo "California Housing MLP optimizer comparison complete. See outputs/jax_SGD_*_california and outputs/jax_NE_*_california."

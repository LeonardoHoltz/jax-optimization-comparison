#!/bin/bash
# Reproduces (in JAX) the MLP baseline on California Housing from Gorishniy et al. (2021),
# "Revisiting Deep Learning Models for Tabular Data", training the same architecture
# (configs/model/jax/mlp_california.yaml) with gradient-based (optax), evolutionary
# (evosax) and native-PyTorch optimizers, replicated over several seeds.
#
# Note: mlp_california has ~235K parameters. Full-covariance CMA-ES needs an O(D^2)
# covariance matrix (~220GB here), so the ES sweep uses sep_cma_es (diagonal-covariance
# CMA-ES, O(D)) instead of plain cma_es.
source "$(dirname "$0")/../lib/common.sh"

CAMPAIGN=${CAMPAIGN:-california_optimizers}
EPOCHS=${EPOCHS:-200}
GENERATIONS=${GENERATIONS:-300}
SEEDS=${SEEDS:-"0,1,2"}
GD_OPTIMIZERS=${1:-"jax/adam,jax/adamw,jax/sgd"}
ES_OPTIMIZERS=${2:-"jax/sep_cma_es,jax/simple_ga,jax/particle_swarm,jax/open_es,jax/pgpe,jax/snes,jax/ars,jax/differential_evolution,jax/cr_fm_nes"}
TORCH_OPTIMIZERS=${3:-"torch/adam,torch/adamw,torch/sgd"}

section "Gradient-based (optax): $GD_OPTIMIZERS | seeds=$SEEDS"
$TRAIN --config-name california_mlp_sgd_exp --multirun \
    optimizer=$GD_OPTIMIZERS epochs=$EPOCHS seed=$SEEDS

section "Evolutionary (evosax): $ES_OPTIMIZERS | seeds=$SEEDS"
$TRAIN --config-name california_mlp_es_exp --multirun \
    optimizer=$ES_OPTIMIZERS generations=$GENERATIONS seed=$SEEDS

section "Native PyTorch: $TORCH_OPTIMIZERS | seeds=$SEEDS"
$TRAIN --config-name california_mlp_torch_exp --multirun \
    optimizer=$TORCH_OPTIMIZERS epochs=$EPOCHS seed=$SEEDS

section "Evaluating on the held-out test set"
$TEST 'outputs/runs/california/*/*/*/seed_*' --campaign "$CAMPAIGN"

write_manifest "$CAMPAIGN" "epochs=$EPOCHS" "generations=$GENERATIONS" "seeds=$SEEDS" \
    "gd_optimizers=$GD_OPTIMIZERS" "es_optimizers=$ES_OPTIMIZERS" "torch_optimizers=$TORCH_OPTIMIZERS"

echo "Complete. See outputs/reports/${CAMPAIGN}.csv and ${CAMPAIGN}_summary.csv"

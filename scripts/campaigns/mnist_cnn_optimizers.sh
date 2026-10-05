#!/bin/bash
# Image-classification counterpart to the tabular California campaign: one small CNN on
# MNIST, trained by gradient descent (optax) and by evolutionary strategies (evosax).
#
# Budget note: "equal budget" has no clean definition across these two families. Gradient
# runs get EPOCHS passes over the full training set; ES runs get GENERATIONS population
# evaluations on mini-batches of FITNESS_BATCH. ES is given a deliberately generous
# generation count — it still sees far fewer useful updates per wall-clock second, and the
# comparison is about reachable accuracy, not matched FLOPs.
source "$(dirname "$0")/../lib/common.sh"

CAMPAIGN=${CAMPAIGN:-mnist_cnn_optimizers}
EPOCHS=${EPOCHS:-10}
GENERATIONS=${GENERATIONS:-2000}
EVAL_EVERY_GEN=${EVAL_EVERY_GEN:-100}
FITNESS_BATCH=${FITNESS_BATCH:-256}
SEEDS=${SEEDS:-"0"}
GD_OPTIMIZERS=${1:-"jax/adam,jax/sgd"}
ES_OPTIMIZERS=${2:-"jax/sep_cma_es,jax/snes,jax/pgpe,jax/open_es,jax/ars,jax/particle_swarm,jax/simple_ga,jax/differential_evolution,jax/cr_fm_nes"}

section "Gradient-based (optax): $GD_OPTIMIZERS | epochs=$EPOCHS | seeds=$SEEDS"
$TRAIN --config-name mnist_cnn_sgd_exp --multirun \
    optimizer=$GD_OPTIMIZERS epochs=$EPOCHS seed=$SEEDS

section "Evolutionary (evosax): $ES_OPTIMIZERS | generations=$GENERATIONS | seeds=$SEEDS"
$TRAIN --config-name mnist_cnn_es_exp --multirun \
    optimizer=$ES_OPTIMIZERS generations=$GENERATIONS eval_every_gen=$EVAL_EVERY_GEN \
    fitness_batch_size=$FITNESS_BATCH seed=$SEEDS

section "Evaluating on the held-out test set"
$TEST 'outputs/runs/mnist/*/*/*/seed_*' --campaign "$CAMPAIGN"

write_manifest "$CAMPAIGN" "epochs=$EPOCHS" "generations=$GENERATIONS" \
    "fitness_batch=$FITNESS_BATCH" "seeds=$SEEDS" \
    "gd_optimizers=$GD_OPTIMIZERS" "es_optimizers=$ES_OPTIMIZERS"

echo "Complete. See outputs/reports/${CAMPAIGN}.csv and ${CAMPAIGN}_summary.csv"

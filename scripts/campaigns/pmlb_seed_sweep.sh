#!/bin/bash
# Seed replication for a single SGD configuration on a PMLB dataset -- the variance
# baseline: how much does the result move with nothing but the seed changing?
source "$(dirname "$0")/../lib/common.sh"

CAMPAIGN=${CAMPAIGN:-pmlb_seed_sweep}
DATASET=${DATASET:-pmlb_penguins}
MODEL=${MODEL:-jax/linear}
EPOCHS=${EPOCHS:-100}
SEEDS=${1:-"40,41,42,43,44"}

section "Seed sweep on $DATASET | model=$MODEL | seeds=$SEEDS"
$TRAIN --config-name sgd_exp --multirun \
    dataset=$DATASET model=$MODEL epochs=$EPOCHS seed=$SEEDS

write_manifest "$CAMPAIGN" "dataset=$DATASET" "model=$MODEL" "epochs=$EPOCHS" "seeds=$SEEDS"
echo "Complete. Runs under outputs/runs/*/jax_SGD/*/baseline/seed_*"

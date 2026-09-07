#!/bin/bash
# Sweeps the evolutionary population size on a PMLB dataset. Population size is encoded in
# `variant` because run_dir keys on dataset/framework_method/algorithm/variant/seed --
# without it, every swept size would write into the same directory.
source "$(dirname "$0")/../lib/common.sh"

CAMPAIGN=${CAMPAIGN:-pmlb_population_sweep}
DATASET=${DATASET:-pmlb_penguins}
MODEL=${MODEL:-jax/mlp_3_layers_5}
GENERATIONS=${GENERATIONS:-100}
SEEDS=${SEEDS:-"42"}
POP_SIZES=${1:-"50,100,200,300,400,500,1000"}

section "Population sweep on $DATASET | sizes=$POP_SIZES | seeds=$SEEDS"
$TRAIN --config-name simple_ga_exp --multirun \
    dataset=$DATASET model=$MODEL generations=$GENERATIONS seed=$SEEDS \
    optimizer.population_size=$POP_SIZES \
    'variant=pop_${optimizer.population_size}'

write_manifest "$CAMPAIGN" "dataset=$DATASET" "model=$MODEL" "generations=$GENERATIONS" \
    "seeds=$SEEDS" "population_sizes=$POP_SIZES"
echo "Complete. Runs under outputs/runs/*/jax_NE/*/pop_*/seed_*"

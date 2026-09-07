#!/bin/bash
# Compares model architectures under a fixed optimizer on a PMLB dataset.
#
# The architecture goes into `variant` because run_dir keys on
# dataset/framework_method/algorithm/variant/seed -- without it, every swept model would
# write into the same directory. The label is built here in the shell, per model, rather
# than interpolated in the config: config interpolation would have to reference a field
# like ${model.hidden_dims}, which doesn't exist on every model (jax/linear has none) and
# renders list values with spaces and brackets.
source "$(dirname "$0")/../lib/common.sh"

CAMPAIGN=${CAMPAIGN:-pmlb_model_sweep}
DATASET=${DATASET:-pmlb_penguins}
EPOCHS=${EPOCHS:-200}
SEEDS=${SEEDS:-"40"}
MODELS=${1:-"jax/linear,jax/mlp,jax/mlp_2_layers_5,jax/mlp_3_layers_5"}

IFS=',' read -ra MODEL_LIST <<< "$MODELS"
for MODEL in "${MODEL_LIST[@]}"; do
    section "Model $MODEL on $DATASET | seeds=$SEEDS"
    $TRAIN --config-name sgd_exp --multirun \
        dataset=$DATASET model="$MODEL" epochs=$EPOCHS seed=$SEEDS \
        variant="model_${MODEL##*/}"
done

write_manifest "$CAMPAIGN" "dataset=$DATASET" "models=$MODELS" "epochs=$EPOCHS" "seeds=$SEEDS"
echo "Complete. Runs under outputs/runs/*/jax_SGD/*/model_*/seed_*"

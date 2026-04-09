#!/bin/bash
# Multiple experiment runs with different seeds

GENS=100
MODEL=linear
EXP_NAME="SGD_adam_penguins_seed_\${seed}" # '$' here will refer to what is in hydra
export HYDRA_FULL_ERROR=1
SEEDS=${1:-"40,41,42,43,44"}


echo "Starting seed multirun experiment..."

# Penguins Dataset
echo "=> Running trainings..."
pixi run python src/jax_optim_env/tools/train.py --config-name sgd_exp --multirun \
    exp_name=$EXP_NAME dataset=pmlb_penguins epochs=$GENS \
    model=$MODEL seed=$SEEDS

echo "Seed multirun complete."

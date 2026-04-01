#!/bin/bash
# Baseline Optimization Comparison (BOC)
# Runs 2 datasets (Mushroom, Penguins) across 3 optimizers (SGD, CMA-ES, SimpleGA)

GENS=${1:-100}
EVAL_FREQ=1
MODEL=mlp
EXP_NAME=SGD_adam_penguins_${GENS}_epochs_${MODEL}_model
export HYDRA_FULL_ERROR=1


echo "Starting multirun experiment..."

# Penguins Dataset
echo "=> Running Penguins..."
pixi run python src/jax_optim_env/tools/train.py --config-name sgd_exp -m \
    exp_name=$EXP_NAME dataset=pmlb_penguins epochs=$GENS eval_every=$EVAL_FREQ \
    model=$MODEL

echo "BOC Complete."

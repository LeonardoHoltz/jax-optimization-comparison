#!/bin/bash
# Multiple experiment runs with different seeds

GENS=200
EXP_NAME="SGD_adam_penguins_model_\${model._target_}" # '$' here will refer to what is in hydra
export HYDRA_FULL_ERROR=1
MODELS=${1:-"linear,mlp,mlp_2_layers_5,mlp_3_layers_5"}


echo "Starting seed multirun experiment..."

# Penguins Dataset
echo "=> Running trainings..."
pixi run python src/jax_optim_env/tools/train.py --config-name sgd_exp --multirun \
    exp_name=$EXP_NAME dataset=pmlb_penguins epochs=$GENS \
    model=$MODELS seed=40

echo "Seed multirun complete."

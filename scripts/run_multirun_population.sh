#!/bin/bash
# Multiple experiment runs with different population sizes

GENS=100
MODEL=mlp_deep
EXP_NAME="NE_simple_GA_penguins_pop_size_\${optimizer.population_size}" # '$' here will refer to what is in hydra
export HYDRA_FULL_ERROR=1
POP_SIZES=${1:-"50,100,200,300,400,500,1000"}


echo "Starting pop size multirun experiment..."

# Penguins Dataset
echo "=> Running trainings..."
pixi run python src/jax_optim_env/tools/train.py --config-name simple_ga_exp --multirun \
    exp_name=$EXP_NAME dataset=pmlb_penguins generations=$GENS \
    model=$MODEL optimizer.population_size=$POP_SIZES

echo "Pop size multirun complete."

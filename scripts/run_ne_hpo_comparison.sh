#!/bin/bash
# Compares each evosax neuroevolution optimizer's performance with and without
# hyperparameter optimization (see src/optim_env/tools/hpo.py) on the California
# Housing MLP task.
#
# For every optimizer in ES_OPTIMIZERS, runs:
#   (a) baseline    -- california_mlp_es_exp with default hyperparameters
#                      (population_size=64, init_std_multiplier=1.0, es_learning_rate=1e-3),
#                      exp_name jax_NE_<Algo>_california
#   (b) HPO-tuned   -- tools/hpo.py searches population_size / init_std_multiplier (and
#                      es_learning_rate, for algorithms with an internal optimizer), then
#                      retrains once at full generation budget with the best hyperparameters
#                      found, exp_name jax_NE_<Algo>_hpo_tabm_california
# then evaluates every resulting checkpoint on the held-out test set via tools/test.py,
# producing one CSV with baseline and HPO-tuned runs side by side.

GENERATIONS=${GENERATIONS:-300}
SEED=${SEED:-0}
N_TRIALS=${N_TRIALS:-20}
HPO_GENERATIONS=${HPO_GENERATIONS:-100}
POPULATION_SIZE_MAX=${POPULATION_SIZE_MAX:-96}
ES_OPTIMIZERS=${1:-"jax/sep_cma_es,jax/simple_ga,jax/particle_swarm,jax/open_es,jax/pgpe,jax/snes,jax/ars,jax/differential_evolution,jax/cr_fm_nes"}
export HYDRA_FULL_ERROR=1

echo "Starting neuroevolution HPO comparison (with vs. without hyperparameter optimization)..."

echo "=> Running baseline (no HPO) trainings: $ES_OPTIMIZERS"
pixi run python src/optim_env/tools/train.py --config-name california_mlp_es_exp --multirun \
    optimizer=$ES_OPTIMIZERS generations=$GENERATIONS seed=$SEED

IFS=',' read -ra OPTIMIZERS <<< "$ES_OPTIMIZERS"
for OPT in "${OPTIMIZERS[@]}"; do
    echo "=> Running HPO search for $OPT ($N_TRIALS trials, $HPO_GENERATIONS generations/trial)..."
    pixi run python src/optim_env/tools/hpo.py --optimizer "$OPT" \
        --n-trials "$N_TRIALS" --hpo-generations "$HPO_GENERATIONS" \
        --population-size-max "$POPULATION_SIZE_MAX" \
        --final-generations "$GENERATIONS" --seed "$SEED"
done

echo "=> Evaluating baseline and HPO-tuned checkpoints on the held-out test set..."
pixi run python src/optim_env/tools/test.py "outputs/jax_NE_*california*" \
    --csv-name california_ne_hpo_comparison.csv

echo "Neuroevolution HPO comparison complete. See outputs/tests/california_ne_hpo_comparison.csv"

#!/bin/bash
# Compares each evosax neuroevolution optimizer with and without hyperparameter
# optimization (see src/optim_env/tools/hpo.py) on the California Housing MLP task.
#
# For every optimizer in ES_OPTIMIZERS:
#   (a) baseline -- default hyperparameters (population_size=64, init_std_multiplier=1.0,
#       es_learning_rate=1e-3), replicated over SEEDS
#   (b) hpo      -- Optuna/TPE searches population_size, init_std_multiplier and (where the
#       algorithm has an internal optimizer) es_learning_rate, then retrains at full budget
#       with the best hyperparameters found
# Both land in outputs/runs/<dataset>/jax_NE/<Algorithm>/{baseline,hpo}/seed_<n>/, and the
# report lines them up side by side.
#
# The HPO stage is the expensive one: one training run per trial, per optimizer.
source "$(dirname "$0")/../lib/common.sh"

CAMPAIGN=${CAMPAIGN:-california_ne_hpo}
GENERATIONS=${GENERATIONS:-300}
SEEDS=${SEEDS:-"0,1,2"}
HPO_SEED=${HPO_SEED:-0}
N_TRIALS=${N_TRIALS:-20}
HPO_GENERATIONS=${HPO_GENERATIONS:-100}
POPULATION_SIZE_MAX=${POPULATION_SIZE_MAX:-96}
ES_OPTIMIZERS=${1:-"jax/sep_cma_es,jax/simple_ga,jax/particle_swarm,jax/open_es,jax/pgpe,jax/snes,jax/ars,jax/differential_evolution,jax/cr_fm_nes"}

section "Baseline (no HPO): $ES_OPTIMIZERS | seeds=$SEEDS"
$TRAIN --config-name california_mlp_es_exp --multirun \
    optimizer=$ES_OPTIMIZERS generations=$GENERATIONS seed=$SEEDS

IFS=',' read -ra OPTIMIZERS <<< "$ES_OPTIMIZERS"
for OPT in "${OPTIMIZERS[@]}"; do
    section "HPO search for $OPT ($N_TRIALS trials x $HPO_GENERATIONS generations)"
    $HPO --optimizer "$OPT" \
        --n-trials "$N_TRIALS" --hpo-generations "$HPO_GENERATIONS" \
        --population-size-max "$POPULATION_SIZE_MAX" \
        --final-generations "$GENERATIONS" --seed "$HPO_SEED" --final-seeds "$SEEDS"
done

section "Evaluating baseline and HPO-tuned checkpoints on the held-out test set"
$TEST 'outputs/runs/california/jax_NE/*/*/seed_*' --campaign "$CAMPAIGN"

write_manifest "$CAMPAIGN" "generations=$GENERATIONS" "seeds=$SEEDS" "hpo_seed=$HPO_SEED" \
    "n_trials=$N_TRIALS" "hpo_generations=$HPO_GENERATIONS" \
    "population_size_max=$POPULATION_SIZE_MAX" "es_optimizers=$ES_OPTIMIZERS"

echo "Complete. See outputs/reports/${CAMPAIGN}.csv and ${CAMPAIGN}_summary.csv"

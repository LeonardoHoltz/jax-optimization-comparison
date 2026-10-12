#!/bin/bash
# Every ES optimizer on the California MLP, untuned (default settings, over SEEDS) and
# tuned (Optuna search, then the best settings over SEEDS). The search is the slow part.
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
        --n-trials "$N_TRIALS" --hpo-budget "$HPO_GENERATIONS" \
        --population-size-max "$POPULATION_SIZE_MAX" \
        --final-budget "$GENERATIONS" --seed "$HPO_SEED" --final-seeds "$SEEDS"
done

section "Evaluating baseline and HPO-tuned checkpoints on the held-out test set"
$TEST 'outputs/runs/california/jax_NE/*/*/seed_*' --campaign "$CAMPAIGN"

write_manifest "$CAMPAIGN" "generations=$GENERATIONS" "seeds=$SEEDS" "hpo_seed=$HPO_SEED" \
    "n_trials=$N_TRIALS" "hpo_generations=$HPO_GENERATIONS" \
    "population_size_max=$POPULATION_SIZE_MAX" "es_optimizers=$ES_OPTIMIZERS"

echo "Complete. See outputs/reports/${CAMPAIGN}.csv and ${CAMPAIGN}_summary.csv"

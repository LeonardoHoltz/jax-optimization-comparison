#!/bin/bash
# Does tuning rescue the ES optimizers that scored near chance on MNIST (SNES, PGPE, PSO)?
# Pass a comma-separated optimizer list as $1 to tune others.
source "$(dirname "$0")/../lib/common.sh"

CAMPAIGN=${CAMPAIGN:-mnist_cnn_ne_hpo}
GENERATIONS=${GENERATIONS:-2000}
SEEDS=${SEEDS:-"0"}
HPO_SEED=${HPO_SEED:-0}
N_TRIALS=${N_TRIALS:-20}
HPO_GENERATIONS=${HPO_GENERATIONS:-200}
EVAL_EVERY_GEN=${EVAL_EVERY_GEN:-50}
FITNESS_BATCH=${FITNESS_BATCH:-256}
# Lower than the tabular campaign's 96: ES fitness is vmapped over the whole population, so
# the first conv layer's activations alone cost roughly pop x batch x 28 x 28 x 16 floats.
POPULATION_SIZE_MAX=${POPULATION_SIZE_MAX:-64}
ES_OPTIMIZERS=${1:-"jax/snes,jax/pgpe,jax/particle_swarm"}

# fitness_batch_size has to match the baseline runs or the tuned and untuned numbers are
# not measuring the same thing. eval_every_gen is finer than the baseline's 100 so that a
# short trial yields more than one validation reading to optimise against; it changes when
# the model is measured, not how it is trained.
HPO_SET=(--set "fitness_batch_size=$FITNESS_BATCH" --set "eval_every_gen=$EVAL_EVERY_GEN")

IFS=',' read -ra OPTIMIZERS <<< "$ES_OPTIMIZERS"
for OPT in "${OPTIMIZERS[@]}"; do
    section "HPO search for $OPT ($N_TRIALS trials x $HPO_GENERATIONS generations)"
    $HPO --config-name mnist_cnn_es_exp --dataset mnist --model jax/cnn_mnist \
        --optimizer "$OPT" \
        --n-trials "$N_TRIALS" --hpo-budget "$HPO_GENERATIONS" \
        --population-size-max "$POPULATION_SIZE_MAX" \
        --final-budget "$GENERATIONS" --seed "$HPO_SEED" --final-seeds "$SEEDS" \
        "${HPO_SET[@]}"
done

# Picks up the baseline runs left by mnist_cnn_optimizers alongside the new hpo variants,
# so the report lines each algorithm up tuned against untuned.
section "Evaluating baseline and HPO-tuned checkpoints on the held-out test set"
$TEST 'outputs/runs/mnist/jax_NE/*/*/seed_*' --campaign "$CAMPAIGN"

write_manifest "$CAMPAIGN" "generations=$GENERATIONS" "seeds=$SEEDS" "hpo_seed=$HPO_SEED" \
    "n_trials=$N_TRIALS" "hpo_generations=$HPO_GENERATIONS" \
    "eval_every_gen=$EVAL_EVERY_GEN" "fitness_batch=$FITNESS_BATCH" \
    "population_size_max=$POPULATION_SIZE_MAX" "es_optimizers=$ES_OPTIMIZERS"

echo "Complete. See outputs/reports/${CAMPAIGN}.csv and ${CAMPAIGN}_summary.csv"

#!/bin/bash
# Image-track counterpart to california_ne_hpo.sh: does hyperparameter tuning rescue the
# neuroevolution optimizers that collapsed on MNIST?
#
# The baseline mnist_cnn_optimizers campaign left three algorithms at or near chance
# accuracy (SNES 0.152, PGPE 0.184, PSO 0.203, against 0.10 for random guessing) even
# though all three were stable on the tabular California task. Two explanations fit that
# evidence equally well: the default hyperparameters happen to be wrong for a 20K-parameter
# CNN, or these algorithms genuinely do not cope with this search landscape. This campaign
# separates them -- if a TPE search over population size, initial std and (where it
# applies) the internal learning rate still cannot beat chance, the failure is the
# algorithm rather than the defaults.
#
# Defaults to those three optimizers because that is the open question and the search is
# the expensive stage (one training run per trial). Pass a comma-separated list as $1 to
# tune others, e.g. the full set used by the baseline campaign:
#   scripts/campaigns/mnist_cnn_ne_hpo.sh "jax/sep_cma_es,jax/snes,jax/pgpe,jax/open_es,jax/ars,jax/particle_swarm,jax/simple_ga,jax/differential_evolution,jax/cr_fm_nes"
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
        --n-trials "$N_TRIALS" --hpo-generations "$HPO_GENERATIONS" \
        --population-size-max "$POPULATION_SIZE_MAX" \
        --final-generations "$GENERATIONS" --seed "$HPO_SEED" \
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

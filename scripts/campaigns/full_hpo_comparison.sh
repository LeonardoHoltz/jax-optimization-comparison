#!/bin/bash
# Fair tuned-vs-untuned comparison across both tracks.
#
# Every optimizer gets the same treatment on each track: a baseline run on shared defaults,
# then an Optuna/TPE search, then a confirmation run at the full budget using the best
# hyperparameters found. Earlier campaigns tuned only a subset, which made the reports mix
# tuned and untuned numbers in the same column -- this one does not.
#
# Two things differ from the earlier per-track scripts:
#
#   1. Trials run at a large fraction of the final budget (all of it on tabular, half on
#      image) instead of a tenth. Scoring a trial at a short budget rewards optimizers that
#      plateau early and penalises ones that improve slowly, which is backwards for picking
#      an optimizer you intend to run long. PSO beat SNES at 200 generations on MNIST and
#      lost to it threefold at 2000.
#   2. That is affordable because tools/hpo.py now streams each trial's intermediate
#      validation loss to Optuna, so MedianPruner kills trials that fall behind instead of
#      paying for their full budget.
#
# Stale California runs from the hpo.py smoke tests (1-2 generations, but shaped exactly
# like real results) are moved aside first, since test.py globs the runs tree and would
# otherwise aggregate them into the report.
#
# Resumable: comment out a stage you have already completed. Expect several hours; run it
# detached.
#   setsid nohup ./scripts/campaigns/full_hpo_comparison.sh > logs/full_hpo.log 2>&1 < /dev/null &
source "$(dirname "$0")/../lib/common.sh"

CAMPAIGN=${CAMPAIGN:-full_hpo_comparison}
SEEDS=${SEEDS:-"0,1,2"}
HPO_SEED=${HPO_SEED:-0}
N_TRIALS=${N_TRIALS:-20}
ES_OPTIMIZERS=${ES_OPTIMIZERS:-"jax/sep_cma_es,jax/snes,jax/pgpe,jax/open_es,jax/ars,jax/particle_swarm,jax/simple_ga,jax/differential_evolution,jax/cr_fm_nes"}

# --- tabular ----------------------------------------------------------------------------
CAL_GENERATIONS=${CAL_GENERATIONS:-300}
CAL_HPO_GENERATIONS=${CAL_HPO_GENERATIONS:-300}   # full budget: trials measure what the final run measures
CAL_POP_MAX=${CAL_POP_MAX:-96}

# --- image ------------------------------------------------------------------------------
MNIST_GENERATIONS=${MNIST_GENERATIONS:-2000}
MNIST_HPO_GENERATIONS=${MNIST_HPO_GENERATIONS:-1000}
MNIST_EVAL_EVERY=${MNIST_EVAL_EVERY:-50}
MNIST_FITNESS_BATCH=${MNIST_FITNESS_BATCH:-256}
# Below the tabular cap: ES fitness is vmapped over the whole population, so the first conv
# layer's activations alone cost pop x batch x 28 x 28 x 16 floats.
MNIST_POP_MAX=${MNIST_POP_MAX:-64}

IFS=',' read -ra OPTIMIZERS <<< "$ES_OPTIMIZERS"

section "Quarantining stale California runs (smoke-test leftovers, 1-2 generations)"
TRASH="outputs/.trash/$(date +%Y%m%d-%H%M%S)"
for d in outputs/runs/california/jax_NE/*/baseline outputs/runs/california/jax_NE/*/hpo; do
    [ -d "$d" ] || continue
    mkdir -p "$TRASH/$(dirname "$d")"
    mv "$d" "$TRASH/$d"
    echo "  moved $d -> $TRASH/$d"
done
for f in outputs/reports/california_ne_hpo*; do
    [ -e "$f" ] || continue
    mkdir -p "$TRASH/outputs/reports"
    mv "$f" "$TRASH/outputs/reports/" && echo "  moved $f"
done
echo "  (nothing deleted; restore with: cp -r $TRASH/outputs/* outputs/)"

# =========================== TABULAR =====================================================
section "TABULAR baseline: $ES_OPTIMIZERS | generations=$CAL_GENERATIONS | seeds=$SEEDS"
$TRAIN --config-name california_mlp_es_exp --multirun \
    optimizer=$ES_OPTIMIZERS generations=$CAL_GENERATIONS seed=$SEEDS

for OPT in "${OPTIMIZERS[@]}"; do
    section "TABULAR HPO: $OPT ($N_TRIALS trials x $CAL_HPO_GENERATIONS generations)"
    $HPO --config-name california_mlp_es_exp --dataset tabm_california --model jax/mlp_california \
        --optimizer "$OPT" --n-trials "$N_TRIALS" \
        --hpo-generations "$CAL_HPO_GENERATIONS" --final-generations "$CAL_GENERATIONS" \
        --population-size-max "$CAL_POP_MAX" --seed "$HPO_SEED"
done

# =========================== IMAGE =======================================================
# Baselines already exist at 2000 generations from mnist_cnn_optimizers.sh and are valid;
# re-run them only if you want more seeds.
if [ "${RERUN_MNIST_BASELINE:-0}" = "1" ]; then
    section "IMAGE baseline: $ES_OPTIMIZERS | generations=$MNIST_GENERATIONS | seeds=$SEEDS"
    $TRAIN --config-name mnist_cnn_es_exp --multirun \
        optimizer=$ES_OPTIMIZERS generations=$MNIST_GENERATIONS \
        eval_every_gen=$MNIST_EVAL_EVERY fitness_batch_size=$MNIST_FITNESS_BATCH seed=$SEEDS
fi

for OPT in "${OPTIMIZERS[@]}"; do
    section "IMAGE HPO: $OPT ($N_TRIALS trials x $MNIST_HPO_GENERATIONS generations)"
    $HPO --config-name mnist_cnn_es_exp --dataset mnist --model jax/cnn_mnist \
        --optimizer "$OPT" --n-trials "$N_TRIALS" \
        --hpo-generations "$MNIST_HPO_GENERATIONS" --final-generations "$MNIST_GENERATIONS" \
        --population-size-max "$MNIST_POP_MAX" --seed "$HPO_SEED" \
        --set "fitness_batch_size=$MNIST_FITNESS_BATCH" --set "eval_every_gen=$MNIST_EVAL_EVERY"
done

# =========================== REPORTS =====================================================
section "Evaluating every checkpoint on the held-out test sets"
$TEST 'outputs/runs/california/jax_NE/*/*/seed_*' --campaign "${CAMPAIGN}_tabular"
$TEST 'outputs/runs/mnist/jax_NE/*/*/seed_*'      --campaign "${CAMPAIGN}_image"

write_manifest "$CAMPAIGN" "seeds=$SEEDS" "hpo_seed=$HPO_SEED" "n_trials=$N_TRIALS" \
    "cal_generations=$CAL_GENERATIONS" "cal_hpo_generations=$CAL_HPO_GENERATIONS" \
    "mnist_generations=$MNIST_GENERATIONS" "mnist_hpo_generations=$MNIST_HPO_GENERATIONS" \
    "mnist_fitness_batch=$MNIST_FITNESS_BATCH" "es_optimizers=$ES_OPTIMIZERS"

echo "Complete. See outputs/reports/${CAMPAIGN}_tabular_summary.csv and ${CAMPAIGN}_image_summary.csv"

#!/bin/bash
# DUST round of the reproduction study: untuned (no HPO), 5 seeds, on the tabular and
# image tasks; text tasks left out for now. Resumable: finished runs are skipped. Then
# re-scores every task with DUST runs, rebuilds the XLSX and draws the interpretability
# grids (tuned runs plus untuned DUST, seed 0) into outputs/reports/reproduction/interpretability/.
set -e
cd "$(dirname "$0")/../.."
TASKS=${TASKS:-"california adult higgs_small churn house mnist fashion_mnist"}

# --tasks re-scores each task at its end, so California gets its DUST rows too.
pixi run python scripts/campaigns/reproduction_study.py --stage ne --optimizers jax/dust --no-hpo --tasks $TASKS

OUT=outputs/reports/reproduction/interpretability
for d in mnist fashion_mnist; do
    for a in paper simple; do
        for m in features saliency; do
            pixi run python src/optim_env/tools/interpretability_grid.py --method $m --out $OUT/${m}_${d}_${a}.png \
                "outputs/runs/$d/jax_*/*/${a}_hpo/seed_0" "outputs/runs/$d/jax_NE/DUST/${a}_baseline/seed_0"
        done
    done
done
echo "DUST round done"

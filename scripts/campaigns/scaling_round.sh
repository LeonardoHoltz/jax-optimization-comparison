#!/bin/bash
# Scaling study on MNIST (src/optim_env/configs/study/scaling_mnist.yaml) over the GPUs you
# list. Each GPU takes a group of widths and runs tune, sweep and AdamW for both algorithms
# on them. The groups hold about equal GPU time, so 1, 2 or 4 GPUs share the work evenly.
# Then fits and plots go to outputs/reports/scaling_mnist/.
#   setsid nohup scripts/campaigns/scaling_round.sh 0 1 2 3 > scaling_round.log 2>&1 < /dev/null &
# Arguments: GPU indices (default 0). Logs: scaling_gpu<index>.log.
# Progress: outputs/study/scaling_mnist/progress.md, rewritten at each run's start and end.
# Resume: launch the same command again. Finished runs (metrics.json) are skipped; a run cut
# off midway starts over. Stop: kill -- -$(cat outputs/study/scaling_mnist/round.pgid)
cd "$(dirname "$0")/../.."
# JAX would reserve 75% of the GPU up front; allocate on demand (shared GPUs).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
S="pixi run python scripts/campaigns/scaling_study.py"
GPUS=("${@:-0}")
GROUPS_=("16" "32 1" "64 2" "8 4")  # ~15 GPU h each on an RTX 3070 (cost probe)
mkdir -p outputs/study/scaling_mnist
ps -o pgid= $$ | tr -d ' ' > outputs/study/scaling_mnist/round.pgid
trap 'kill 0' INT TERM  # Ctrl-C or kill also stops the training runs

for i in "${!GPUS[@]}"; do
    widths=""
    for ((g = i; g < ${#GROUPS_[@]}; g += ${#GPUS[@]})); do widths="$widths ${GROUPS_[g]}"; done
    [ -z "$widths" ] && continue  # more GPUs than groups
    (export CUDA_VISIBLE_DEVICES=${GPUS[i]}
     $S tune --widths $widths && $S sweep --widths $widths && $S adamw --widths $widths
    ) > scaling_gpu${GPUS[i]}.log 2>&1 &
done
wait
$S status
pixi run python src/optim_env/tools/scaling_analysis.py
echo "scaling round done"

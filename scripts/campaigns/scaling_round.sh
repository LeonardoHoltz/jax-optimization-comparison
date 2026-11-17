#!/bin/bash
# Scaling study on MNIST (src/optim_env/configs/study/scaling_mnist.yaml) on two GPUs:
# DUST on the first; Open-ES and the AdamW reference on the second. Then fits and plots go
# to outputs/reports/scaling_mnist/.
#   setsid nohup scripts/campaigns/scaling_round.sh 0 1 > scaling_round.log 2>&1 < /dev/null &
# Arguments: GPU indices (default 0 1). One index runs everything on that GPU, one job at a time.
# Progress: outputs/study/scaling_mnist/progress.md, rewritten at each run's start and end.
# Resume: launch the same command again. Finished runs (metrics.json) are skipped; a run cut
# off midway starts over. Stop: kill -- -$(cat outputs/study/scaling_mnist/round.pgid)
cd "$(dirname "$0")/../.."
# JAX would reserve 75% of the GPU up front; allocate on demand (shared GPUs).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
S="pixi run python scripts/campaigns/scaling_study.py"
GPU_A=${1:-0}
GPU_B=${2:-$GPU_A}
mkdir -p outputs/study/scaling_mnist
ps -o pgid= $$ | tr -d ' ' > outputs/study/scaling_mnist/round.pgid
trap 'kill 0' INT TERM  # Ctrl-C or kill also stops the training runs

if [ "$GPU_A" = "$GPU_B" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_A
    { $S tune && $S sweep && $S adamw; } > scaling_gpu$GPU_A.log 2>&1
else
    (export CUDA_VISIBLE_DEVICES=$GPU_A; $S tune --algorithms dust && $S sweep --algorithms dust) > scaling_gpu$GPU_A.log 2>&1 &
    (export CUDA_VISIBLE_DEVICES=$GPU_B; $S tune --algorithms open_es && $S sweep --algorithms open_es && $S adamw) > scaling_gpu$GPU_B.log 2>&1 &
    wait
fi
$S status
pixi run python src/optim_env/tools/scaling_analysis.py
echo "scaling round done"

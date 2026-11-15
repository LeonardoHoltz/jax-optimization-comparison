#!/bin/bash
# Scaling study on MNIST (src/optim_env/configs/study/scaling_mnist.yaml), split over two
# GPUs: DUST on GPU 0; Open-ES and the AdamW reference on GPU 1. Resumable: finished runs
# are skipped. Then fits and plots go to outputs/reports/scaling_mnist/.
#   setsid nohup scripts/campaigns/scaling_round.sh > scaling_round.log 2>&1 < /dev/null &
cd "$(dirname "$0")/../.."
# JAX would reserve 75% of the GPU up front; allocate on demand (shared GPUs).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
S="pixi run python scripts/campaigns/scaling_study.py"

(export CUDA_VISIBLE_DEVICES=0; $S tune --algorithms dust && $S sweep --algorithms dust) > scaling_gpu0.log 2>&1 &
(export CUDA_VISIBLE_DEVICES=1; $S tune --algorithms open_es && $S sweep --algorithms open_es && $S adamw) > scaling_gpu1.log 2>&1 &
wait
pixi run python src/optim_env/tools/scaling_analysis.py
echo "scaling round done"

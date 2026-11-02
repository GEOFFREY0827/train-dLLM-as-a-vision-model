#!/usr/bin/env bash
# Kernel-shape sweep for the 2D token conv. Results go to results.csv,
# samples to ckpt/*_samples.txt.
#
#   off  : no conv (baseline)
#   1x1  : same extra params, no spatial mixing (control)
#   3x3  : square in cells -> tall in physical space
#   3x5  : ~square in physical space (cells are ~2x taller than wide)
#   5x5  : bigger square in cells
#   1x5  : horizontal only
#   3x1  : vertical only
#
# Usage: bash sweep_kernel.sh [source] [style] [steps] [seeds...]
#   bash sweep_kernel.sh                       # mnist mixed 10000 0
#   bash sweep_kernel.sh shapes edge 2000 0    # quick check
#   bash sweep_kernel.sh mnist mixed 10000 0 1 2
set -e
SOURCE=${1:-mnist}
STYLE=${2:-mixed}
STEPS=${3:-10000}
shift $(( $# < 3 ? $# : 3 ))
SEEDS=${@:-0}

for SEED in $SEEDS; do
  python train_dllm.py --source $SOURCE --style $STYLE --steps $STEPS --seed $SEED
  for K in 1x1 3x3 3x5 5x5 1x5 3x1; do
    python train_dllm.py --source $SOURCE --style $STYLE --steps $STEPS --seed $SEED --conv --kernel $K
  done
done

echo
echo "=== results ==="
column -s, -t < results.csv

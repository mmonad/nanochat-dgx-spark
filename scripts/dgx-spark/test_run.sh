#!/bin/bash
# Quick smoke test: single-GPU, minimal depth, no checkpoints/evals.
# Usage: ./scripts/dgx-spark/test_run.sh

set -e
source "$(dirname "$0")/env.sh"

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=4 \
    --run="test" \
    --model-tag="test" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1

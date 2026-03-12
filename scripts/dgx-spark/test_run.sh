#!/bin/bash
# Quick smoke test: minimal depth, no checkpoints/evals.
# Supports single-node (default) or cluster mode.
#
# Usage:
#   ./test_run.sh                        # single GPU
#   ./test_run.sh --cluster              # auto-discover peers
#   ./test_run.sh --cluster 10.0.0.2     # explicit peer

set -e
source "$(dirname "$0")/env.sh"

CLUSTER=false
PEERS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster|-c) CLUSTER=true; shift ;;
        -p|--port) CLUSTER_PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--cluster] [PEER_IP ...]"
            echo "  --cluster, -c  Run distributed across discovered/specified peers"
            exit 0 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) PEERS+=("$1"); CLUSTER=true; shift ;;
    esac
done

TRAIN_ARGS="--depth=4 --run=test --model-tag=test --core-metric-every=999999 --sample-every=-1 --save-every=-1"

if [ "$CLUSTER" = true ]; then
    source "$(dirname "$0")/cluster.sh"
    [ ${#PEERS[@]} -eq 0 ] && discover_peers
    run_distributed "$TRAIN_ARGS"
else
    OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- $TRAIN_ARGS
fi

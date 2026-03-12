#!/bin/bash
# Pretraining for DGX Spark. Single-node by default, cluster with -c.
#
# Usage:
#   ./pretrain.sh                              # single GPU, depth=20
#   ./pretrain.sh -d 24 -b 16                  # custom depth + batch
#   ./pretrain.sh -c                            # cluster, auto-discover peers
#   ./pretrain.sh -c 169.254.129.198            # cluster, explicit peer
#   ./pretrain.sh --fp8                          # single GPU, FP8 training
#   ./pretrain.sh --fp4                          # single GPU, NVFP4 training

set -e
source "$(dirname "$0")/env.sh"

CLUSTER=false
DEPTH=20
BATCH_SIZE=32
FP8=false
FP4=false
PEERS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--cluster) CLUSTER=true; shift ;;
        -d|--depth) DEPTH="$2"; shift 2 ;;
        -b|--batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --fp8) FP8=true; shift ;;
        --fp4) FP4=true; shift ;;
        -p|--port) CLUSTER_PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [options] [PEER_IP ...]"
            echo "  -c, --cluster     Run distributed across peers"
            echo "  -d, --depth       Model depth (default: 20)"
            echo "  -b, --batch-size  Device batch size (default: 32)"
            echo "  --fp8             Enable FP8 training"
            echo "  --fp4             Enable NVFP4 training (Blackwell only)"
            echo "  -p, --port        Master port (default: 29500, cluster only)"
            exit 0 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) PEERS+=("$1"); CLUSTER=true; shift ;;
    esac
done

[ "$FP8" = true ] && [ "$FP4" = true ] && { echo "Error: --fp8 and --fp4 are mutually exclusive"; exit 1; }

TRAIN_ARGS="--depth=$DEPTH --device-batch-size=$BATCH_SIZE --sample-every=100 --save-every=1000 --run=nanochat-pretrain"
[ "$FP8" = true ] && TRAIN_ARGS="$TRAIN_ARGS --fp8"
[ "$FP4" = true ] && TRAIN_ARGS="$TRAIN_ARGS --fp4"

if [ "$CLUSTER" = true ]; then
    source "$(dirname "$0")/cluster.sh"
    [ ${#PEERS[@]} -eq 0 ] && discover_peers
    NNODES=$(( ${#PEERS[@]} + 1 ))
    CLUSTER_ARGS="--depth=$DEPTH --device-batch-size=$BATCH_SIZE --sample-every=100 --save-every=1000 --run=nanochat-${NNODES}spark-pretrain"
    [ "$FP8" = true ] && CLUSTER_ARGS="$CLUSTER_ARGS --fp8"
    [ "$FP4" = true ] && CLUSTER_ARGS="$CLUSTER_ARGS --fp4"
    run_distributed "$CLUSTER_ARGS"
else
    OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- $TRAIN_ARGS
fi

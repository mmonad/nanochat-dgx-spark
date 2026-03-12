#!/bin/bash
# Cluster utilities for distributed training on DGX Sparks.
# Source this after env.sh. Provides: discover_peers, sync_peers, run_distributed.

CLUSTER_PORT=${CLUSTER_PORT:-29500}
PEERS=("${PEERS[@]}")  # preserve any PEERS already set by caller

# Detect InfiniBand interface and local IP
detect_network() {
    IB_IF=$(ibdev2netdev 2>/dev/null | awk '/(Up|ACTIVE)/{print $5; exit}')
    if [ -z "$IB_IF" ]; then
        IB_IF=$(ip route | awk '/default/{print $5; exit}')
    fi
    MASTER_IP=$(ip -o -4 addr show dev "$IB_IF" | awk '{print $4}' | cut -d/ -f1)
    [ -n "$MASTER_IP" ] || { echo "Cannot determine local IP on $IB_IF"; exit 1; }
}

# Auto-discover peer Sparks via avahi-browse on IB interface
discover_peers() {
    detect_network
    echo "Discovering Sparks on $IB_IF..."
    if ! command -v avahi-browse &>/dev/null; then
        echo "avahi-browse not found. Provide peer IPs as arguments."
        exit 1
    fi
    local all_ips
    all_ips=$(avahi-browse -p -r -f -t _ssh._tcp 2>/dev/null \
        | grep "$IB_IF" | grep "^=" | grep "IPv4" \
        | awk -F';' '{print $8}' | sort -u)
    for ip in $all_ips; do
        [ "$ip" != "$MASTER_IP" ] && PEERS+=("$ip")
    done
    if [ ${#PEERS[@]} -eq 0 ]; then
        echo "No peers discovered. Check InfiniBand or provide IPs manually."
        exit 1
    fi
}

# Rsync repo + training data to all peers
sync_peers() {
    local repo_dir="$(pwd)"
    local cache_dir="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
    echo "Syncing to ${#PEERS[@]} peer(s)..."
    for peer in "${PEERS[@]}"; do
        echo "  $peer: repo..."
        rsync -a --delete \
            --exclude '.venv' --exclude '__pycache__' --exclude 'wandb/' \
            "$repo_dir/" "$USER@$peer:$repo_dir/"
        echo "  $peer: data..."
        ssh -o StrictHostKeyChecking=no "$USER@$peer" "mkdir -p '$cache_dir'"
        rsync -a \
            --include 'base_data_climbmix/***' \
            --include 'eval_bundle/***' \
            --include 'tokenizer/***' \
            --exclude '*' \
            "$cache_dir/" "$USER@$peer:$cache_dir/"
        echo "  $peer: venv + triton cache..."
        ssh -o StrictHostKeyChecking=no "$USER@$peer" bash -l -c \
            "'mkdir -p ~/.triton/cache && chmod -R u+w ~/.triton/cache && cd \"$repo_dir\" && uv sync'"
    done
}

# Launch distributed training: run_distributed "torchrun training args"
run_distributed() {
    local train_args="$1"
    local nnodes=$(( ${#PEERS[@]} + 1 ))
    local repo_dir="$(pwd)"

    detect_network
    sync_peers

    echo "Master:  $MASTER_IP ($IB_IF)"
    echo "Peers:   ${PEERS[*]}"
    echo "Nodes:   $nnodes"

    # Cleanup: kill local SSH children + remote torchrun on any exit
    WORKER_PIDS=()
    _MASTER_EXIT_CODE=0
    _cluster_cleanup() {
        trap - INT TERM EXIT
        echo "Shutting down..."
        for pid in "${WORKER_PIDS[@]}"; do kill "$pid" 2>/dev/null; done
        for ip in "${PEERS[@]}"; do
            ssh -o ConnectTimeout=5 "$USER@$ip" "pkill -f 'torchrun.*base_train'" 2>/dev/null &
        done
        wait
        exit "${_MASTER_EXIT_CODE}"
    }
    trap '_MASTER_EXIT_CODE=130; _cluster_cleanup' INT TERM
    trap '_cluster_cleanup' EXIT

    # Launch workers
    for i in "${!PEERS[@]}"; do
        local rank=$(( i + 1 ))
        local peer="${PEERS[$i]}"
        echo "Launching rank $rank on $peer..."
        ssh -o StrictHostKeyChecking=no "$USER@$peer" bash -l <<REMOTE &
cd "$repo_dir"
source scripts/dgx-spark/env.sh
export NCCL_SOCKET_IFNAME=$IB_IF
OMP_NUM_THREADS=1 torchrun \\
    --nproc_per_node=1 \\
    --nnodes=$nnodes \\
    --node_rank=$rank \\
    --master_addr=$MASTER_IP \\
    --master_port=$CLUSTER_PORT \\
    -m scripts.base_train -- $train_args
REMOTE
        WORKER_PIDS+=($!)
    done

    # Run master locally — capture exit code for cleanup
    echo "Starting master (rank 0)..."
    export NCCL_SOCKET_IFNAME=$IB_IF
    OMP_NUM_THREADS=1 torchrun \
        --nproc_per_node=1 \
        --nnodes=$nnodes \
        --node_rank=0 \
        --master_addr="$MASTER_IP" \
        --master_port="$CLUSTER_PORT" \
        -m scripts.base_train -- $train_args || _MASTER_EXIT_CODE=$?
}

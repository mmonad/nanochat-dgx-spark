#!/bin/bash
# Environment setup for DGX Spark (Blackwell GB10).
# Usage: source scripts/dgx-spark/env.sh

# Navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

# CUDA 13.0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-$CUDA_HOME/bin/ptxas}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Memory optimization for unified memory architecture
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-max_split_size_mb:512}"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "No .venv found. Run: uv sync --extra gpu"
    return 1 2>/dev/null || exit 1
fi

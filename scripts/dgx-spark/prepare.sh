#!/bin/bash
# One-shot data preparation for DGX Spark: download data, train tokenizer, fetch eval bundle.
# Usage:
#   ./prepare.sh              # Full setup (240 shards)
#   ./prepare.sh -n 170       # Fewer shards (enough for GPT-2)

set -e
source "$(dirname "$0")/env.sh"

NUM_SHARDS=240

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-shards)
            [[ -n "$2" && "$2" =~ ^[0-9]+$ ]] || { echo "Error: -n requires a numeric argument"; exit 1; }
            NUM_SHARDS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [-n NUM_SHARDS]"
            echo "  -n, --num-shards  Number of data shards to download (default: 240)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Download dataset
python -m nanochat.dataset -n "$NUM_SHARDS"

# Train tokenizer
python -m scripts.tok_train
python -m scripts.tok_eval

# Download eval bundle (use same base dir as Python code)
BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
EVAL_DIR="$BASE_DIR/eval_bundle"
if [ ! -d "$EVAL_DIR" ]; then
    TMPFILE=$(mktemp /tmp/eval_bundle_XXXXXX.zip)
    curl -L -o "$TMPFILE" https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
    unzip -q "$TMPFILE" -d "$BASE_DIR"
    rm "$TMPFILE"
fi

echo "Done. To train: source scripts/dgx-spark/env.sh && torchrun --standalone --nproc_per_node=1 -m scripts.base_train"

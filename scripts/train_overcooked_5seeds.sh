#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -gt 0 ]]; then
  ALG="$1"
  shift
else
  ALG="ippo"
fi

case "$ALG" in
  ippo)
    TRAIN_SCRIPT="baselines/IPPO/ippo_rnn_overcooked_v2.py"
    ;;
  mappo_v2)
    TRAIN_SCRIPT="baselines/TARL/mappo_rnn_overcooked_v2_v2.py"
    ;;
  *)
    echo "Unsupported algorithm: $ALG" >&2
    echo "Usage: $0 [ippo|mappo_v2] [HYDRA_OVERRIDES...]" >&2
    exit 1
    ;;
esac

export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"

python "$TRAIN_SCRIPT" \
  WANDB_MODE="${WANDB_MODE:-online}" \
  SEED="${SEED:-0}" \
  NUM_SEEDS="${NUM_SEEDS:-5}" \
  "$@"

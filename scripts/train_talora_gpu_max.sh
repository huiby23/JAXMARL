#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="$SCRIPT_DIR/train_overcooked_v3_5seeds.sh"

# Throughput-oriented defaults. Any key can still be overridden from CLI.
exec "$BASE_SCRIPT" talora \
  WANDB_MODE="${WANDB_MODE:-disabled}" \
  NUM_SEEDS="${NUM_SEEDS:-5}" \
  NUM_ENVS="${NUM_ENVS:-512}" \
  NUM_STEPS="${NUM_STEPS:-128}" \
  NUM_MINIBATCHES="${NUM_MINIBATCHES:-8}" \
  UPDATE_EPOCHS="${UPDATE_EPOCHS:-2}" \
  METRIC_LOG_INTERVAL_UPDATES="${METRIC_LOG_INTERVAL_UPDATES:-10}" \
  CHECKPOINT_INTERVAL_UPDATES="${CHECKPOINT_INTERVAL_UPDATES:-50}" \
  "$@"

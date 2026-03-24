#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ALG="mappo"
if [[ $# -gt 0 ]]; then
  case "$1" in
    ippo|mappo|mappo_lora_tta|both)
      ALG="$1"
      shift
      ;;
  esac
fi

# Default W&B destination. You can still override via env or CLI,
# e.g. PROJECT=xxx ENTITY=yyy scripts/train_mappo_v3_5s.sh
DEFAULT_PROJECT="2023-0323-Overcooked-MARL"
DEFAULT_ENTITY="huiby_tsinghua23"

get_override() {
  local key="$1"
  shift
  local arg
  local value=""
  for arg in "$@"; do
    if [[ "$arg" == "$key="* ]]; then
      value="${arg#*=}"
    fi
  done
  printf '%s' "$value"
}

EFFECTIVE_WANDB_MODE="$(get_override WANDB_MODE "$@")"
if [[ -z "$EFFECTIVE_WANDB_MODE" ]]; then
  EFFECTIVE_WANDB_MODE="${WANDB_MODE:-online}"
fi

EFFECTIVE_PROJECT="$(get_override PROJECT "$@")"
if [[ -z "$EFFECTIVE_PROJECT" ]]; then
  EFFECTIVE_PROJECT="${PROJECT:-$DEFAULT_PROJECT}"
fi

EFFECTIVE_ENTITY="$(get_override ENTITY "$@")"
if [[ -z "$EFFECTIVE_ENTITY" ]]; then
  EFFECTIVE_ENTITY="${ENTITY:-$DEFAULT_ENTITY}"
fi

if [[ "$EFFECTIVE_WANDB_MODE" == "online" && -z "$EFFECTIVE_PROJECT" ]]; then
  echo "PROJECT is empty. Set PROJECT=<wandb_project> (or export PROJECT) for online logging." >&2
  exit 1
fi

EFFECTIVE_LAYOUT="$(get_override ENV_KWARGS.layout "$@")"
if [[ -z "$EFFECTIVE_LAYOUT" ]]; then
  EFFECTIVE_LAYOUT="${LAYOUT:-forced_coord}"
fi

case "$ALG" in
  ippo)
    GROUP_ALGO_TOKEN="ippo"
    ;;
  mappo)
    GROUP_ALGO_TOKEN="mappo"
    ;;
  mappo_lora_tta)
    GROUP_ALGO_TOKEN="mappo_lora_tta"
    ;;
  both)
    GROUP_ALGO_TOKEN="ippo_mappo"
    ;;
  *)
    GROUP_ALGO_TOKEN="$ALG"
    ;;
esac

GROUP_ALGO_TOKEN="${GROUP_ALGO_TOKEN//\//_}"
GROUP_LAYOUT_TOKEN="${EFFECTIVE_LAYOUT//\//_}"
GROUP_LAYOUT_TOKEN="${GROUP_LAYOUT_TOKEN// /_}"
WANDB_GROUP_DEFAULT="${GROUP_ALGO_TOKEN}_${GROUP_LAYOUT_TOKEN}_$(date +%Y%m%d_%H%M%S)"

EFFECTIVE_WANDB_GROUP="$(get_override WANDB_GROUP "$@")"
if [[ -z "$EFFECTIVE_WANDB_GROUP" ]]; then
  EFFECTIVE_WANDB_GROUP="${WANDB_GROUP:-$WANDB_GROUP_DEFAULT}"
fi

EFFECTIVE_NUM_SEEDS="$(get_override NUM_SEEDS "$@")"
if [[ -z "$EFFECTIVE_NUM_SEEDS" ]]; then
  EFFECTIVE_NUM_SEEDS="${NUM_SEEDS:-5}"
fi

XLA_FLAGS_DEFAULT="--xla_gpu_deterministic_ops=true"

COMMON_OVERRIDES=(
  "PROJECT=$EFFECTIVE_PROJECT"
  "ENTITY=$EFFECTIVE_ENTITY"
  "WANDB_MODE=$EFFECTIVE_WANDB_MODE"
  "WANDB_GROUP=$EFFECTIVE_WANDB_GROUP"
  "SEED=${SEED:-0}"
  "NUM_SEEDS=$EFFECTIVE_NUM_SEEDS"
  "TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-3e7}"
  "NUM_STEPS=${NUM_STEPS:-256}"
  "NUM_ENVS=${NUM_ENVS:-256}"
  "NUM_MINIBATCHES=${NUM_MINIBATCHES:-64}"
  "UPDATE_EPOCHS=${UPDATE_EPOCHS:-4}"
  "ENV_KWARGS.layout=$EFFECTIVE_LAYOUT"
)

USER_OVERRIDES=()
for arg in "$@"; do
  case "$arg" in
    PROJECT=*|ENTITY=*|WANDB_MODE=*|WANDB_GROUP=*|NUM_SEEDS=*|ENV_KWARGS.layout=*)
      ;;
    *)
      USER_OVERRIDES+=("$arg")
      ;;
  esac
done

run_cmd() {
  echo
  echo ">>> $*"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  "$@"
}

run_ippo() {
  export XLA_FLAGS="${XLA_FLAGS:-$XLA_FLAGS_DEFAULT}"
  run_cmd python baselines/IPPO/ippo_rnn_overcooked_v2_v3.py \
    "WANDB_ALGO_LABEL=${IPPO_ALGO_LABEL:-IPPO}" \
    "SAVE_PATH=${IPPO_SAVE_PATH:-check_points/ippo_rnn_overcooked_v2_v3}" \
    "${COMMON_OVERRIDES[@]}" \
    "${USER_OVERRIDES[@]}"
}

run_mappo() {
  export XLA_FLAGS="${XLA_FLAGS:-$XLA_FLAGS_DEFAULT}"
  run_cmd python baselines/TARL/mappo_rnn_overcooked_v2_v3.py \
    "WANDB_ALGO_LABEL=${MAPPO_ALGO_LABEL:-MAPPO}" \
    "WORLD_STATE_SOURCE=${WORLD_STATE_SOURCE:-obs_concat}" \
    "SAVE_PATH=${MAPPO_SAVE_PATH:-check_points/mappo_rnn_overcooked_v2_v3}" \
    "TENSORBOARD_ENABLED=${TENSORBOARD_ENABLED:-False}" \
    "${COMMON_OVERRIDES[@]}" \
    "${USER_OVERRIDES[@]}"
}

run_mappo_lora_tta() {
  export XLA_FLAGS="${XLA_FLAGS:-$XLA_FLAGS_DEFAULT}"
  run_cmd python baselines/TARL/mappo_rnn_overcooked_v2_lora_tta.py \
    "WANDB_ALGO_LABEL=${MAPPO_LORA_ALGO_LABEL:-MAPPO_LORA_TTA}" \
    "WORLD_STATE_SOURCE=${WORLD_STATE_SOURCE:-obs_concat}" \
    "SAVE_PATH=${MAPPO_LORA_SAVE_PATH:-check_points/mappo_rnn_overcooked_v2_lora_tta}" \
    "TENSORBOARD_ENABLED=${TENSORBOARD_ENABLED:-False}" \
    "${COMMON_OVERRIDES[@]}" \
    "${USER_OVERRIDES[@]}"
}

echo "Repository : $REPO_ROOT"
echo "Algorithm  : $ALG"
echo "W&B mode   : $EFFECTIVE_WANDB_MODE"
echo "Project    : ${EFFECTIVE_PROJECT:-<empty>}"
echo "Entity     : ${EFFECTIVE_ENTITY:-<default_account>}"
echo "Group      : $EFFECTIVE_WANDB_GROUP"
echo "Layout     : $EFFECTIVE_LAYOUT"
echo "Num seeds  : $EFFECTIVE_NUM_SEEDS"

case "$ALG" in
  ippo)
    run_ippo
    ;;
  mappo)
    run_mappo
    ;;
  mappo_lora_tta)
    run_mappo_lora_tta
    ;;
  both)
    run_ippo
    run_mappo
    ;;
  *)
    echo "Unsupported algorithm: $ALG" >&2
    echo "Usage: $0 [ippo|mappo|mappo_lora_tta|both] [HYDRA_OVERRIDES...]" >&2
    exit 1
    ;;
esac

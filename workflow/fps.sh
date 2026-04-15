#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_PY="$SCRIPT_DIR/utils/fps_qbc_ops.py"
FPS_EXTRACT_PY="$SCRIPT_DIR/utils/extract.py"
FPS_ONLY_OPS_PY="$SCRIPT_DIR/utils/fps_only_ops.py"

usage() {
  cat <<'EOF'
Usage:
  workflow/fps.sh \
    [--config PATH] \
    --input_xyz PATH \
    --init_size N \
    --select_size M \
    --max_rounds R \
    [--workdir DIR] \
    [--fps_model PATH] \
    [--seed INT] \
    [--device cuda|cpu] \
    [--fps_batch_size INT] \
    [--fps_feature_key KEY] \
    [--fps_normalization none|zscore]
EOF
}

INPUT_XYZ=""
INIT_SIZE=""
SELECT_SIZE=""
MAX_ROUNDS=""

WORKDIR="workflow/fps_runs"
FPS_MODEL=""
GLOBAL_SEED=42
DEVICE="cuda"
FPS_BATCH_SIZE=32
FPS_FEATURE_KEY="structure_features"
FPS_NORMALIZATION="zscore"
CONFIG_FILE=""

apply_config_kv() {
  local key="$1"
  local value="$2"
  case "$key" in
    input_xyz) INPUT_XYZ="$value" ;;
    init_size) INIT_SIZE="$value" ;;
    select_size) SELECT_SIZE="$value" ;;
    max_rounds) MAX_ROUNDS="$value" ;;
    workdir) WORKDIR="$value" ;;
    fps_model) FPS_MODEL="$value" ;;
    seed) GLOBAL_SEED="$value" ;;
    device) DEVICE="$value" ;;
    fps_batch_size) FPS_BATCH_SIZE="$value" ;;
    batch_size) FPS_BATCH_SIZE="$value" ;;
    fps_feature_key) FPS_FEATURE_KEY="$value" ;;
    fps_normalization) FPS_NORMALIZATION="$value" ;;
    *) echo "[ERROR] Unknown config key: $key"; exit 1 ;;
  esac
}

ARGV=("$@")
for ((i=0; i<${#ARGV[@]}; i++)); do
  if [[ "${ARGV[$i]}" == "--config" ]]; then
    if (( i + 1 >= ${#ARGV[@]} )); then
      echo "[ERROR] --config requires a value"
      exit 1
    fi
    CONFIG_FILE="${ARGV[$((i+1))]}"
  fi
done

if [[ ! -f "$OPS_PY" ]]; then
  echo "[ERROR] Helper script not found: $OPS_PY"
  exit 1
fi

if [[ ! -f "$FPS_ONLY_OPS_PY" ]]; then
  echo "[ERROR] Helper script not found: $FPS_ONLY_OPS_PY"
  exit 1
fi

if [[ -n "$CONFIG_FILE" ]]; then
  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE"
    exit 1
  fi
  while IFS=$'\t' read -r cfg_key cfg_value; do
    apply_config_kv "$cfg_key" "$cfg_value"
  done < <(python "$OPS_PY" config-env --config "$CONFIG_FILE")
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) shift 2 ;;
    --input_xyz) INPUT_XYZ="$2"; shift 2 ;;
    --init_size) INIT_SIZE="$2"; shift 2 ;;
    --select_size) SELECT_SIZE="$2"; shift 2 ;;
    --max_rounds) MAX_ROUNDS="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --fps_model) FPS_MODEL="$2"; shift 2 ;;
    --seed) GLOBAL_SEED="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --fps_batch_size) FPS_BATCH_SIZE="$2"; shift 2 ;;
    --fps_feature_key) FPS_FEATURE_KEY="$2"; shift 2 ;;
    --fps_normalization) FPS_NORMALIZATION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$INPUT_XYZ" || -z "$INIT_SIZE" || -z "$SELECT_SIZE" || -z "$MAX_ROUNDS" ]]; then
  echo "[ERROR] Missing required arguments"
  usage
  exit 1
fi

if [[ ! -f "$INPUT_XYZ" ]]; then
  echo "[ERROR] Input xyz not found: $INPUT_XYZ"
  exit 1
fi

if ! [[ "$FPS_BATCH_SIZE" =~ ^[0-9]+$ ]] || [[ "$FPS_BATCH_SIZE" -le 0 ]]; then
  echo "[ERROR] --fps_batch_size must be a positive integer, got: $FPS_BATCH_SIZE"
  exit 1
fi

case "$FPS_NORMALIZATION" in
  none|zscore) ;;
  *) echo "[ERROR] Invalid --fps_normalization: $FPS_NORMALIZATION"; exit 1 ;;
esac

mkdir -p "$WORKDIR"

echo "[INFO] Workdir: $WORKDIR"
echo "[INFO] FPS batch size: $FPS_BATCH_SIZE"
echo "[INFO] FPS normalization: $FPS_NORMALIZATION"

python "$OPS_PY" init-split \
  --input_xyz "$INPUT_XYZ" \
  --init_size "$INIT_SIZE" \
  --seed "$GLOBAL_SEED" \
  --train_out "$WORKDIR/train_round0.xyz" \
  --pool_out "$WORKDIR/pool_round0.xyz"

for ((round=0; round<MAX_ROUNDS; round++)); do
  round_dir="$WORKDIR/round${round}"
  next_round_dir="$WORKDIR/round$((round+1))"
  mkdir -p "$round_dir" "$next_round_dir"

  train_xyz="$WORKDIR/train_round${round}.xyz"
  pool_xyz="$WORKDIR/pool_round${round}.xyz"
  next_train_xyz="$WORKDIR/train_round$((round+1)).xyz"
  next_pool_xyz="$WORKDIR/pool_round$((round+1)).xyz"

  if [[ ! -f "$train_xyz" || ! -f "$pool_xyz" ]]; then
    echo "[ERROR] Missing round input files: $train_xyz or $pool_xyz"
    exit 1
  fi

  pool_size=$(python "$OPS_PY" count-frames --input_xyz "$pool_xyz")
  if [[ "$pool_size" -le 0 ]]; then
    echo "[INFO] Round $round: pool is empty, stop"
    break
  fi

  echo "[INFO] ===== Round $round ====="
  echo "[INFO] train=$train_xyz pool=$pool_xyz (n_pool=$pool_size)"

  train_feature_file="$round_dir/train_features.npz"
  pool_feature_file="$round_dir/pool_features.npz"
  next_train_feature_file="$next_round_dir/train_features.npz"
  next_pool_feature_file="$next_round_dir/pool_features.npz"
  fps_prefix="$round_dir/selected_fps"

  if (( round == 0 )); then
    echo "[INFO] Round 0: extracting FPS features for train/pool"
    extract_train_cmd=(
      python "$FPS_EXTRACT_PY"
      --input "$train_xyz"
      --output "$train_feature_file"
      --device "$DEVICE"
      --batch_size "$FPS_BATCH_SIZE"
    )
    if [[ -n "$FPS_MODEL" ]]; then
      extract_train_cmd+=(--model "$FPS_MODEL")
    fi
    "${extract_train_cmd[@]}"

    extract_pool_cmd=(
      python "$FPS_EXTRACT_PY"
      --input "$pool_xyz"
      --output "$pool_feature_file"
      --device "$DEVICE"
      --batch_size "$FPS_BATCH_SIZE"
    )
    if [[ -n "$FPS_MODEL" ]]; then
      extract_pool_cmd+=(--model "$FPS_MODEL")
    fi
    "${extract_pool_cmd[@]}"
  else
    if [[ ! -f "$train_feature_file" || ! -f "$pool_feature_file" ]]; then
      echo "[ERROR] Missing cached feature files for round $round: $train_feature_file or $pool_feature_file"
      exit 1
    fi
    echo "[INFO] Round $round: reusing cached FPS features"
  fi

  selected_k="$SELECT_SIZE"
  if [[ "$pool_size" -lt "$SELECT_SIZE" ]]; then
    selected_k="$pool_size"
  fi
  echo "[INFO] Round $round: selecting selected_k=$selected_k by anchored FPS (anchors=train set)"

  python "$OPS_PY" anchored-fps \
    --train_feature_file "$train_feature_file" \
    --pool_feature_file "$pool_feature_file" \
    --feature_key "$FPS_FEATURE_KEY" \
    --pool_xyz "$pool_xyz" \
    --candidate_k "$selected_k" \
    --normalization "$FPS_NORMALIZATION" \
    --output_prefix "$fps_prefix"

  candidate_xyz="${fps_prefix}_${selected_k}.extxyz"
  candidate_pool_indices="${fps_prefix}_${selected_k}_indices.npy"
  if [[ ! -f "$candidate_xyz" || ! -f "$candidate_pool_indices" ]]; then
    echo "[ERROR] FPS outputs missing for round $round"
    exit 1
  fi

  selected_local_npy="$round_dir/selected_candidate_local_indices.npy"
  python "$FPS_ONLY_OPS_PY" select-all-candidates \
    --candidate_pool_indices_npy "$candidate_pool_indices" \
    --selected_local_out "$selected_local_npy"

  python "$OPS_PY" update-datasets \
    --train_xyz "$train_xyz" \
    --pool_xyz "$pool_xyz" \
    --candidate_pool_indices_npy "$candidate_pool_indices" \
    --selected_local_npy "$selected_local_npy" \
    --next_train_xyz "$next_train_xyz" \
    --next_pool_xyz "$next_pool_xyz"

  python "$OPS_PY" update-feature-sets \
    --train_feature_file "$train_feature_file" \
    --pool_feature_file "$pool_feature_file" \
    --feature_key "$FPS_FEATURE_KEY" \
    --candidate_pool_indices_npy "$candidate_pool_indices" \
    --selected_local_npy "$selected_local_npy" \
    --next_train_feature_file "$next_train_feature_file" \
    --next_pool_feature_file "$next_pool_feature_file"
done

echo "[INFO] FPS workflow finished"

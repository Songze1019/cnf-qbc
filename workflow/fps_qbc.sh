#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_PY="$SCRIPT_DIR/utils/fps_qbc_ops.py"
FPS_EXTRACT_PY="$SCRIPT_DIR/utils/extract.py"

usage() {
  cat <<'EOF'
Usage:
  workflow/fps_qbc.sh \
    [--config PATH] \
    --input_xyz PATH \
    --init_size N \
    --candidate_pct PCT \
    --select_size M \
    --max_rounds R \
    [--workdir DIR] \
    [--train_work_dir DIR] \
    [--log_dir DIR] \
    [--model_dir DIR] \
    [--checkpoints_dir DIR] \
    [--results_dir DIR] \
    [--downloads_dir DIR] \
    [--valid_xyz PATH] \
    [--test_xyz PATH] \
    [--fps_model PATH] \
    [--seeds 0,1,2] \
    [--seed INT] \
    [--device cuda|cpu] \
    [--batch_size INT] \
    [--fps_batch_size INT] \
    [--eval_batch_size INT] \
    [--max_num_epochs INT] \
    [--r_max FLOAT] \
    [--hidden_irreps STRING] \
    [--force_weight FLOAT] \
    [--uncertainty_metric force_std_p95|force_std_mean|energy_std_abs] \
    [--uncertainty_threshold FLOAT] \
    [--fps_feature_key KEY] \
    [--fps_normalization none|zscore] \
    [--enable_cueq True|False] \
    [--metrics_file PATH] \
    [--metrics_interval INT]
EOF
}

INPUT_XYZ=""
INIT_SIZE=""
CANDIDATE_PCT=""
SELECT_SIZE=""
MAX_ROUNDS=""

WORKDIR="workflow/fps_qbc_runs"
TRAIN_WORK_DIR=""
LOG_DIR=""
MODEL_DIR=""
CHECKPOINTS_DIR=""
RESULTS_DIR=""
DOWNLOADS_DIR=""
VALID_XYZ=""
TEST_XYZ=""
FPS_MODEL=""
SEEDS="0,1,2"
GLOBAL_SEED=42
DEVICE="cuda"
BATCH_SIZE=4
FPS_BATCH_SIZE=""
EVAL_BATCH_SIZE=""
MAX_NUM_EPOCHS=5
R_MAX=6.0
HIDDEN_IRREPS='32x0e + 32x1o'
FORCES_WEIGHT=10.0
UNCERTAINTY_METRIC="force_std_p95"
UNCERTAINTY_THRESHOLD=""
FPS_FEATURE_KEY="structure_features"
FPS_NORMALIZATION="zscore"
ENABLE_CUEQ="True"
METRICS_FILE=""
METRICS_INTERVAL=1
CONFIG_FILE=""

apply_config_kv() {
  local key="$1"
  local value="$2"
  case "$key" in
    input_xyz) INPUT_XYZ="$value" ;;
    init_size) INIT_SIZE="$value" ;;
    candidate_pct) CANDIDATE_PCT="$value" ;;
    select_size) SELECT_SIZE="$value" ;;
    max_rounds) MAX_ROUNDS="$value" ;;
    workdir) WORKDIR="$value" ;;
    train_work_dir) TRAIN_WORK_DIR="$value" ;;
    log_dir) LOG_DIR="$value" ;;
    model_dir) MODEL_DIR="$value" ;;
    checkpoints_dir) CHECKPOINTS_DIR="$value" ;;
    results_dir) RESULTS_DIR="$value" ;;
    downloads_dir) DOWNLOADS_DIR="$value" ;;
    valid_xyz) VALID_XYZ="$value" ;;
    test_xyz) TEST_XYZ="$value" ;;
    fps_model) FPS_MODEL="$value" ;;
    seeds) SEEDS="$value" ;;
    seed) GLOBAL_SEED="$value" ;;
    device) DEVICE="$value" ;;
    batch_size) BATCH_SIZE="$value" ;;
    fps_batch_size) FPS_BATCH_SIZE="$value" ;;
    eval_batch_size) EVAL_BATCH_SIZE="$value" ;;
    max_num_epochs) MAX_NUM_EPOCHS="$value" ;;
    r_max) R_MAX="$value" ;;
    hidden_irreps) HIDDEN_IRREPS="$value" ;;
    force_weight) FORCES_WEIGHT="$value" ;;
    uncertainty_metric) UNCERTAINTY_METRIC="$value" ;;
    uncertainty_threshold) UNCERTAINTY_THRESHOLD="$value" ;;
    fps_feature_key) FPS_FEATURE_KEY="$value" ;;
    fps_normalization) FPS_NORMALIZATION="$value" ;;
    enable_cueq) ENABLE_CUEQ="$value" ;;
    cueq) ENABLE_CUEQ="$value" ;;
    metrics_file) METRICS_FILE="$value" ;;
    metrics_interval) METRICS_INTERVAL="$value" ;;
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
    --candidate_pct) CANDIDATE_PCT="$2"; shift 2 ;;
    --select_size) SELECT_SIZE="$2"; shift 2 ;;
    --max_rounds) MAX_ROUNDS="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --train_work_dir) TRAIN_WORK_DIR="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --model_dir) MODEL_DIR="$2"; shift 2 ;;
    --checkpoints_dir) CHECKPOINTS_DIR="$2"; shift 2 ;;
    --results_dir) RESULTS_DIR="$2"; shift 2 ;;
    --downloads_dir) DOWNLOADS_DIR="$2"; shift 2 ;;
    --valid_xyz) VALID_XYZ="$2"; shift 2 ;;
    --test_xyz) TEST_XYZ="$2"; shift 2 ;;
    --fps_model) FPS_MODEL="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --seed) GLOBAL_SEED="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --fps_batch_size) FPS_BATCH_SIZE="$2"; shift 2 ;;
    --eval_batch_size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --max_num_epochs) MAX_NUM_EPOCHS="$2"; shift 2 ;;
    --r_max) R_MAX="$2"; shift 2 ;;
    --hidden_irreps) HIDDEN_IRREPS="$2"; shift 2 ;;
    --force_weight) FORCES_WEIGHT="$2"; shift 2 ;;
    --uncertainty_metric) UNCERTAINTY_METRIC="$2"; shift 2 ;;
    --uncertainty_threshold) UNCERTAINTY_THRESHOLD="$2"; shift 2 ;;
    --fps_feature_key) FPS_FEATURE_KEY="$2"; shift 2 ;;
    --fps_normalization) FPS_NORMALIZATION="$2"; shift 2 ;;
    --enable_cueq) ENABLE_CUEQ="$2"; shift 2 ;;
    --metrics_file) METRICS_FILE="$2"; shift 2 ;;
    --metrics_interval) METRICS_INTERVAL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$INPUT_XYZ" || -z "$INIT_SIZE" || -z "$CANDIDATE_PCT" || -z "$SELECT_SIZE" || -z "$MAX_ROUNDS" ]]; then
  echo "[ERROR] Missing required arguments"
  usage
  exit 1
fi

if [[ ! -f "$INPUT_XYZ" ]]; then
  echo "[ERROR] Input xyz not found: $INPUT_XYZ"
  exit 1
fi

if [[ -z "$CHECKPOINTS_DIR" ]]; then
  CHECKPOINTS_DIR="$WORKDIR/checkpoints"
elif [[ "$CHECKPOINTS_DIR" != /* && "$CHECKPOINTS_DIR" != "$WORKDIR"/* ]]; then
  CHECKPOINTS_DIR="$WORKDIR/$CHECKPOINTS_DIR"
fi

if [[ -z "$TRAIN_WORK_DIR" ]]; then
  TRAIN_WORK_DIR="$WORKDIR/train_runs"
elif [[ "$TRAIN_WORK_DIR" != /* && "$TRAIN_WORK_DIR" != "$WORKDIR"/* ]]; then
  TRAIN_WORK_DIR="$WORKDIR/$TRAIN_WORK_DIR"
fi

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$TRAIN_WORK_DIR/logs"
elif [[ "$LOG_DIR" != /* && "$LOG_DIR" != "$WORKDIR"/* ]]; then
  LOG_DIR="$WORKDIR/$LOG_DIR"
fi

if [[ -z "$MODEL_DIR" ]]; then
  MODEL_DIR="$TRAIN_WORK_DIR/model"
elif [[ "$MODEL_DIR" != /* && "$MODEL_DIR" != "$WORKDIR"/* ]]; then
  MODEL_DIR="$WORKDIR/$MODEL_DIR"
fi

if [[ -z "$RESULTS_DIR" ]]; then
  RESULTS_DIR="$TRAIN_WORK_DIR/results"
elif [[ "$RESULTS_DIR" != /* && "$RESULTS_DIR" != "$WORKDIR"/* ]]; then
  RESULTS_DIR="$WORKDIR/$RESULTS_DIR"
fi

if [[ -z "$DOWNLOADS_DIR" ]]; then
  DOWNLOADS_DIR="$TRAIN_WORK_DIR/downloads"
elif [[ "$DOWNLOADS_DIR" != /* && "$DOWNLOADS_DIR" != "$WORKDIR"/* ]]; then
  DOWNLOADS_DIR="$WORKDIR/$DOWNLOADS_DIR"
fi

if [[ -z "$METRICS_FILE" ]]; then
  METRICS_FILE="$WORKDIR/round_metrics.csv"
elif [[ "$METRICS_FILE" != /* && "$METRICS_FILE" != "$WORKDIR"/* ]]; then
  METRICS_FILE="$WORKDIR/$METRICS_FILE"
fi

if [[ -z "$FPS_BATCH_SIZE" ]]; then
  FPS_BATCH_SIZE="$BATCH_SIZE"
fi

if [[ -z "$EVAL_BATCH_SIZE" ]]; then
  EVAL_BATCH_SIZE="$BATCH_SIZE"
fi

if ! [[ "$FPS_BATCH_SIZE" =~ ^[0-9]+$ ]] || [[ "$FPS_BATCH_SIZE" -le 0 ]]; then
  echo "[ERROR] --fps_batch_size must be a positive integer, got: $FPS_BATCH_SIZE"
  exit 1
fi

if ! [[ "$EVAL_BATCH_SIZE" =~ ^[0-9]+$ ]] || [[ "$EVAL_BATCH_SIZE" -le 0 ]]; then
  echo "[ERROR] --eval_batch_size must be a positive integer, got: $EVAL_BATCH_SIZE"
  exit 1
fi

if ! [[ "$METRICS_INTERVAL" =~ ^[0-9]+$ ]] || [[ "$METRICS_INTERVAL" -le 0 ]]; then
  echo "[ERROR] --metrics_interval must be a positive integer, got: $METRICS_INTERVAL"
  exit 1
fi

IFS=',' read -r -a SEEDS_ARR <<< "$SEEDS"
if [[ ${#SEEDS_ARR[@]} -ne 3 ]]; then
  echo "[ERROR] --seeds must contain exactly 3 seeds, got: $SEEDS"
  exit 1
fi

case "$UNCERTAINTY_METRIC" in
  force_std_p95|force_std_mean|energy_std_abs) ;;
  *) echo "[ERROR] Invalid --uncertainty_metric: $UNCERTAINTY_METRIC"; exit 1 ;;
esac

case "$FPS_NORMALIZATION" in
  none|zscore) ;;
  *) echo "[ERROR] Invalid --fps_normalization: $FPS_NORMALIZATION"; exit 1 ;;
esac

mkdir -p "$WORKDIR" "$TRAIN_WORK_DIR" "$LOG_DIR" "$MODEL_DIR" "$CHECKPOINTS_DIR" "$RESULTS_DIR" "$DOWNLOADS_DIR"

echo "[INFO] Workdir: $WORKDIR"
echo "[INFO] Train work dir: $TRAIN_WORK_DIR"
echo "[INFO] Logs: $LOG_DIR"
echo "[INFO] Model dir: $MODEL_DIR"
echo "[INFO] Checkpoints: $CHECKPOINTS_DIR"
echo "[INFO] Results dir: $RESULTS_DIR"
echo "[INFO] Downloads dir: $DOWNLOADS_DIR"
echo "[INFO] Metrics file: $METRICS_FILE"
echo "[INFO] FPS batch size: $FPS_BATCH_SIZE"
echo "[INFO] Eval batch size: $EVAL_BATCH_SIZE"
echo "[INFO] Metrics interval: $METRICS_INTERVAL"
echo "[INFO] FPS normalization: $FPS_NORMALIZATION"

if [[ -z "$TEST_XYZ" ]]; then
  echo "[WARN] --test_xyz is not set; test uncertainty/RMSE/L4 will be saved as nan"
fi

python "$OPS_PY" init-split \
  --input_xyz "$INPUT_XYZ" \
  --init_size "$INIT_SIZE" \
  --seed "$GLOBAL_SEED" \
  --train_out "$WORKDIR/train_round0.xyz" \
  --pool_out "$WORKDIR/pool_round0.xyz"

for ((round=0; round<MAX_ROUNDS; round++)); do
  round_dir="$WORKDIR/round${round}"
  next_round_dir="$WORKDIR/round$((round+1))"
  mkdir -p "$round_dir"

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

  model_name="fps_qbc_round${round}"
  for seed in "${SEEDS_ARR[@]}"; do
    echo "[INFO] Training model round=$round seed=$seed"

    train_cmd=(
      mace_run_train
      --name="$model_name"
      --train_file="$train_xyz"
      --forces_weight="$FORCES_WEIGHT"
      --E0s='average'
      --hidden_irreps="$HIDDEN_IRREPS"
      --r_max="$R_MAX"
      --batch_size="$BATCH_SIZE"
      --max_num_epochs="$MAX_NUM_EPOCHS"
      --ema
      --ema_decay=0.99
      --amsgrad
      --device="$DEVICE"
      --default_dtype='float32'
      --scheduler_patience=5
      --work_dir="$TRAIN_WORK_DIR"
      --log_dir="$LOG_DIR"
      --model_dir="$MODEL_DIR"
      --checkpoints_dir="$CHECKPOINTS_DIR"
      --results_dir="$RESULTS_DIR"
      --downloads_dir="$DOWNLOADS_DIR"
      --seed="$seed"
      --enable_cueq="$ENABLE_CUEQ"
      --pair_repulsion
    )

    if [[ -n "$VALID_XYZ" ]]; then
      train_cmd+=(--valid_file="$VALID_XYZ")
    else
      train_cmd+=(--valid_fraction=0.1)
    fi

    if [[ -n "$TEST_XYZ" ]]; then
      train_cmd+=(--test_file="$TEST_XYZ")
    fi

    "${train_cmd[@]}"
  done

  model_paths=()
  for seed in "${SEEDS_ARR[@]}"; do
    model_path="$CHECKPOINTS_DIR/${model_name}_run-${seed}.model"
    if [[ ! -f "$model_path" ]]; then
      echo "[ERROR] Expected model not found: $model_path"
      exit 1
    fi
    model_paths+=("$model_path")
  done

  candidate_k=$(python "$OPS_PY" candidate-k --pool_size "$pool_size" --candidate_pct "$CANDIDATE_PCT")
  echo "[INFO] Round $round: selecting candidate_k=$candidate_k by anchored FPS (anchors=train set)"

  train_feature_file="$round_dir/train_features.npz"
  pool_feature_file="$round_dir/pool_features.npz"
  next_train_feature_file="$next_round_dir/train_features.npz"
  next_pool_feature_file="$next_round_dir/pool_features.npz"
  fps_prefix="$round_dir/candidates_fps"

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

  python "$OPS_PY" anchored-fps \
    --train_feature_file "$train_feature_file" \
    --pool_feature_file "$pool_feature_file" \
    --feature_key "$FPS_FEATURE_KEY" \
    --pool_xyz "$pool_xyz" \
    --candidate_k "$candidate_k" \
    --normalization "$FPS_NORMALIZATION" \
    --output_prefix "$fps_prefix"

  candidate_xyz="${fps_prefix}_${candidate_k}.extxyz"
  candidate_pool_indices="${fps_prefix}_${candidate_k}_indices.npy"
  if [[ ! -f "$candidate_xyz" || ! -f "$candidate_pool_indices" ]]; then
    echo "[ERROR] FPS outputs missing for round $round"
    exit 1
  fi

  selected_local_npy="$round_dir/selected_candidate_local_indices.npy"
  stats_npy="$round_dir/candidate_uncertainty.npy"

  echo "[INFO] Round $round: computing committee uncertainty"
  max_uncert=$(python "$OPS_PY" committee-uncertainty \
    --candidate_xyz "$candidate_xyz" \
    --select_size "$SELECT_SIZE" \
    --metric "$UNCERTAINTY_METRIC" \
    --selected_local_out "$selected_local_npy" \
    --stats_out "$stats_npy" \
    --device "$DEVICE" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --enable_cueq "$ENABLE_CUEQ" \
    --model_paths "${model_paths[@]}")

  echo "[INFO] Round $round: max uncertainty = $max_uncert"

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

  next_pool_size=$(python "$OPS_PY" count-frames --input_xyz "$next_pool_xyz")
  test_mean_uncertainty="nan"
  test_uncertainty_stats_npy="$round_dir/test_uncertainty.npy"
  should_eval_metrics=1
  if [[ "$METRICS_INTERVAL" -gt 1 ]] && (( round % METRICS_INTERVAL != 0 )); then
    should_eval_metrics=0
  fi
  if [[ -n "$TEST_XYZ" && "$should_eval_metrics" -eq 1 ]]; then
    test_mean_uncertainty=$(python "$OPS_PY" test-uncertainty \
      --test_xyz "$TEST_XYZ" \
      --metric "$UNCERTAINTY_METRIC" \
      --stats_out "$test_uncertainty_stats_npy" \
      --device "$DEVICE" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --enable_cueq "$ENABLE_CUEQ" \
      --model_paths "${model_paths[@]}")
  elif [[ -n "$TEST_XYZ" ]]; then
    echo "[INFO] Round $round: skip test uncertainty (metrics_interval=$METRICS_INTERVAL)"
  fi

  test_energy_rmse="nan"
  test_force_rmse="nan"
  test_energy_l4="nan"
  test_force_l4="nan"
  if [[ -n "$TEST_XYZ" && "$should_eval_metrics" -eq 1 ]]; then
    read -r test_energy_rmse test_force_rmse test_energy_l4 test_force_l4 <<< "$(python "$OPS_PY" test-rmse \
      --test_xyz "$TEST_XYZ" \
      --device "$DEVICE" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --enable_cueq "$ENABLE_CUEQ" \
      --model_paths "${model_paths[@]}")"
  elif [[ -n "$TEST_XYZ" ]]; then
    echo "[INFO] Round $round: skip test RMSE (metrics_interval=$METRICS_INTERVAL)"
  fi

  selected_size_round="$SELECT_SIZE"
  if [[ "$candidate_k" -lt "$SELECT_SIZE" ]]; then
    selected_size_round="$candidate_k"
  fi
  python "$OPS_PY" append-round-metrics \
    --metrics_file "$METRICS_FILE" \
    --round "$round" \
    --candidate_k "$candidate_k" \
    --selected_size "$selected_size_round" \
    --pool_size_after_select "$next_pool_size" \
    --uncertainty_metric "$UNCERTAINTY_METRIC" \
    --max_candidate_uncertainty "$max_uncert" \
    --test_mean_uncertainty "$test_mean_uncertainty" \
    --test_energy_rmse "$test_energy_rmse" \
    --test_force_rmse "$test_force_rmse" \
    --test_energy_l4 "$test_energy_l4" \
    --test_force_l4 "$test_force_l4"

  echo "[INFO] Round $round metrics: test_mean_unc=$test_mean_uncertainty test_E_RMSE=$test_energy_rmse test_F_RMSE=$test_force_rmse test_E_L4=$test_energy_l4 test_F_L4=$test_force_l4"

  if [[ -n "$UNCERTAINTY_THRESHOLD" ]]; then
    should_stop=$(python "$OPS_PY" threshold-stop --max_uncert "$max_uncert" --threshold "$UNCERTAINTY_THRESHOLD")
    if [[ "$should_stop" == "1" ]]; then
      echo "[INFO] Round $round: max uncertainty <= threshold ($UNCERTAINTY_THRESHOLD), stop"
      break
    fi
  fi
done

echo "[INFO] FPS+QBC workflow finished"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-${ROOT_DIR}/rets/transition1x-neb-idpp2stage}"
PREDICTED_TS_DIR="${OUTPUT_DIR}/predicted_ts"

COMMON_ARGS=(
  python "${ROOT_DIR}/utils/evaluate_transition1x_neb_round_models.py"
  --device cuda
  --no_enable_cueq
  --fmax 0.10
  --steps 200
  --resume
  --output_dir "${OUTPUT_DIR}"
)

EXPECTED_SPECS=(
  "fps_qbc:4"
  "fps_qbc:12"
  "fps_qbc:20"
  "qbc:4"
  "qbc:12"
  "qbc:20"
  "fps:4"
  "fps:12"
  "fps:20"
  "random:4"
  "random:12"
  "random:20"
  "all_data:all"
)

mapfile -t missing_specs < <(
  python - "${PREDICTED_TS_DIR}" "${EXPECTED_SPECS[@]}" <<'PY'
from pathlib import Path
import sys
from ase.io import iread

predicted_ts_dir = Path(sys.argv[1])
expected_specs = sys.argv[2:]

for spec in expected_specs:
    family, round_label = spec.split(":", 1)
    path = predicted_ts_dir / f"{family}_round{round_label}_test_ts.xyz"
    if not path.exists():
        print(spec)
        continue
    try:
        sum(1 for _ in iread(path, index=":"))
    except Exception:
        print(spec)
PY
)

if [[ ${#missing_specs[@]} -eq 0 ]]; then
  echo "[INFO] No missing NEB specs found in ${PREDICTED_TS_DIR}"
  exit 0
fi

echo "[INFO] Output dir: ${OUTPUT_DIR}"
echo "[INFO] Missing specs to run:"
for spec in "${missing_specs[@]}"; do
  echo "  - ${spec}"
done

echo
echo "[WARN] This script only fills missing specs based on predicted_ts files."
echo "[WARN] If ${OUTPUT_DIR}/transition1x_neb_details.csv is absent, the final summary will still be incomplete."
echo

for spec in "${missing_specs[@]}"; do
  family="${spec%%:*}"
  round_label="${spec##*:}"
  echo "[INFO] Running ${family} round=${round_label}"

  if [[ "${family}" == "all_data" ]]; then
    srun -p 4V100PX --gres=gpu:1 --qos=improper-gpu \
      "${COMMON_ARGS[@]}" \
      --only_all_data
  else
    srun -p 4V100PX --gres=gpu:1 --qos=improper-gpu \
      "${COMMON_ARGS[@]}" \
      --families "${family}" \
      --rounds "${round_label}" \
      --no_all_data
  fi
done

echo "[INFO] Missing-spec run loop finished."

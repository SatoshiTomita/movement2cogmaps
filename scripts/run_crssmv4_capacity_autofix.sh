#!/usr/bin/env bash
set -u -o pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

export CUDA_VISIBLE_DEVICES=2
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/movement2cogmaps-matplotlib}"
export PYTHONUNBUFFERED=1

python_bin="${PYTHON_BIN:-$repo_dir/.venv311/bin/python}"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_dir="$repo_dir/logs"
log_file="$log_dir/crssmv4_capacity_autofix_${run_stamp}.log"
mkdir -p "$log_dir"

run_policy() {
  local beta="$1"
  local max_beta="$2"
  local save_name="crssmv4_capacity_adaptive_beta${beta}_${run_stamp}"

  echo "[$(date -Is)] starting adaptive capacity policy beta=${beta}, max_beta=${max_beta}" | tee -a "$log_file"
  "$python_bin" maincrssmV4.py \
    --model_name 0824base \
    --save-name "$save_name" \
    --alpha 1.0 --seed 0 --device 0 --world --train \
    --curriculum crawl --epochs 1500 --disable-early-stop --num-workers 8 \
    --kl-target 0.2 --kl-capacity-beta "$beta" --kl-capacity-mode quadratic \
    --adaptive-kl-capacity --capacity-adjustment-patience 3 \
    --capacity-adjustment-factor 2 --max-kl-capacity-beta "$max_beta" \
    --clean-old-checkpoints 2>&1 | tee -a "$log_file"
  local status=${PIPESTATUS[0]}
  return "$status"
}

run_policy 1 64
status=$?
if [ "$status" -eq 0 ]; then
  echo "[$(date -Is)] adaptive capacity policy completed" | tee -a "$log_file"
  exit 0
fi
if [ "$status" -ne 20 ]; then
  echo "[$(date -Is)] aborting: training failed with exit=${status}, not a capacity violation" | tee -a "$log_file" >&2
  exit "$status"
fi

echo "[$(date -Is)] first adaptive policy exhausted its capacity range; trying stronger initialization" | tee -a "$log_file"
run_policy 4 128
status=$?
if [ "$status" -eq 0 ]; then
  echo "[$(date -Is)] stronger adaptive capacity policy completed" | tee -a "$log_file"
  exit 0
fi
if [ "$status" -eq 20 ]; then
  echo "[$(date -Is)] no adaptive capacity policy produced a valid crawl" | tee -a "$log_file" >&2
else
  echo "[$(date -Is)] aborting: training failed with exit=${status}, not a capacity violation" | tee -a "$log_file" >&2
fi

exit "$status"

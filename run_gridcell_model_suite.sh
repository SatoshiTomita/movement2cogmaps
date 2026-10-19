#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

# CUDA device 0 inside each Python process maps to physical GPU 2.
export CUDA_VISIBLE_DEVICES=2
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/movement2cogmaps-matplotlib}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1

python_bin="${PYTHON_BIN:-$repo_dir/.venv311/bin/python}"
epochs="${EPOCHS:-1500}"
log_dir="${LOG_DIR:-$repo_dir/logs/gridcell_model_suite_lr1e4_20261012}"
mkdir -p "$log_dir" "$MPLCONFIGDIR"

common_args=(
  --behaviour adult
  --env box_messy
  --epochs "$epochs"
  --epoch_act "$epochs"
  --bptt_steps 9
  --stride 10
  --reset_hidden_at 10
  --n_gridcells 25
  --gridcells_softmax
  --gridcells_modules 0.2,0.4,0.6
  --gridcells_orientations 0.1
  --num_workers 8
  --seed 1
  --name_prefix gridcell_suite_lr1e4_20261012
  --wandb
)

# Re-running the suite resumes it: finished models are skipped, and a model
# whose final checkpoint exists but whose activity analysis is missing only
# runs the activity part (training is not repeated).
run_model() {
  local label="$1"
  local template="$2"
  shift 2
  local exp_dir="$repo_dir/data/box/adult/predictions/box_messy/vanilla/adult_${template}"
  local model_args=("$@" --model_name_template "$template")

  if compgen -G "$exp_dir/act_adult_epoch${epochs}*" > /dev/null; then
    echo "[$(date -Is)] skipping $label (already complete)" | tee -a "$log_dir/suite.log"
    return
  fi

  if [[ -f "$exp_dir/rnn_epoch${epochs}.pth" ]]; then
    local act_args=()
    for arg in "${common_args[@]}"; do
      [[ "$arg" == "--wandb" ]] || act_args+=("$arg")
    done
    echo "[$(date -Is)] starting $label (activity only)" | tee -a "$log_dir/suite.log"
    "$python_bin" RNN_experiment.py "${act_args[@]}" "${model_args[@]}" --activity_only \
      2>&1 | tee -a "$log_dir/${label}.log"
  else
    echo "[$(date -Is)] starting $label" | tee -a "$log_dir/suite.log"
    "$python_bin" RNN_experiment.py "${common_args[@]}" "${model_args[@]}" \
      2>&1 | tee "$log_dir/${label}.log"
  fi
  echo "[$(date -Is)] completed $label" | tee -a "$log_dir/suite.log"
}

# Model sizes reproduce the established parameter-matched comparison set.
# The paper's grid-cell condition uses a common 1e-4 learning rate.
run_model rnn500 gridcell_paperlr1e4_RNN500 \
  --architecture rnn \
  --latent_dim 500 \
  --lr 1e-4

run_model gru267 gridcell_paperlr1e4_GRU267 \
  --architecture gru \
  --latent_dim 267 \
  --lr 1e-4 \
  --activity_transform halfshift

run_model lstm221 gridcell_paperlr1e4_LSTM221 \
  --architecture lstm \
  --latent_dim 221 \
  --lr 1e-4 \
  --activity_transform halfshift

run_model rssm_d512 gridcell_paperlr1e4_RSSM_d512_z8x8 \
  --architecture rssm \
  --latent_dim 512 \
  --stoch_dim 8 \
  --stoch_dist categorical \
  --stoch_n_class 8 \
  --nonlinearity sigmoid \
  --kl_scale 1.0 \
  --free_nats 0.5 \
  --lr 1e-4 \
  --activity_transform identity \
  --cell-activity-source combined

run_model ctrnn gridcell_paperlr1e4_CTRNN500_tau1 \
  --architecture ctrnn \
  --latent_dim 500 \
  --ctrnn_tau 1 \
  --lr 1e-4

run_model ctrssm gridcell_paperlr1e4_CTRSSM_d512_z8x8_tau3 \
  --architecture ctrssm \
  --latent_dim 512 \
  --stoch_dim 8 \
  --stoch_dist categorical \
  --stoch_n_class 8 \
  --ctrnn_tau 3 \
  --kl_scale 1.0 \
  --free_nats 0.1 \
  --lr 1e-4 \
  --cell-activity-source combined


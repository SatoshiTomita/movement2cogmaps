#!/usr/bin/env bash
# Analyse the existing trained checkpoints, then export all state-wise tests.
# Usage: CUDA_VISIBLE_DEVICES=2 bash R_stats/run_selected_model_stats.sh
# For an existing R environment, run this script via micromamba run and set
# RETICULATE_PYTHON to that environment's Python executable.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="$repo_root/.venv311/bin/python"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/movement2cogmaps-matplotlib}"
export MPLBACKEND=Agg
export RETICULATE_PYTHON="${RETICULATE_PYTHON:-$python_bin}"

if ! command -v Rscript >/dev/null 2>&1; then
    echo 'Rscript is required. Install R plus the DescTools and reticulate packages first.' >&2
    exit 1
fi
Rscript -e 'stopifnot(requireNamespace("DescTools", quietly=TRUE), requireNamespace("reticulate", quietly=TRUE)); reticulate::import("numpy")'

"$python_bin" RNN_experiment.py \
    --architecture gru \
    --name_prefix 0720_267dim \
    --latent_dim 267 \
    --curriculum crawl,walk,run,adult \
    --env box_messy \
    --epochs 1500 --epoch_act 1500 \
    --seed 1 --num_workers 8 \
    --activity_only \
    --activity_transform minmax

"$python_bin" RNN_experiment.py \
    --architecture rssm \
    --name_prefix rssm_baseline \
    --latent_dim 128 --stoch_dim 8 \
    --stoch_dist categorical --stoch_n_class 8 \
    --kl_scale 1.0 --free_nats 0.1 \
    --curriculum crawl,walk,run,adult \
    --env box_messy \
    --epochs 1500 --epoch_act 1500 \
    --seed 1 --num_workers 8 \
    --activity_only \
    --activity_transform minmax \
    --cell-activity-source combined

"$python_bin" analyze_world_model.py \
    --model-name 0824base \
    --alpha 1.0 --seed 0 --device 0 \
    --stages crawl,walk,run,adult \
    --activity-transform minmax \
    --cell-activity-source combined \
    --no-deterministic-latents \
    --skip-plots

"$python_bin" R_stats/export_model_stats.py --latents sampled --run-r

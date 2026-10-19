#!/bin/bash
# Simulate trajectories and render frames for every behaviour.
# Override with env vars, e.g. DATA_ROOT=data_seed27 SEEDS="27" CUDA_VISIBLE_DEVICES=2 bash simulation/run_multiple_simulations.sh
set -e
SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SIM_DIR")"
PYTHON="${PYTHON:-$REPO_ROOT/.venv311/bin/python}"
DATA_ROOT="$(realpath -m "${DATA_ROOT:-$REPO_ROOT/data}")"
SEEDS="${SEEDS:-$(seq -s ' ' 1 26)}"

for behaviour in crawl walk run adult; do
    "$PYTHON" "$SIM_DIR/riab_simulation/riabox_simulation.py" --behaviour "$behaviour" --environment box \
        --save_experiment --data_root "$DATA_ROOT" --seeds $SEEDS
    "$PYTHON" "$SIM_DIR/venv_blender/render_animation.py" --behaviours "$behaviour" --env box_messy \
        --data-root "$DATA_ROOT" --seeds $SEEDS
done

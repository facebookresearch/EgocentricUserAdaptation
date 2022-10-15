#!/usr/bin/env bash

# For sgd
grid_cfg_names="SOLVER.BASE_LR,SOLVER.MOMENTUM,SOLVER.NESTEROV,SOLVER.MOMENTUM_FEAT,SOLVER.MOMENTUM_HEAD" # Split by comma
grid_overwrite_args=""

# GRID momentum strength FEAT-only
val_idx=3
gridvals=("0.0" "0.3" "0.6" "0.9")
grid_arg="SOLVER.MOMENTUM_FEAT ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

# GRID momentum strength CLASSIFIER-only
val_idx=0
gridvals=("0.0" "0.3" "0.6" "0.9")
grid_arg="SOLVER.MOMENTUM_HEAD ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

# Only best adaptation LR:
val_idx=0
gridvals=("1e-2")
grid_arg="SOLVER.BASE_LR ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

# NESTEROV ONLY
val_idx=0
gridvals=(True)
grid_arg="SOLVER.NESTEROV ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

# Grid specific resources
grid_overwrite_args+=" GPU_IDS 6 NUM_USERS_PER_DEVICE 2 GRID_RESUME_LATEST False DATA_LOADER.NUM_WORKERS 6 WANDB.MODE 'online'" # 0,7

# Report final
echo "grid_overwrite_args=$grid_overwrite_args"

# Run script in current dir (same process with source)
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${__dir}/ego4d_action_recog.sh" "${grid_cfg_names}" "${grid_overwrite_args}"

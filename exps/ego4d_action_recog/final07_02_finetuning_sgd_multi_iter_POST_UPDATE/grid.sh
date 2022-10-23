#!/usr/bin/env bash

# For sgd
grid_cfg_names="SOLVER.BASE_LR,TRAIN.INNER_LOOP_ITERS" # Split by comma
grid_overwrite_args=""

val_idx=0 # TODO RUN
gridvals=( "1e-2" )
grid_arg="SOLVER.BASE_LR ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

val_idx=7
gridvals=("1" "2" "3" "5" "10" "15" "20" "30")
grid_arg="TRAIN.INNER_LOOP_ITERS ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

# Grid specific resources
grid_overwrite_args+=" DATA_LOADER.NUM_WORKERS 10 GPU_IDS 0 NUM_USERS_PER_DEVICE 2 GRID_RESUME_LATEST False" # 1,2,3,4,5,7

# Report final
echo "grid_overwrite_args=$grid_overwrite_args"

# Run script in current dir (same process with source)
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${__dir}/ego4d_action_recog.sh" "${grid_cfg_names}" "${grid_overwrite_args}"

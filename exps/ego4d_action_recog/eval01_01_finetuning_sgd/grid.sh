#!/usr/bin/env bash

# For sgd
grid_cfg_names="SOLVER.BASE_LR,SOLVER.MOMENTUM,SOLVER.NESTEROV" # Split by comma
grid_overwrite_args=""

val_idx=0 # TODO RUN
gridvals=( "1e-2" )
grid_arg="SOLVER.BASE_LR ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

val_idx=0
gridvals=( "0." )
grid_arg="SOLVER.MOMENTUM ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

val_idx=0
gridvals=( True False )
grid_arg="SOLVER.NESTEROV ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"


# Grid specific resources
grid_overwrite_args+=" GPU_IDS 0,1 NUM_USERS_PER_DEVICE 2 GRID_RESUME_LATEST False" # 0,7

# Report final
echo "grid_overwrite_args=$grid_overwrite_args"

# Run script in current dir (same process with source)
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${__dir}/ego4d_action_recog.sh" "${grid_cfg_names}" "${grid_overwrite_args}"

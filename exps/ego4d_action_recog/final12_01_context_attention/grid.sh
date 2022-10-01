#!/usr/bin/env bash

# For sgd
grid_cfg_names="SOLVER.BASE_LR,CONTEXT_ADAPT.MEM_SIZE" # Split by comma
grid_overwrite_args=""

val_idx=0 # TODO RUN
gridvals=( "1e-1" "1e-2" "1e-3")
grid_arg="SOLVER.BASE_LR ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

val_idx=3
gridvals=( "5" "10" "20" "30" )
grid_arg="CONTEXT_ADAPT.MEM_SIZE ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"


# Grid specific resources
grid_overwrite_args+=" GPU_IDS 7 NUM_USERS_PER_DEVICE 1 GRID_RESUME_LATEST False" # 0,7

# Report final
echo "grid_overwrite_args=$grid_overwrite_args"

# Run script in current dir (same process with source)
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${__dir}/ego4d_action_recog.sh" "${grid_cfg_names}" "${grid_overwrite_args}"

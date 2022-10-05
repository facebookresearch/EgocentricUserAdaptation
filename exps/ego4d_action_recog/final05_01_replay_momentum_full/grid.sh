#!/usr/bin/env bash

# For sgd
grid_cfg_names="SOLVER.MOMENTUM" # Split by comma
grid_overwrite_args=""

val_idx=0
gridvals=( "0.3" "0.6" "0.9" )
grid_arg="SOLVER.MOMENTUM ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

# Grid specific resources
grid_overwrite_args+=" GPU_IDS 1 NUM_USERS_PER_DEVICE 1 GRID_RESUME_LATEST True" # 0,7

# Report final
echo "grid_overwrite_args=$grid_overwrite_args"

# Run script in current dir (same process with source)
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${__dir}/ego4d_action_recog.sh" "${grid_cfg_names}" "${grid_overwrite_args}"

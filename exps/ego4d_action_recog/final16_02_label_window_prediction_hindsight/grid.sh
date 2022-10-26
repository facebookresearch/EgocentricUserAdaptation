#!/usr/bin/env bash

# For sgd
grid_cfg_names="ANALYZE_STREAM.WINDOW_SIZE_SAMPLES" # Split by comma


grid_overwrite_args=""

val_idx=4
gridvals=( "1" "4" "8" "32" "10000000" )
grid_arg="ANALYZE_STREAM.WINDOW_SIZE_SAMPLES ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

# Grid specific resources
grid_overwrite_args+=" GPU_IDS 3 NUM_USERS_PER_DEVICE 1 GRID_RESUME_LATEST False" # 1,3,4,5,6
#grid_overwrite_args+=" GPU_IDS 0 NUM_USERS_PER_DEVICE 1 GRID_RESUME_LATEST False" # 1,2,3,4,5,7

# Report final
echo "grid_overwrite_args=$grid_overwrite_args"

# Run script in current dir (same process with source)
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${__dir}/ego4d_action_recog.sh" "${grid_cfg_names}" "${grid_overwrite_args}"

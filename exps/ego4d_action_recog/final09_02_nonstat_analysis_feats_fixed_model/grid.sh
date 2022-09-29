#!/usr/bin/env bash

# For sgd
grid_cfg_names="ANALYZE_STREAM.LOOKBACK_STRIDE_ITER,ANALYZE_STREAM.WINDOW_SIZE_SAMPLES" # Split by comma


grid_overwrite_args=""

val_idx=2
gridvals=( "5" "10" "20" )
grid_arg="ANALYZE_STREAM.LOOKBACK_STRIDE_ITER ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

val_idx=3
gridvals=( "10" "20" "40" "80")
grid_arg="ANALYZE_STREAM.WINDOW_SIZE_SAMPLES ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

# Grid specific resources
grid_overwrite_args+=" GPU_IDS 0 NUM_USERS_PER_DEVICE 10 GRID_RESUME_LATEST False" # 1,2,3,4,5,7

# Report final
echo "grid_overwrite_args=$grid_overwrite_args"

# Run script in current dir (same process with source)
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${__dir}/ego4d_action_recog.sh" "${grid_cfg_names}" "${grid_overwrite_args}"

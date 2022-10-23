#!/usr/bin/env bash

grid_cfg_names="METHOD.REPLAY.MEMORY_SIZE_SAMPLES,METHOD.REPLAY.STORAGE_POLICY" # Split by comma

val_idx=1
gridvals=("reservoir_stream" "reservoir_action" "window")
grid_arg="METHOD.REPLAY.STORAGE_POLICY ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

val_idx=0
gridvals=( 64 1000000 )
grid_arg="METHOD.REPLAY.MEMORY_SIZE_SAMPLES ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

grid_overwrite_args+=" GPU_IDS 0,1,2,3,4,5 NUM_USERS_PER_DEVICE 1 GRID_RESUME_LATEST True" # 0,7
echo "grid_overwrite_args=$grid_overwrite_args"

# Run script in current dir (same process with source)
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${__dir}/ego4d_action_recog.sh" "${grid_cfg_names}" "${grid_overwrite_args}"

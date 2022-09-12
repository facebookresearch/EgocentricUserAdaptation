#!/usr/bin/env bash


#BASE_LR: 1e-4
#LR_POLICY: constant
#MAX_EPOCH: 1
#MOMENTUM: 0.9
#OPTIMIZING_METHOD: sgd

# For sgd
grid_cfg_names="SOLVER.BASE_LR" # Split by comma
grid_overwrite_args=""

val_idx=3
gridvals=( "1e-1" "1e-2" "1e-3" "1e-4")
grid_arg="SOLVER.BASE_LR ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

# Grid specific resources
grid_overwrite_args+=" GPU_IDS 4,5 NUM_USERS_PER_DEVICE 1" # 0,7,5 # 1 crashed

# Report final
echo "grid_overwrite_args=$grid_overwrite_args"

# Run script in current dir (same process with source)
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${__dir}/ego4d_action_recog.sh" "${grid_cfg_names}" "${grid_overwrite_args}"

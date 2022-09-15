#!/usr/bin/env bash


#BASE_LR: 1e-4
#LR_POLICY: constant
#MAX_EPOCH: 1
#MOMENTUM: 0.9
#OPTIMIZING_METHOD: sgd

# For sgd
grid_cfg_names="SOLVER.BASE_LR,SOLVER.MOMENTUM,SOLVER.NESTEROV" # Split by comma
grid_overwrite_args=""

val_idx=1 # TODO RUN
gridvals=( "1e-1" "1e-2" "1e-3")
grid_arg="SOLVER.BASE_LR ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

val_idx=2
gridvals=( "0." "0.3" "0.6" "0.9" ) 
grid_arg="SOLVER.MOMENTUM ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

val_idx=1
gridvals=( True False )
grid_arg="SOLVER.NESTEROV ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"


# Grid specific resources
grid_overwrite_args+=" GPU_IDS 7 NUM_USERS_PER_DEVICE 2" # GRID_RESUME_LATEST False" # 0,7

# Report final
echo "grid_overwrite_args=$grid_overwrite_args"

# Run script in current dir (same process with source)
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${__dir}/ego4d_action_recog.sh" "${grid_cfg_names}" "${grid_overwrite_args}"

#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Change dir to current script
this_script_dirpath=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
CONFIG="$this_script_dirpath/cfg.yaml"
this_script_filepath="${this_script_dirpath}/$(basename "${BASH_SOURCE[0]}")"
grid_overwrite_args="--cfg "${CONFIG}" --parent_script "${this_script_filepath}

# 1 vs 10 iterations
grid_overwrite_args+=" TRAIN.INNER_LOOP_ITERS 1"

# For gridsearch
grid_cfg_names="SOLVER.BASE_LR,SOLVER.MOMENTUM,SOLVER.NESTEROV" # Split by comma
grid_overwrite_args+=" GRID_NODES ${grid_cfg_names}"

val_idx=0
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

# Checkpoint loading
PRETRAIN_MODEL="/YOUR/PATH/TO/pretrain_148usersplit_incl_nan/checkpoints/best_model.ckpt" # Our user-pretrained model
grid_overwrite_args+=" CHECKPOINT_FILE_PATH ${PRETRAIN_MODEL}"
grid_overwrite_args+=" CHECKPOINT_LOAD_MODEL_HEAD True" # Load population head
grid_overwrite_args+=" MODEL.FREEZE_BACKBONE False"     # Don't freeze

# Report final
echo "grid_overwrite_args=$grid_overwrite_args"

# Run script in current dir (same process with source)
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${__dir}/../../call_train_user_streams.sh" "${grid_overwrite_args}"

#!/usr/bin/env bash

# Change dir to current script
this_script_dirpath=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
CONFIG="$this_script_dirpath/cfg.yaml"
this_script_filepath="${this_script_dirpath}/$(basename "${BASH_SOURCE[0]}")"
grid_overwrite_args="--cfg "${CONFIG}" --parent_script "${this_script_filepath}

# For gridsearch
grid_cfg_names="METHOD.REPLAY.MEMORY_SIZE_SAMPLES,METHOD.REPLAY.STORAGE_POLICY" # Split by comma
grid_overwrite_args+=" GRID_NODES ${grid_cfg_names}"

val_idx=0
gridvals=("FIFO" "reservoir" "hybrid_CBRS")
grid_arg="METHOD.REPLAY.STORAGE_POLICY ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

val_idx=0
gridvals=( 8 64 128 1000000 ) # Select last for Full-replay, with any storage policy
grid_arg="METHOD.REPLAY.MEMORY_SIZE_SAMPLES ${gridvals[${val_idx}]}"
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

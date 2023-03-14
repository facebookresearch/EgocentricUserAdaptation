#!/usr/bin/env bash
# Set DATA.USER_SUBSET to 'train' once, for U_train results on pretrain model and
# to 'test' for U_test results on the pretrain model.


# Change dir to current script
this_script_dirpath=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
CONFIG="$this_script_dirpath/cfg.yaml"
this_script_filepath="${this_script_dirpath}/$(basename "${BASH_SOURCE[0]}")"
grid_overwrite_args="--cfg "${CONFIG}" --parent_script "${this_script_filepath}

# For gridsearch
grid_cfg_names="SOLVER.BASE_LR" # Split by comma
grid_overwrite_args+=" GRID_NODES ${grid_cfg_names}"

val_idx=0
gridvals=("1e-1" "1e-2" "1e-3")
grid_arg="SOLVER.BASE_LR ${gridvals[${val_idx}]}"
grid_overwrite_args+=" ${grid_arg}"

# Checkpoint loading
PRETRAIN_MODEL="/YOUR/PATH/TO/pretrain_148usersplit_incl_nan/checkpoints/best_model.ckpt" # Our user-pretrained model
grid_overwrite_args+=" CHECKPOINT_FILE_PATH ${PRETRAIN_MODEL}"

# Freeze full model
grid_overwrite_args+=" CHECKPOINT_LOAD_MODEL_HEAD True" # Always load population head
grid_overwrite_args+=" MODEL.FREEZE_MODEL True"

# Report final
echo "grid_overwrite_args=$grid_overwrite_args"

# Run script in current dir (same process with source)
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${__dir}/../../call_train_user_streams.sh" "${grid_overwrite_args}"

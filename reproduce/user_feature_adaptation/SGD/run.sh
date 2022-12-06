#!/usr/bin/env bash

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

# FREEZING CONFIG
grid_overwrite_args+=" CHECKPOINT_LOAD_MODEL_HEAD True" # Always load population head

# Freeze feature extractor only: F
grid_overwrite_args+=" MODEL.FREEZE_BACKBONE True" # Learn features
grid_overwrite_args+=" MODEL.FREEZE_HEAD False"    # Learn head

# Freeze head only: H
#grid_overwrite_args+=" MODEL.FREEZE_BACKBONE False"          # Learn features
#grid_overwrite_args+=" MODEL.FREEZE_HEAD True"          # Learn head

# Report final
echo "grid_overwrite_args=$grid_overwrite_args"

# Run script in current dir (same process with source)
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${__dir}/../../main_script.sh" "${grid_overwrite_args}"

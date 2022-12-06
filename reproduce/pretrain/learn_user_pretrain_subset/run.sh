#!/usr/bin/env bash

# Change dir to current script
this_script_dirpath=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
CONFIG="$this_script_dirpath/cfg.yaml"
this_script_filepath="${this_script_dirpath}/$(basename "${BASH_SOURCE[0]}")"
grid_overwrite_args="--cfg "${CONFIG}" --parent_script "${this_script_filepath}

# Checkpoint loading
BACKBONE_WTS="/ego4d_models/long_term_anticipation/kinetics_slowfast8x8.ckpt" # Download from original ego4d repo
OVERWRITE_CFG_ARGS+=" DATA.CHECKPOINT_MODULE_FILE_PATH ${BACKBONE_WTS}"       # Start from Kinetics model
grid_overwrite_args+=" CHECKPOINT_LOAD_MODEL_HEAD False"
grid_overwrite_args+=" MODEL.FREEZE_BACKBONE False"

# Resources
grid_overwrite_args+=" NUM_GPUS 8" # 8 x 28G

# Report final
echo "grid_overwrite_args=$grid_overwrite_args"

# Run script in current dir (same process with source)
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${__dir}/../../call_pretrain.sh" "${grid_overwrite_args}"

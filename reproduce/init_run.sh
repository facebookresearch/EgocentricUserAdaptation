#!/usr/bin/env bash
# This script adds modules to path for execution and adds run-specific arguments:
# e.g. run-specific unique output dir of experiment, and run-specific WandB tags based on parent directory names.

#-----------------------------------------------------------------------------------------------#
# Add ego4d code modules to pythonpath
run_script_dirpath=$(pwd) # Change location to current script
root_path=${run_script_dirpath}/../../..
ego4d_code_root="$root_path/src"
export PYTHONPATH=$PYTHONPATH:$ego4d_code_root

# Unique run id
timestamp=$(date '+%Y-%m-%d_%H-%M-%S')
uuid=$(uuidgen)
RUN_ID="${timestamp}_UID${uuid}"
echo "RUN-ID=${RUN_ID}"

#-----------------------------------------------------------------------------------------------#
# PATHS
#-----------------------------------------------------------------------------------------------#
# Logging (stdout/tensorboard) output path
p_dirname="$(basename "${run_script_dirpath}")"
pp_dirname="$(basename "$(dirname -- "${run_script_dirpath}")")"
OUTPUT_DIR="$root_path/results/${pp_dirname}/${p_dirname}/logs/${RUN_ID}"

#-----------------------------------------------------------------------------------------------#
# CONFIG (Overwrite with args)
#-----------------------------------------------------------------------------------------------#
# Add from calling script
OVERWRITE_CFG_ARGS="$@"

# DEBUG
#OVERWRITE_CFG_ARGS+=" DATA_LOADER.NUM_WORKERS 10" # Workers per dataloader (i.e. per user process)
#OVERWRITE_CFG_ARGS+=" USER_SELECTION 104,108,324,30" # Subset of users to process
#OVERWRITE_CFG_ARGS+=" FAST_DEV_RUN True FAST_DEV_DATA_CUTOFF 30" # Only process small part of stream

# RESOURCES
#OVERWRITE_CFG_ARGS+=" GPU_IDS 1,2,3,4,5 NUM_USERS_PER_DEVICE 2"

# Logger tags
OVERWRITE_CFG_ARGS+=" WANDB.TAGS '${p_dirname}','${pp_dirname}'"

# Paths
OVERWRITE_CFG_ARGS+=" OUTPUT_DIR ${OUTPUT_DIR}"

export OVERWRITE_CFG_ARGS
export RUN_UID
export OUTPUT_DIR

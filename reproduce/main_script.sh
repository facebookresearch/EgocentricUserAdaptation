#!/usr/bin/env bash
echo "PWD="$(pwd)

#-----------------------------------------------------------------------------------------------#
# Add ego4d code modules to pythonpath
run_script_dirpath=$(pwd) # Change location to current script
root_path=${run_script_dirpath}/../../..
ego4d_code_root="$root_path/src"
export PYTHONPATH=$PYTHONPATH:$ego4d_code_root
echo "run_script_dirpath=${run_script_dirpath}"
echo "ego4d_code_root=${ego4d_code_root}"

# Unique run id
timestamp=$(date '+%Y-%m-%d_%H-%M-%S')
uuid=$(uuidgen)
run_id="${timestamp}_UID${uuid}"
echo "RUN-ID=${run_id}"

#-----------------------------------------------------------------------------------------------#
# PATHS
#-----------------------------------------------------------------------------------------------#
# Logging (stdout/tensorboard) output path
p_dirname="$(basename "${run_script_dirpath}")"
pp_dirname="$(basename "$(dirname -- "${run_script_dirpath}")")"
OUTPUT_DIR="$root_path/results/${pp_dirname}/${p_dirname}/logs/${run_id}"

# Data paths
EGO4D_ANNOTS=$ego4d_code_root/data/long_term_anticipation/annotations_local/
EGO4D_VIDEOS=$ego4d_code_root/data/long_term_anticipation/clips_root_local/clips

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
OVERWRITE_CFG_ARGS+=" DATA.PATH_TO_DATA_DIR ${EGO4D_ANNOTS}"
OVERWRITE_CFG_ARGS+=" DATA.PATH_PREFIX ${EGO4D_VIDEOS}"
OVERWRITE_CFG_ARGS+=" OUTPUT_DIR ${OUTPUT_DIR}"

# Start in screen detached mode (-dm), and give indicative name via (-S)
#screenname="${run_id}"
#screen -dmS "${screenname}" \
python -m continual_ego4d.run_train_user_streams \
  --job_name "$run_id" \
  --working_directory "${OUTPUT_DIR}" \
  ${OVERWRITE_CFG_ARGS}

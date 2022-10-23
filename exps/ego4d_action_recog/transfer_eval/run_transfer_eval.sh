#!/usr/bin/env bash
# Provide checkpoint path as argument if you want to resume

#-----------------------------------------------------------------------------------------------#
# Add ego4d code modules to pythonpath
this_script_dirpath=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) # Change location to current script
root_path=${this_script_dirpath}/../../../                                    # One dir up
ego4d_code_root="$root_path/forecasting"
export PYTHONPATH=$PYTHONPATH:$ego4d_code_root

# Unique run id
timestamp=$(date '+%Y-%m-%d_%H-%M-%S')
uuid=$(uuidgen)
run_id="${timestamp}_UID${uuid}"
echo "RUN-ID=${run_id}"
#-----------------------------------------------------------------------------------------------#
# PATHS
#-----------------------------------------------------------------------------------------------#
CONFIG="$this_script_dirpath/TRANSFER_EVAL_CONFIG.yaml"
this_script_filepath="${this_script_dirpath}/$(basename "${BASH_SOURCE[0]}")"

# Logging (stdout/tensorboard) output path
p_dirname="$(basename "${this_script_dirpath}")"
pp_dirname="$(basename "$(dirname -- "${this_script_dirpath}")")"
#OUTPUT_DIR="$root_path/results/${pp_dirname}/${p_dirname}/logs/${run_id}" # Alternative:/home/matthiasdelange/data/ego4d/continual_ego4d_pretrained_models_usersplit

# Data paths
#EGO4D_ANNOTS=$ego4d_code_root/data/long_term_anticipation/annotations_local/
#EGO4D_VIDEOS=$ego4d_code_root/data/long_term_anticipation/clips_root_local/clips

#-----------------------------------------------------------------------------------------------#
# CONFIG (Overwrite with args)
#-----------------------------------------------------------------------------------------------#
OVERWRITE_CFG_ARGS="WANDB.TAGS '${p_dirname}','${pp_dirname}'"

#-----------------------------------------------------------------------------------------------#
# Get GRID PARAMS
if [[ $# -gt 0 ]]; then

  GRID_CFG_NAMES=$1
  echo "GRID_CFG_NAMES=${GRID_CFG_NAMES}"
  OVERWRITE_CFG_ARGS+=" GRID_NODES ${GRID_CFG_NAMES}"

  GRID_OVERWRITE_CFG_ARGS=$2
  echo "GRID_OVERWRITE_CFG_ARGS=${GRID_OVERWRITE_CFG_ARGS}"
  OVERWRITE_CFG_ARGS+="${GRID_OVERWRITE_CFG_ARGS}"

  echo "OVERWRITE_CFG_ARGS from grid=${OVERWRITE_CFG_ARGS}"

fi
#-----------------------------------------------------------------------------------------------#
OVERWRITE_CFG_ARGS+=" GPU_IDS 1,2,3,4,5" #5G per run with BS 10
#OVERWRITE_CFG_ARGS+=" GPU_IDS wandb_export_2022-09-21T15_36_21.953-07_00.csv" #5G per run with BS 10

#OVERWRITE_CFG_ARGS+=" TRANSFER_EVAL.NUM_EXPECTED_USERS 10 GPU_IDS 1 FAST_DEV_RUN True FAST_DEV_DATA_CUTOFF 5 PREDICT_PHASE.BATCH_SIZE 2 NUM_USERS_PER_DEVICE 5" # DEBUG

# Start in screen detached mode (-dm), and give indicative name via (-S)
screenname="${run_id}_MATT"
screen -dmS "${screenname}" \
python -m continual_ego4d.run_transfer_eval \
  --cfg "${CONFIG}" \
  ${OVERWRITE_CFG_ARGS}
#  --job_name "$run_id" \
#  --working_directory "${OUTPUT_DIR}" \
#  --parent_script "${this_script_filepath}" \

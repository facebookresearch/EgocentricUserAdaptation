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
this_script_filepath="${this_script_dirpath}/$(basename "${BASH_SOURCE[0]}")"

# Logging (stdout/tensorboard) output path
p_dirname="$(basename "${this_script_dirpath}")"
pp_dirname="$(basename "$(dirname -- "${this_script_dirpath}")")"
OUTPUT_DIR="$root_path/results/${pp_dirname}/${p_dirname}/logs/${run_id}" # Alternative:/home/matthiasdelange/data/ego4d/continual_ego4d_pretrained_models_usersplit

# Data paths
EGO4D_ANNOTS=$ego4d_code_root/data/long_term_anticipation/annotations_local/
EGO4D_VIDEOS=$ego4d_code_root/data/long_term_anticipation/clips_root_local/clips

#-----------------------------------------------------------------------------------------------#
# CONFIG (Overwrite with args)
#-----------------------------------------------------------------------------------------------#

# All that matters for this script is to define which config (data paths in config) to summarize
CONFIG="$this_script_dirpath/MULTISLOWFAST_8x8_R101.yaml"

OVERWRITE_CFG_ARGS="WANDB.TAGS '${p_dirname}','${pp_dirname}'"
OVERWRITE_CFG_ARGS+=" DATA_LOADER.NUM_WORKERS 8" # Workers per dataloader (i.e. per user process)

# Resume previous run (use ckpts)
#OVERWRITE_CFG_ARGS+=" RESUME_OUTPUT_DIR /home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/exps/ego4d_action_recog/summarize_streams/../../..//results/ego4d_action_recog/summarize_streams/logs/2022-08-04_18-05-40_UID8f0427cb-8048-47d5-b70e-bc3878a1cb3a"

# TODO: Define your paths here to summarize
#OVERWRITE_CFG_ARGS+=" DATA.USER_SUBSET 'train'" # Check train or test set
#OVERWRITE_CFG_ARGS+=" DATA.PATH_TO_DATA_SPLIT_JSON.TRAIN_SPLIT '/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/forecasting/continual_ego4d/usersplit_data/2022-07-27_21-05-14_ego4d_LTA_usersplit/ego4d_LTA_train_usersplit_10users.json'"
#OVERWRITE_CFG_ARGS+=" DATA.PATH_TO_DATA_SPLIT_JSON.TEST_SPLIT '/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/forecasting/continual_ego4d/usersplit_data/2022-07-27_21-05-14_ego4d_LTA_usersplit/ego4d_LTA_test_usersplit_40users.json'"

#OVERWRITE_CFG_ARGS+=" DATA.CHECKPOINT_MODULE_FILE_PATH ${BACKBONE_WTS}" # Start from Kinetics model
OVERWRITE_CFG_ARGS+=" CHECKPOINT_FILE_PATH ''"         # Start from Kinetics model
OVERWRITE_CFG_ARGS+=" CHECKPOINT_LOAD_MODEL_HEAD True" # Load population head
OVERWRITE_CFG_ARGS+=" MODEL.FREEZE_BACKBONE True"      # Learn features as well

# Paths
OVERWRITE_CFG_ARGS+=" DATA.PATH_TO_DATA_DIR ${EGO4D_ANNOTS}"
OVERWRITE_CFG_ARGS+=" DATA.PATH_PREFIX ${EGO4D_VIDEOS}"
OVERWRITE_CFG_ARGS+=" OUTPUT_DIR ${OUTPUT_DIR}"

# Start in screen detached mode (-dm), and give indicative name via (-S)
screenname="${run_id}_MATT"
#screen -dmS "${screenname}" \
  python -m continual_ego4d.run_summarize_streams \
  --job_name "$screenname" \
  --working_directory "${OUTPUT_DIR}" \
  --cfg "${CONFIG}" \
  --parent_script "${this_script_filepath}" \
  ${OVERWRITE_CFG_ARGS}

#-----------------------------------------------------------------------------------------------#
#                                    OTHER OPTIONS                                              #
#-----------------------------------------------------------------------------------------------#
# # MViT
# BACKBONE_WTS=$PWD/pretrained_models/long_term_anticipation/kinetics_mvit16x4.ckpt
# run mvit \
#     configs/Ego4dRecognition/MULTIMVIT_16x4.yaml \
#     DATA.CHECKPOINT_MODULE_FILE_PATH ${BACKBONE_WTS}

# # Debug locally using a smaller batch size / fewer GPUs
# CLUSTER_ARGS="NUM_GPUS 2 TRAIN.BATCH_SIZE 8 TEST.BATCH_SIZE 32"

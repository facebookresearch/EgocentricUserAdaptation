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
CONFIG="$ego4d_code_root/continual_ego4d/configs/Ego4dContinualActionRecog/pretrain/MULTISLOWFAST_8x8_R101.yaml"
this_script_filepath="${this_script_dirpath}/$(basename "${BASH_SOURCE[0]}")"

# Logging (stdout/tensorboard) output path
p_dirname="$(basename "${this_script_dirpath}")"
pp_dirname="$(basename "$(dirname -- "${this_script_dirpath}")")"
OUTPUT_DIR="$root_path/results/${pp_dirname}/${p_dirname}/logs/${run_id}" # Alternative:/home/matthiasdelange/data/ego4d/continual_ego4d_pretrained_models_usersplit
mkdir -p "${OUTPUT_DIR}"
cp "${CONFIG}" "${OUTPUT_DIR}"               # Make a copy of the config file (if we want to rerun later)
cp "${this_script_filepath}" "${OUTPUT_DIR}" # Make a copy of current script file (if we want to rerun later)

#-----------------------------------------------------------------------------------------------#
# CONFIG (Overwrite with args)
#-----------------------------------------------------------------------------------------------#
#export CUDA_VISIBLE_DEVICES="1,2,3,4,5" # Set as environment variable for this script

OVERWRITE_CFG_ARGS=""
OVERWRITE_CFG_ARGS+=" NUM_GPUS 8"
#OVERWRITE_CFG_ARGS+=" GPU_IDS '6,7'" # Not compatible with DDP
OVERWRITE_CFG_ARGS+=" FAST_DEV_RUN False"

# RESUME
#OVERWRITE_CFG_ARGS+=" RESUME_OUTPUT_DIR /home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/exps/ego4d_action_recog/test_exp_00/logs/2022-07-26_14-33-36_UIDd8a02398-ec4c-4d49-b81b-2d357c63bcf4"

# Checkpoint loading
# START FROM KINETICS400 pretrained model
BACKBONE_WTS="/home/matthiasdelange/data/ego4d/ego4d_pretrained_models/pretrained_models/long_term_anticipation/kinetics_slowfast8x8.ckpt"
OVERWRITE_CFG_ARGS+=" DATA.CHECKPOINT_MODULE_FILE_PATH ${BACKBONE_WTS}" # Start from Kinetics model (Direclty state-dict)
#OVERWRITE_CFG_ARGS+=" CHECKPOINT_FILE_PATH ${BACKBONE_WTS}" # Start from a Lightning checkpoint
OVERWRITE_CFG_ARGS+=" CHECKPOINT_LOAD_MODEL_HEAD False"
OVERWRITE_CFG_ARGS+=" MODEL.FREEZE_BACKBONE False"

# Data paths
#EGO4D_ANNOTS="/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/forecasting/continual_ego4d/usersplit_data/2022-07-27_18-05-01_ego4d_LTA_usersplit/ego4d_LTA_pretrain_usersplit_147users.json"
EGO4D_VIDEOS=$ego4d_code_root/data/long_term_anticipation/clips_root/clips
#OVERWRITE_CFG_ARGS+=" DATA.PATH_TO_DATA_FILE.TRAIN ${EGO4D_ANNOTS}"
OVERWRITE_CFG_ARGS+=" DATA.PATH_PREFIX ${EGO4D_VIDEOS}"
OVERWRITE_CFG_ARGS+=" OUTPUT_DIR ${OUTPUT_DIR}"

# Start in screen detached mode (-dm), and give indicative name via (-S), \
# reattach with screen -r, list session with screen -ls
# To detach again: ctrl+a, followed by ctrl+d
screenname="${run_id}_MATT"
screen -dmS "${screenname}" \
  python -m scripts.run_lta \
  --job_name "$screenname" \
  --working_directory "${OUTPUT_DIR}" \
  --cfg "${CONFIG}" \
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
# LOL

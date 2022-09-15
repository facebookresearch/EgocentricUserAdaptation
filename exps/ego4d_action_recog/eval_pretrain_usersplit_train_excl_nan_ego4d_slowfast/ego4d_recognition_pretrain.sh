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
CONFIG="$this_script_dirpath/MULTISLOWFAST_8x8_R101.yaml"
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

OVERWRITE_CFG_ARGS="WANDB.TAGS '${p_dirname}','${pp_dirname}'"
OVERWRITE_CFG_ARGS+=" NUM_GPUS 1" # For solid benchmarking
export CUDA_VISIBLE_DEVICES="3" # Set as environment variable for this script

OVERWRITE_CFG_ARGS+=" FAST_DEV_RUN False"
OVERWRITE_CFG_ARGS+=" DATA_LOADER.NUM_WORKERS 8"

# RESUME
#OVERWRITE_CFG_ARGS+=" RESUME_OUTPUT_DIR /home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/exps/ego4d_action_recog/test_exp_00/logs/2022-07-26_14-33-36_UIDd8a02398-ec4c-4d49-b81b-2d357c63bcf4"

# Checkpoint loading
# START FROM KINETICS400 pretrained model
#BACKBONE_WTS="/fb-agios-acai-efs/mattdl/ego4d_models/ego4d_pretrained_models/pretrained_models/long_term_anticipation/kinetics_slowfast8x8.ckpt"
#BACKBONE_WTS="/fb-agios-acai-efs/mattdl/ego4d_models/ego4d_pretrained_models/pretrained_models/long_term_anticipation/ego4d_slowfast8x8.ckpt"
BACKBONE_WTS="/fb-agios-acai-efs/mattdl/ego4d_models/continual_ego4d_pretrained_models_usersplit/pretrain_147usersplit_excl_nan/2022-07-28_17-06-20_UIDe499d926-a3ff-4a28-9632-7d01054644fe/lightning_logs/version_0/checkpoints/last.ckpt"
#OVERWRITE_CFG_ARGS+=" DATA.CHECKPOINT_MODULE_FILE_PATH ${BACKBONE_WTS}" # IMPORTANT NOT TO LOAD FROM THIS, THIS WONT LOAD THE HEAD
OVERWRITE_CFG_ARGS+=" CHECKPOINT_FILE_PATH ${BACKBONE_WTS}" # Start from a Lightning checkpoint
OVERWRITE_CFG_ARGS+=" CHECKPOINT_LOAD_MODEL_HEAD True"
OVERWRITE_CFG_ARGS+=" MODEL.FREEZE_BACKBONE True"

# Data paths
#EGO4D_ANNOTS="/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/forecasting/continual_ego4d/usersplit_data/2022-07-27_18-05-01_ego4d_LTA_usersplit/ego4d_LTA_pretrain_usersplit_147users.json"
EGO4D_VIDEOS=$ego4d_code_root/data/long_term_anticipation/clips_root_local/clips
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

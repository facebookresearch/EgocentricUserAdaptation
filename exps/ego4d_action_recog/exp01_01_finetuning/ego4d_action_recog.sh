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

# Data paths
EGO4D_ANNOTS=$ego4d_code_root/data/long_term_anticipation/annotations/
EGO4D_VIDEOS=$ego4d_code_root/data/long_term_anticipation/clips_root/clips

# Checkpoint path (Make unique)
#if [ $# -eq 0 ]; then # Default checkpoint path is based on 2 parent dirs + Unique id
#  parent_parent_dir="$(basename "$(dirname "$this_script_dirpath")")"
#  parent_dir="$(basename "$this_script_dirpath")"
#  CHECKPOINT_DIR="${root_path}/checkpoints/${parent_parent_dir}/${parent_dir}/${run_id}"
#else # When RESUMING
#  CHECKPOINT_DIR=$1
#fi
#mkdir -p "$CHECKPOINT_DIR"
#echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"

#-----------------------------------------------------------------------------------------------#
# CONFIG (Overwrite with args)
#-----------------------------------------------------------------------------------------------#
OVERWRITE_CFG_ARGS=""
OVERWRITE_CFG_ARGS+=" DATA_LOADER.NUM_WORKERS 8" # Workers per dataloader (i.e. per user process)
OVERWRITE_CFG_ARGS+=" GPU_IDS '1,3,4,5,6'"
#OVERWRITE_CFG_ARGS+=" DATA_LOADER.NUM_WORKERS 0 TRAIN.BATCH_SIZE 10 TRAIN.CONTINUAL_EVAL_BATCH_SIZE 16 CHECKPOINT_step_freq 300" # DEBUG
#OVERWRITE_CFG_ARGS+=" FAST_DEV_RUN True FAST_DEV_DATA_CUTOFF 30" # DEBUG

# RESUME
OVERWRITE_CFG_ARGS+=" RESUME_OUTPUT_DIR /home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/test_exp_00/logs/2022-08-02_10-46-07_UIDed095550-f431-4c1e-ae62-b072fc4c9a87"

# Architecture: aggregator/decoder only for LTA, SlowFast model directly performs Action Classification
#OVERWRITE_CFG_ARGS+=" FORECASTING.AGGREGATOR TransformerAggregator"
#OVERWRITE_CFG_ARGS+=" FORECASTING.DECODER MultiHeadDecoder"

# Checkpoint loading
#BACKBONE_WTS="/home/matthiasdelange/data/ego4d/ego4d_pretrained_models/pretrained_models/long_term_anticipation/k400_slowfast8x8.ckpt"
#BACKBONE_WTS="/home/matthiasdelange/data/ego4d/ego4d_pretrained_models/pretrained_models/long_term_anticipation/ego4d_slowfast8x8.ckpt" # Tmp debug with ego4d
BACKBONE_WTS="/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/pretrain_slowfast/logs/2022-07-28_17-06-20_UIDe499d926-a3ff-4a28-9632-7d01054644fe/lightning_logs/version_0/checkpoints/last.ckpt"

#OVERWRITE_CFG_ARGS+=" DATA.CHECKPOINT_MODULE_FILE_PATH ${BACKBONE_WTS}" # Start from Kinetics model
OVERWRITE_CFG_ARGS+=" CHECKPOINT_FILE_PATH ${BACKBONE_WTS}" # Start from Kinetics model
OVERWRITE_CFG_ARGS+=" CHECKPOINT_LOAD_MODEL_HEAD True"      # Load population head
OVERWRITE_CFG_ARGS+=" MODEL.FREEZE_BACKBONE False"          # Learn features as well

# Paths
OVERWRITE_CFG_ARGS+=" DATA.PATH_TO_DATA_DIR ${EGO4D_ANNOTS}"
OVERWRITE_CFG_ARGS+=" DATA.PATH_PREFIX ${EGO4D_VIDEOS}"
OVERWRITE_CFG_ARGS+=" OUTPUT_DIR ${OUTPUT_DIR}"

# Start in screen detached mode (-dm), and give indicative name via (-S)
screenname="${run_id}_MATT"
screen -dmS "${screenname}" \
python -m continual_ego4d.run_recog_CL \
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

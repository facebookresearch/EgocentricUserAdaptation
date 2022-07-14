#!/usr/bin/env bash
# Provide checkpoint path as argument if you want to resume


#-----------------------------------------------------------------------------------------------#
# Add ego4d code modules to pythonpath
this_script_path=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) # Change location to current script
root_path=${this_script_path}/../../../ # One dir up
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
BACKBONE_WTS="/home/matthiasdelange/data/ego4d/ego4d_pretrained_models/pretrained_models/long_term_anticipation/k400_slowfast8x8.ckpt"
CONFIG="$ego4d_code_root/continual_ego4d/configs/Ego4dContinualActionRecog/MULTISLOWFAST_8x8_R101.yaml"

# Logging (stdout/tensorboard) output path
OUTPUT_DIR="./logs/${run_id}"
mkdir -p "${OUTPUT_DIR}"
cp "${CONFIG}" "${OUTPUT_DIR}" # Make a copy of the config file (if we want to rerun later)

# Data paths
EGO4D_ANNOTS=$ego4d_code_root/data/long_term_anticipation/annotations/
EGO4D_VIDEOS=$ego4d_code_root/data/long_term_anticipation/clips_root/resized_clips

# Checkpoint path (Make unique)
if [ $# -eq 0 ]; then # Default checkpoint path is based on 2 parent dirs + Unique id
    parent_parent_dir="$(basename "$(dirname "$this_script_path")")"
    parent_dir="$(basename "$this_script_path")"
    CHECKPOINT_DIR="${root_path}/checkpoints/${parent_parent_dir}/${parent_dir}/${run_id}"
else # When RESUMING
  CHECKPOINT_DIR=$1
fi
mkdir -p "$CHECKPOINT_DIR"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"

#-----------------------------------------------------------------------------------------------#
# CONFIG (Overwrite with args)
#-----------------------------------------------------------------------------------------------#
OVERWRITE_CFG_ARGS="NUM_GPUS 2 TRAIN.BATCH_SIZE 8 TRAIN.CONTINUAL_EVAL_BATCH_SIZE 32 CHECKPOINT_step_freq 300" # DEBUG
#OVERWRITE_CFG_ARGS+=" FORECASTING.NUM_INPUT_CLIPS 4"

# Architecture: aggregator/decoder only for LTA, SlowFast model directly performs Action Classification
#OVERWRITE_CFG_ARGS+=" FORECASTING.AGGREGATOR TransformerAggregator"
#OVERWRITE_CFG_ARGS+=" FORECASTING.DECODER MultiHeadDecoder"

# Checkpoint loading
#OVERWRITE_CFG_ARGS+=" DATA.CHECKPOINT_MODULE_FILE_PATH ${BACKBONE_WTS}" # Start from Kinetics model
OVERWRITE_CFG_ARGS+=" CHECKPOINT_FILE_PATH ${BACKBONE_WTS}" # Start from Kinetics model
OVERWRITE_CFG_ARGS+=" CHECKPOINT_LOAD_MODEL_HEAD False"
OVERWRITE_CFG_ARGS+=" MODEL.FREEZE_BACKBONE False"

# Paths
#OVERWRITE_CFG_ARGS+=" DATA.PATH_TO_DATA_DIR ${EGO4D_ANNOTS}"
#OVERWRITE_CFG_ARGS+=" DATA.PATH_PREFIX ${EGO4D_VIDEOS}"
OVERWRITE_CFG_ARGS+=" OUTPUT_DIR ${OUTPUT_DIR}"

# Start in screen detached mode (-dm), and give indicative name via (-S)
screenname="MATT_${run_id}"
#screen -dmS "${screenname}" \
python -m continual_ego4d.run_lta_CL \
      --job_name "$screenname" \
      --working_directory "${CHECKPOINT_DIR}" \
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


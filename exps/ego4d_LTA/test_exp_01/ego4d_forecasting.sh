function run(){

  # Only runs on cluster/slurm using arg --on_cluster
  python -m scripts.run_lta \
    --job_name $JOB_NAME \
    --working_directory ${CHECKPOINT_DIR} \
    --cfg $CONFIG \
    ${ARGS} \
    DATA.PATH_TO_DATA_DIR ${EGO4D_ANNOTS} \
    DATA.PATH_PREFIX ${EGO4D_VIDEOS} \
    CHECKPOINT_LOAD_MODEL_HEAD False \
    MODEL.FREEZE_BACKBONE True \
    DATA.CHECKPOINT_MODULE_FILE_PATH "" \
    FORECASTING.AGGREGATOR "" \
    FORECASTING.DECODER "" \
    "$@"
}


#-----------------------------------------------------------------------------------------------#
# Add ego4d code modules to pythonpath
this_script_path=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) # Change location to current script
root_path=${this_script_path}/../../../ # One dir up
ego4d_code_root="$root_path/forecasting"
export PYTHONPATH=$PYTHONPATH:$ego4d_code_root

#-----------------------------------------------------------------------------------------------#
# CONFIG
#-----------------------------------------------------------------------------------------------#
# Add any arguments here
JOB_NAME="slowfast_trf" #
ARGS="NUM_GPUS 2 TRAIN.BATCH_SIZE 8 TEST.BATCH_SIZE 32" # DEBUG
CONFIG="$ego4d_code_root/configs/Ego4dLTA/MULTISLOWFAST_8x8_R101.yaml"
BACKBONE_WTS="/home/matthiasdelange/data/ego4d/ego4d_pretrained_models/pretrained_models/long_term_anticipation/ego4d_slowfast8x8.ckpt"
#$PWD/pretrained_models/long_term_anticipation/ego4d_slowfast8x8.ckpt # SlowFast-Transformer

# Data paths
EGO4D_ANNOTS=$ego4d_code_root/data/long_term_anticipation/annotations/
EGO4D_VIDEOS=$ego4d_code_root/data/long_term_anticipation/clips_root/resized_clips

# Checkpoint path
if [ $# -eq 0 ]; then # Default checkpoint path is based on 2 parent dirs
    parent_dir="$(basename "$(dirname "$this_script_path")")"
    parent_parent_dir="$(basename "$(dirname "$(realpath "$this_script_path/../")")")"
    CHECKPOINT_DIR="./CHECKPOINTS/$parent_dir/$parent_parent_dir"
else
  CHECKPOINT_DIR=$1
fi
mkdir -p "$CHECKPOINT_DIR"

run FORECASTING.AGGREGATOR TransformerAggregator \
    FORECASTING.DECODER MultiHeadDecoder \
    FORECASTING.NUM_INPUT_CLIPS 4 \
    DATA.CHECKPOINT_MODULE_FILE_PATH ${BACKBONE_WTS}

#-----------------------------------------------------------------------------------------------#
#                                    OTHER OPTIONS                                              #
#-----------------------------------------------------------------------------------------------#

# # SlowFast-Concat
# BACKBONE_WTS=$PWD/pretrained_models/long_term_anticipation/ego4d_slowfast8x8.ckpt
# run slowfast_concat \
#     configs/Ego4dLTA/MULTISLOWFAST_8x8_R101.yaml \
#     FORECASTING.AGGREGATOR ConcatAggregator \
#     FORECASTING.DECODER MultiHeadDecoder \
#     DATA.CHECKPOINT_MODULE_FILE_PATH ${BACKBONE_WTS}

# # MViT-Concat
# BACKBONE_WTS=$PWD/pretrained_models/long_term_anticipation/ego4d_mvit16x4.ckpt
# run mvit_concat \
#     configs/Ego4dLTA/MULTIMVIT_16x4.yaml \
#     FORECASTING.AGGREGATOR ConcatAggregator \
#     FORECASTING.DECODER MultiHeadDecoder \
#     DATA.CHECKPOINT_MODULE_FILE_PATH ${BACKBONE_WTS}

# # Debug locally using a smaller batch size / fewer GPUs
# CLUSTER_ARGS="NUM_GPUS 2 TRAIN.BATCH_SIZE 8 TEST.BATCH_SIZE 32"


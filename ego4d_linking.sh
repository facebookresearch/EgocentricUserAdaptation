#!/bin/bash
# set download dir for Ego4D
# ROOT PATH =/fb-agios-acai-efs/Ego4D # ROOT path
# Meta data: /fb-agios-acai-efs/Ego4D/ego4d_data/ego4d.json
# Annotations: /fb-agios-acai-efs/Ego4D/ego4d_data/v1/annotations

#EGO4D_LTAA_DIR="/fb-agios-acai-efs/Ego4D/lta_video_clips/v1" # LTAA dset path
# CONTAINS:
#clips
#clips_audios
#resized_clips
#slowfast_ego4d_features

# download annotation jsons, clips and models for the FHO tasks
#python -m ego4d.cli.cli \
#    --output_directory=${EGO4D_DIR} \
#    --datasets annotations clips lta_models \
#    --benchmarks FHO

# link data to the current project directory
EGO4D_LOCAL_DIR="./forecasting" # Local
local_annotations_path="${EGO4D_LOCAL_DIR}/data/long_term_anticipation/"
remote_annotations_path="/fb-agios-acai-efs/Ego4D/ego4d_data/v1/annotations/"

mkdir -p ${local_annotations_path}
ln -s "${remote_annotations_path}" "${local_annotations_path}"
#ln -s ${EGO4D_DIR}/v1/clips/* data/long_term_anticipation/clips_hq/
#
## link model files to current project directory
#mkdir -p pretrained_models
#ln -s ${EGO4D_DIR}/v1/lta_models/* pretrained_models/

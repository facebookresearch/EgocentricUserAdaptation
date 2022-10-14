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

python -m continual_ego4d.processing.run_adhoc_add_finished_true_wandb


#!/bin/bash

# One argument to define config file
if [ "$#" -ne 1 ]; then
    echo "Must define config file name."
fi

# DEFINE CONFIG FILE
config=$1 # e.g. "splitmnist_ER.yaml"

# Add root to pythonpath
this_script_path=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) # Change location to current script
root_path=${this_script_path}/../ # One dir up
export PYTHONPATH=$PYTHONPATH:$root_path

# Params
pyscript="$root_path/main.py" # From root
MY_PYTHON="python"
exp_name="reproduce_paper"
config_path="$root_path/reproduce/configs/$config"


$MY_PYTHON "$pyscript" --config_path "$config_path" "$exp_name"
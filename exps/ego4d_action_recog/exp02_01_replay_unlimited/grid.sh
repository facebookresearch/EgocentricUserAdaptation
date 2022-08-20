#!/usr/bin/env bash

grid_cfg_names="METHOD.REPLAY.MEMORY_SIZE_SAMPLES,METHOD.REPLAY.STORAGE_POLICY" # Split by comma

mem_idx=0
mem_sizes=(1000000 100 10)
mem_val="METHOD.REPLAY.MEMORY_SIZE_SAMPLES ${mem_sizes[${mem_idx}]}"
echo "mem_val=$mem_val"

stor_idx=1
stor_policies=("reservoir_stream" "reservoir_action")
stor_val="METHOD.REPLAY.STORAGE_POLICY ${stor_policies[${stor_idx}]}"
echo "stor_val=$stor_val"

grid_overwrite_args=" $mem_val $stor_val"
echo "grid_overwrite_args=$grid_overwrite_args"
# grid_overwrite_args= METHOD.MEMORY_SIZE_SAMPLES 100000 METHOD.STORAGE_POLICY reservoir_stream

# Run script in current dir (same process with source)
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${__dir}/ego4d_action_recog.sh" "${grid_cfg_names}" "${grid_overwrite_args}"

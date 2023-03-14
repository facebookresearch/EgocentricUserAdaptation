#!/usr/bin/env bash

this_script_dirpath="$(dirname -- "${BASH_SOURCE[0]}")"
source ${this_script_dirpath}/init_run.sh

# Start in screen detached mode (-dm), and give indicative name via (-S)
#screenname="${RUN_UID}"
#screen -dmS "${screenname}" \
python -m ego4d.run_lta \
  --job_name "$RUN_ID" \
  --working_directory "${OUTPUT_DIR}" \
  ${OVERWRITE_CFG_ARGS}
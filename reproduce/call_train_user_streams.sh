#!/usr/bin/env bash

this_script_dirpath="$(dirname -- "${BASH_SOURCE[0]}")"
source ${this_script_dirpath}/init_run.sh

# Start in screen detached mode (-dm), and give indicative name via (-S)
#screenname="${run_id}"
#screen -dmS "${screenname}" \
python -m continual_ego4d.run_train_user_streams \
  --job_name "$run_id" \
  --working_directory "${OUTPUT_DIR}" \
  ${OVERWRITE_CFG_ARGS}

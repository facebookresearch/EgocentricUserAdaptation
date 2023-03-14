#!/usr/bin/env bash

this_script_dirpath="$(dirname -- "${BASH_SOURCE[0]}")"
source ${this_script_dirpath}/init_run.sh

# Start in screen detached mode (-dm), and give indicative name via (-S)
#screenname="${RUN_UID}"
#screen -dmS "${screenname}" \
python -m continual_ego4d.run_eval_hindsight \
  ${OVERWRITE_CFG_ARGS}
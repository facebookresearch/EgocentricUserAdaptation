#!/usr/bin/env bash

this_script_dirpath="$(dirname -- "${BASH_SOURCE[0]}")"
source ${this_script_dirpath}/parse_args.sh

# Start in screen detached mode (-dm), and give indicative name via (-S)
#screenname="${run_id}"
#screen -dmS "${screenname}" \
python -m continual_ego4d.run_eval_hindsight \
  ${OVERWRITE_CFG_ARGS}
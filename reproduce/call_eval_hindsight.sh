#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

this_script_dirpath="$(dirname -- "${BASH_SOURCE[0]}")"
source ${this_script_dirpath}/init_run.sh

# Start in screen detached mode (-dm), and give indicative name via (-S)
#screenname="${RUN_UID}"
#screen -dmS "${screenname}" \
python -m continual_ego4d.run_eval_hindsight \
  ${OVERWRITE_CFG_ARGS}

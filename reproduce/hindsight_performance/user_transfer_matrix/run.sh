#!/usr/bin/env bash

# Change dir to current script
this_script_dirpath=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
CONFIG="$this_script_dirpath/cfg.yaml"
this_script_filepath="${this_script_dirpath}/$(basename "${BASH_SOURCE[0]}")"
grid_overwrite_args="--cfg "${CONFIG}" --parent_script "${this_script_filepath}

# Report final
echo "grid_overwrite_args=$grid_overwrite_args"

# Run script in current dir (same process with source)
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${__dir}/../../call_train_user_streams.sh" "${grid_overwrite_args}"

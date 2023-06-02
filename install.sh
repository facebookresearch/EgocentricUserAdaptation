# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Example code for requirements installation, run manually

# Create and activate conda env
conda create -n ego_adapt python=3.9
conda activate ego_adapt # Activate

# Pytorch>=1.9.1 avoids bug resulting in abundant dataloader creation logs
# Install CUDA11.1 version with pip following: https://pytorch.org/get-started/previous-versions/#linux-and-windows-11
# Conda bug, keeps installing cpu version: conda install pytorch==1.9.1 torchvision==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
cat pip_requirements.txt | xargs -n 1 -L 1 pip install # Add pip-installed dependencies from ego4d

# Login to wandb logger
wandb login # Login with API token
wandb verify # Check if able to sync

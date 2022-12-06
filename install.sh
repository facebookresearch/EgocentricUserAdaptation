# Example code for requirements installation, run manually

# Create and activate conda env
conda create -n ego_adapt python=3.9
conda activate ego_adapt # Activate

# Pytorch>=1.9.1 avoids bug resulting in abundant dataloader creation logs
conda install pytorch==1.9.1 torchvision==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge # Pytorch manual install for GPU version
cat pip_requirements.txt | xargs -n 1 -L 1 pip install # Add pip-installed dependencies from ego4d

# Login to wandb logger
wandb login # Login with API token
wandb verify # Check if able to sync

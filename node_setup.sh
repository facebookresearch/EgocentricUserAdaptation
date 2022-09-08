# Don't run directly, but copy fragments of code

mkdir ~/sftp_remote_projects
cd ~/sftp_remote_projects
mkdir ContextualOracle_Matthias
cd ContextualOracle_Matthias

# Now setup SFTP in Pycharm to upload the whole directory

# Then get the environment setup:
env_name=matt_ego4d
# Locally copy explicit (must be same OS)
#conda activate $env_name
#conda list --explicit > spec-file.txt

# On new environment create exactly same
conda create --name matt_ego4d --file spec-file.txt
conda activate matt_ego4d
conda init bash

# Login to wandb logger
pip install wandb
wandb login --host=https://fairwandb.org

# Change bashrc and make symlink to current project
mv ~/.bashrc ~/.bashrc_backup
ln -s ~/sftp_remote_projects/ContextualOracle_Matthias/bashrc_sync ~/.bashrc

# Copy paste to .bashrc
cat >>~/.bashrc <<-EOM
conda activate matt_ego4d
# <<< CONDA CUSTOM INIT

alias jn='conda activate matt_notebook && cd /home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/notebooks && jupyter notebook --no-browser'
alias cdl='cd $(ls -1 | tail -n 1)' # Change dir to last one in sorted
alias cleangpu="nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 pkill"
alias gpusers="nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 ps -u -p 2>/dev/null"
alias killscreens="for scr in $(screen -ls | awk '{print $1}'); do screen -S $scr -X kill; done"

EOM


# Create symbolic links:
cd ~/sftp_remote_projects/ContextualOracle_Matthias
ln -s /fb-agios-acai-efs/mattdl/results .

# Data dir with symbolic links to actual datasets

# First copy datasets locally to nodes for speedup
mkdir -p /home/matthiasdelange/data/ego4d/lta_video_clips/clips
mkdir -p /home/matthiasdelange/data/ego4d/annotations

rsync -chavzP --stats /fb-agios-acai-efs/Ego4D/lta_video_clips/v1/clips/* /home/matthiasdelange/data/ego4d/lta_video_clips/clips
rsync -chavzP --stats /fb-agios-acai-efs/Ego4D/ego4d_data/v1/annotations/* /home/matthiasdelange/data/ego4d/annotations

# Then make symlinks
cd ~/sftp_remote_projects/ContextualOracle_Matthias/forecasting
mkdir data && cd data
mkdir long_term_anticipation && cd long_term_anticipation

ln -s /home/matthiasdelange/data/ego4d/annotations annotations_local
ln -s /home/matthiasdelange/data/ego4d/lta_video_clips clips_root_local


#annotations -> /fb-agios-acai-efs/Ego4D/ego4d_data/v1/annotations/
#annotations_local -> /home/matthiasdelange/data/ego4d/annotations
#clips_root -> /fb-agios-acai-efs/Ego4D/lta_video_clips/v1/
#clips_root_local -> /home/matthiasdelange/data/ego4d/lta_video_clips



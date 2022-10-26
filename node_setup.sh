# Don't run directly, but copy fragments of code

##################### ROOT DIR #####################
mkdir -p ~/sftp_remote_projects/ContextualOracle_Matthias
cd ~/sftp_remote_projects/ContextualOracle_Matthias


##################### PYCHARM SFTP COPY #####################
# Now setup SFTP in Pycharm to upload the whole directory


##################### CONDA ENV #####################
conda env remove -n matt_ego4d
conda env export > environment.yml
conda env export --from-history > environment_from_history.yml # Condensed to those installed in conda

# On new environment create exactly same
#conda create --name matt_ego4d --file spec-file.txt # Doesn't work
conda env create -f environment.yml
conda init bash
conda activate matt_ego4d
conda install pytorch==1.9.1 torchvision==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# Add pip-installed depenedencies from ego4d
cat pip_requirements.txt | xargs -n 1 -L 1 pip install

# Additionals installed (Can add later to requirements.txt)
pip install plotly

# Fix known bug with tensorboard
pip install setuptools==59.5.0

# Login to wandb logger
pip install wandb
wandb login --relogin --host https://fairwandb.org
wandb verify

# Latex for matplotlib logging
sudo apt-get install texlive
sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super


##################### .BASHRC #####################
# Change bashrc and make symlink to current project
mv ~/.bashrc ~/.bashrc_backup
ln -s ~/sftp_remote_projects/ContextualOracle_Matthias/bashrc_sync ~/.bashrc

##################### SYMLINKS #####################
# Create symbolic links:
cd ~/sftp_remote_projects/ContextualOracle_Matthias
ln -s /fb-agios-acai-efs/mattdl/results .

##################### DATASETS + SYMLINKS #####################
# First copy datasets locally to nodes for speedup
mkdir -p /home/matthiasdelange/data/ego4d/lta_video_clips/clips
mkdir -p /home/matthiasdelange/data/ego4d/annotations

rsync -chavzP --stats /fb-agios-acai-efs/Ego4D/lta_video_clips/v1/clips/* /home/matthiasdelange/data/ego4d/lta_video_clips/clips
rsync -chavzP --stats /fb-agios-acai-efs/Ego4D/ego4d_data/v1/annotations/* /home/matthiasdelange/data/ego4d/annotations

# Then make symlinks
mkdir -p ~/sftp_remote_projects/ContextualOracle_Matthias/forecasting/data/long_term_anticipation
cd ~/sftp_remote_projects/ContextualOracle_Matthias/forecasting/data/long_term_anticipation
ln -s /home/matthiasdelange/data/ego4d/annotations annotations_local
ln -s /home/matthiasdelange/data/ego4d/lta_video_clips clips_root_local

#annotations -> /fb-agios-acai-efs/Ego4D/ego4d_data/v1/annotations/
#annotations_local -> /home/matthiasdelange/data/ego4d/annotations
#clips_root -> /fb-agios-acai-efs/Ego4D/lta_video_clips/v1/
#clips_root_local -> /home/matthiasdelange/data/ego4d/lta_video_clips

##################### PERMISSIONS #####################
# Make all dirs writable for other users (different nodes) if needed:
# In current dir:
sudo find . -exec chmod 777 {} \;
sudo find /home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/* -type d -exec chmod 777 {} \;
sudo find /home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/* -exec chmod 777 {} \;



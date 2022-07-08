# ContextualOracle_Matthias

TODOS:
- Make output in exp log with timestampdir
- Checkpoint in checkpoints dir (besides exps dir), also need timestamp? -> Yes because when resume, will be manually!

## Ego4d codebase
See [forecasting](forecasting) for our experimental codebase.

### Data paths
- Meta-data: /fb-agios-acai-efs/Ego4D/ego4d_data/ego4d.json
- Annotations: /fb-agios-acai-efs/Ego4D/ego4d_data/v1/annotations
- LTA: /fb-agios-acai-efs/Ego4D/lta_video_clips
## Experiment scripts
See [exps](exps) for scripts per experiment, each attached with specific config file.
Scripts may overwrite configs for gridsearches, this is implemented through the ego4D arg parser.

## Notebooks
For Ego4D analysis notebooks see [notebooks](notebooks) directory and README.md for more information.

# Requirements
- Follow Ego4d requirements setup.
- Then install Cudatoolkit to use GPUs, install specific version for required Pytorch 1.9.0:


    conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

- To avoid known dependency bug with tensorboard (*AttributeError: module 'setuptools._distutils' has no attribute 'version'*), run this after setting up the environment:
  
  
    pip install setuptools==59.5.0


/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/exps/ego4d_LTA/test_exp_01/../../..//forecasting/data/long_term_anticipation/clips
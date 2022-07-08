# Matthias Continual Learning Benchmark Project

## Ego4d codebase
See [forecasting](forecasting) for our experimental codebase.

### Data paths
- Meta-data: 
  - /fb-agios-acai-efs/Ego4D/ego4d_data/ego4d.json
- Annotations: 
  - /fb-agios-acai-efs/Ego4D/ego4d_data/v1/annotations
- LTA videos/clips: 
  - /fb-agios-acai-efs/Ego4D/lta_video_clips
- Pre-trained Ego4D models (Kinetics + ego4d pretraining): 
  - /home/matthiasdelange/data/ego4d/ego4d_pretrained_models/pretrained_models/long_term_anticipation

### How to define our own splits?
If we want to keep the Ego4d setup, we can define a per-user split by splitting the dataset as in,
with the videos in each json ordered chronologically.

    Custom-split-dataset
      - train
        - user1.json
        - user2.json
        - ...
      - test
        - user6.json
        - user7.json
        - ...


Then we can pass through the config (`cfg`) to the `Ego4dLongTermAnticipation` class,
for each user iteration:

    class Ego4dLongTermAnticipation:
      data_path = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, f'fho_lta_{mode_}.json')

### DISAMBIGUITY NOTE
Any reference to `video` in API's refers to the 5min extracted video-clips from the Ego4D long videos.
The clips are predefined in the annotation-file (e.g. `fho_lta_{mode_}.json`), 
and each such video-clip has a unique `clip_uid` and video-file `video_path=f'{clip_uid}.mp4'`.
From these video-clips, the Pytorch `ClipSampler` samples smaller clips (of 1 to 2s). 
The 'clip' terminology can hence refer to 2 things!

### Data loading
**Dataset wrapper**:
`long_term_anticipation.py:Ego4dLongTermAnticipation` is train/val/test dataset wrapper, 
defines transforms, extraction of labels form the Pytorch Dataset entries, and defines (Distr) Sampler for the Pytorch dataset.

**Pytorch Dataset:**
`ptv_dataset_helper.py:clip_forecasting_dataset()` creates the actual Pytorch dataset.
It directly loads the annotation file (`fho_lta_{mode_}.json`), and groups annotations per (5 minute) video-clip (by `clip_uid`).

LabeledVideoDataset requires the data to be list of tuples with format: 
`(video_path, annotation_dict)`. For forecasting, the `video_path=f'{clip_uid}.mp4'`, and the `annotation_dict` contains 
- the input boundaries to be decoded in the video-clip `(clip_start_sec, clip_end_sec)`
- any observed clip annotations within those boundaries `(verb_label, noun_label)`
- a list of `num_future_actions` clip annotations (including labels and boundaries), these are extracted directly from the annotations in the 5-min video-clip based on order of `action_idx`.


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

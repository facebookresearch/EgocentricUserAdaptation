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
with the videos in each json ordered chronologically (not necessarily in the json, can be done ad-hoc).
Ideally we have some summary statistics generated for our split.

    Custom-split-dataset
      - train_usersplit.json:
        {
        - user1: { annotation entries }
        - user2: { annotation entries }
        - ...
        }

      - test_usersplit.json (MUTUALLY EXCLUSIVE):
        {
        - user7: { annotation entries }
        - user8: { annotation entries }
        - ...
        }


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
`ptv_dataset_helper.py:clip_forecasting_dataset()` creates the actual Pytorch `LabeledVideoDataset`, and therefore directly loads the annotation file (`fho_lta_{mode_}.json`), and groups annotations per (5 minute) video-clip (by `clip_uid`).
`LabeledVideoDataset` assumes a list of videos (`clip_uid.mp4`) and retrieves the next (sub)clip in one of the videos, based on the clip sampler strategy.

LabeledVideoDataset requires the data to be list of tuples with format: 
`(video_path, annotation_dict)`. For forecasting, the `video_path=f'{clip_uid}.mp4'`, and the `annotation_dict` contains 
- the input boundaries to be decoded in the video-clip `(clip_start_sec, clip_end_sec)`
- any observed clip annotations within those boundaries `(verb_label, noun_label)`
- a list of `num_future_actions` clip annotations (including labels and boundaries), these are extracted directly from the annotations in the 5-min video-clip based on order of `action_idx`.


OPEN QUESTION: HOW GO FROM ONE VIDEO TO NEXT? (So how are the videos sampled, not the clips within the vids?)

### How to implement sequential dataloaders?
`long_term_anticipation.py:Ego4dLongTermAnticipation` determines the order, both based on
`clip_sampler` within the video, and between videos with `video_sampler`.


For now the by default
- `video_sampler` is by default `DistributedSampler`, but we should make it a `SequentialDistributedSampler`.
  - Does this sequential nature limit us to using 1 GPU? No we can still benefit from distributed, by instead of online 1-by-1 learning, 
  we can observe the next batch in the buffer (e.g. the next N samples). But this batch-size is limited anyway.
  - Should we instead have 1 user running independently per GPU and scheduling users over GPUs?
- `clip_sampler` "uniform" if mode == "test" else "random" from [0,T]. We should make it [T-delta,T], so 
not all of the past can be observed.

**Video sampler:**
We can use the Pytorch [Sequential Sampler](https://pytorch.org/docs/master/data.html#torch.utils.data.SequentialSampler).
Note that this is NOT distributed! Multiple devices will be used independently, using multi-threading.

Solution 1: Zero effort
- Run all users sequentially, as we only do 1 epoch might be feasible

Solution 2: (Manual scheduling bottleneck) Run 1 job for each user. We could only do this in slurm cluster... 
On single instance, would need manual running per user (not feasible as #Users> 100?).

Solution3:  (GIL bottleneck!) single-machine multi-threading scheduler
- List available devices and assign single user to all of them
- Whenever devices becomes available, schedule next user.
- In the end aggregate all results

**Clip sampler:**
Can we also just use the Pytorch [Sequential Sampler](https://pytorch.org/docs/master/data.html#torch.utils.data.SequentialSampler).
in combination with the sequential video sampler?



### How to implement Experience Replay?
Write custom video sampler that can only sample from subsets of ranges in prev video.
(Or store video features and replay those instead).


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


# Docs
- PytorchVideo:
  - https://pytorchvideo.readthedocs.io/en/latest/index.html
- Pytorch Lightning:
  - p
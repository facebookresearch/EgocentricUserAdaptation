# Matthias Continual Learning Benchmark Project

TODO: Dataloadign problem is in eval past? Maybe to do with Seq sampler
TODO: copy final jsons to another outputdir
TODO: restore previous layout

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


### How to define our own splits? (TODO DEPRECATED)
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


**Pytorch Dataset and video-level entries**
`ptv_dataset_helper.py:clip_forecasting_dataset()` creates the actual Pytorch `LabeledVideoDataset`, and therefore directly loads the annotation file (`fho_lta_{mode_}.json`), and groups annotations per (5 minute) video-clip (by `clip_uid`).
`LabeledVideoDataset` assumes a list of videos (`clip_uid.mp4`) and retrieves the next (sub)clip in one of the videos, based on the clip sampler strategy.

LabeledVideoDataset requires the data to be list of tuples with format: 
`(video_path, annotation_dict)`. For forecasting, the `video_path=f'{clip_uid}.mp4'`, and the `annotation_dict` contains 
- the input boundaries to be decoded in the video-clip `(clip_start_sec, clip_end_sec)`
- any observed clip annotations within those boundaries `(verb_label, noun_label)`
- a list of `num_future_actions` clip annotations (including labels and boundaries), these are extracted directly from the annotations in the 5-min video-clip based on order of `action_idx`.


`ptv_dataset_helper.py:clip_recognition_dataset`
The videos (5min clips) are collected in a list per annotation entry.
So even if we iterate sequentially over the video entries of the dataset (video sampler),
this will only iterate the videos based on action_idx. It might happen that for a video
we actually go back in time at the end.

Possible solutions:
- First-in, first labeled. Meaning that if multiple actions overlap,
the end-time of the first one will be the final end-time. Whatever is left for the next action in line,
the remaining time is allocated to that one (if end-time of Action2 > end-time Action1).

- Multi-label classification:
In run-time: as long as subsequent action_idxs overlap,
make new video-entries for the overlap zones with both labels!
Pre and post are also added as separate video-entries.


**Video sampling**
To go from one video entry to the next, the list of video-entries with their annotations are 
iterated in the `LabeledVideoDataset`. A single 5-min video (based on clip_uid) typically has multiple entries,
1 for each action happening in the 5-min video. Each such action has a start and end time,
the `clip_sampler` samples the clips in between these action-boundaries within the 5-min clip.
The `video_sampler` iterates over the annotation entries (ordered on action_idx per video). 
Subsequent annotation entries hence may have the same associated video and video_path, 
only the range for the action within this 5-min video changes.


OPEN QUESTION: HOW GO FROM ONE VIDEO TO NEXT? (So how are the videos sampled, not the clips within the vids?)

### How to implement a sequential stream?
`long_term_anticipation.py:Ego4dLongTermAnticipation` determines the order, both based on
`clip_sampler` within the video, and between videos with `video_sampler`.

We have to sort annotated entries on 3 levels:

    # Per video (>>5miin): video_metadata.video_start_sec
    # Per 5min clip in video: 'clip_parent_start_sec'
    # Per action annotation in 5min clip: action_idx

Then, we apply a policy for overlapping annotation entries
1. Ignore-policy: We can go back in time if action-annotations overlap, we go back to the starting point of the next action.
2. First Single-action policy: If actions overlap, progress from end point of a1, and only visit the remaining time for a2.
3. Multi-action policy: If actions overlap, break down the periods in multi-class classification samples.

For now the by default
- `video_sampler` is by default `DistributedSampler`, but we should make it a `SequentialSampler`.
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

**UntrimmedClipSampler BUG**:
FIXME THE ERROR IS THE CLIP_DURATION PASSED, WHICH IS UNTRIMMED, RATHER THAN THE TRIMMED ONE!!
`UniformClipSampler` is wrapped in the `UntrimmedClipSampler`, which calls for the `__call__` in `UniformClipSampler`.
It passes directly the last end-clip time. which is UNTRIMMED, while in `UniformClipSampler` it works with TRIMMED VERSION.
Therefore when a second clip is sampled with this one, it is completely out of range. (e.g. when first sample 
has (start=40,end=42s), Instead of 2s in the untrimmed clip, 42s is passed, which can result in completely out of bounds new clip).

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
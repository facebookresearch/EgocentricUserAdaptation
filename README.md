# Matthias Continual Learning Benchmark Project

TODO: Only use default usersplits in config, remove others


TODO give separate nb workers for continual_eval (e.g. 10 for batch_size 10 and stream 100), 
predict (full stream, can take 10 processes as well) and 
train (e.g. 3 because waiting for continual eval mostly gives a lot of time for next batch)


How to do gridsearch?
In .sh script pass CFG attributes that are being gridsearched over. In Python main script these are then also added in the
output path.

The possible gridvalues are defined in the bashscript, and indexed as an array.

We have to run jobs separately anyway with the nodes..


Pass timestamp in shell script instead of python, as in clusters/scheduling it gives the timestamp it was 
launched in queue, not timestamp popped (which may have arbitrary order).



TODO: Check pretrain not running validation all the time, and how much data? Maybe we should use our train usersplit?

TODO: Ego4d dataset is Iterable dataset or not? Even if iterable stil have to define len, which is hard to measure.
TODO: FIX batch count for dataset. Are we evaluating each epoch?


TODO: Dataloadign problem is in eval past? Maybe to do with Seq sampler
TODO: copy final jsons to another outputdir
TODO: restore previous layout



TODOS:
1. Run pretrain reference runs on train/test
2. Adapt `run_adhoc_metric_processing_wandb.py` and run for all methods
3. Adapt `run_transfer_eval.py` to also use the pretrain wandb group results to get delta performance.
   1. For Forgetting experiment: We already have the accuracies! ReexposureForgettingAccMetric
   2. For transfer matrix: Just rerun and instead of only loss, also use the accuracy metrics
   3. Instance counts: 


## Results flow

Make sure we first have the per-user performance for a fixed pretrained model, then get reference to these group names.
- The performance before the stream, and in hindsight is the same for the pretrained model!
- No need to run `run_transfer_eval.py`.


### OAG: Stream generalization
For OAG, we measure the following during training, giving the final absolute avg-ACC on the end of the stream (calculated online), per user:
- train_action_batch/top1_acc_running_avg
- train_verb_batch/top1_acc_running_avg
- train_noun_batch/top1_acc_running_avg
- train_verb_batch/top5_acc_running_avg
- train_noun_batch/top5_acc_running_avg


We use adhoc postprocessing to average and SE these metrics over user streams and upload to wandb
- `run_adhoc_metric_processing_wandb.py`: 
  - Avg absolute performance
  - And use pretrain performance Group Names to get AG results.

Then, update tables in WandB reports, download CSV's, and generate latex tables.
- `csv_to_latex_table_parser.py`

### HAG: Hindsight performance
To get HAG performance:
- Run `run_transfer_eval.py` for the wandb CSV groups: To get the absolute performance values. Once again we can use the same pretrain values to compare to.
- If pretrain run group available, we use this one to process the HAG results (besides the absolute ones). Again per-user delta, then avg + SE over users.

### Training performance
To show training performance over time, we can still plot the original OAG curves.
If not available, we can use the training loss curves from the method, and if we want OAG: get pretrain loss curves and use to calculate deltas. 


## Experiment Pipeline

1. continual_ego4d/run_recog_CL.py: Train using multiprocessing over all user streams as a single group. Each independent user-adaptation stream is scheduled as a separate process. Scheduled automatically
2. continual_ego4d/run_recog_CL.py/processing/run_adhoc_metric_processing_wandb.py: Aggregate all user stream results in single metric averaged over user streams per group.
3. continual_ego4d/run_recog_CL.py/run_transfer_eval.py: Run evaluation of final user models with user streams. 
   1. Diagonal only for calculating HAG.
   2. Full transfer matrix for HAG transfer matrix (N user models x N user streams)


## Pretraining
To pretrain we can use the original ego4d repo. 
However, to enable pretraining from our user-split data you can configure 

    cfg.DATA.PATH_TO_DATA_FILE


## Configs

To determine the dataset (e.g. EGO4d) and which user-split:

    cfg.TRAIN.DATASET: Ego4dContinualRecognition # Dataset Object
    
    # Paths to preprocessed jsons, split per user (result from run_usersplits_ego4d.py)
    cfg.DATA.PATH_TO_DATA_SPLIT_JSON.TRAIN_SPLIT
    cfg.DATA.PATH_TO_DATA_SPLIT_JSON.TEST_SPLIT
    cfg.DATA.PATH_TO_DATA_SPLIT_JSON.PRETRAIN_SPLIT
    
    cfg.DATA.USER_SUBSET # Select one of the previous json paths (train/test/pretrain)
    cfg.DATA.COMPUTED_USER_DS_ENTRIES: # Set dynamically to stream json

## How to unittest

    cd forecasting
    python -m unittest tests.test_metrics


See: https://stackoverflow.com/questions/1896918/running-unittest-with-typical-test-directory-structure 

## Guide on num_workers and batch size
Each worker loads a single batch and returns it only once it’s ready.
Typically use about 8 workers (8 batches loaded concurrently), but the wait-time for 1 batch is limited to a single worker!
Hence if we have a continual eval stream length of 100, and we take batch size 100, only 1 worker will be loading
and on the end will return the batch, hence we have no speed-up with multiple workers. 
On the other hand, having a lower batch size allows multiple batches to be loaded in parallel e.g. with 10 workers, each loads 10 samples.
Then at once, 10 batches will be able to be processed in the forward. The forwarding is not parallelized though, but as IO is main bottleneck we still have speedup!

See: https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813

## How to gridsearch
Use 'grid.sh' script which passes the CFG-node names and values for the nodes to directly overwrite to the main bash script.
The main python script incorporates the node-names in the final cfg.OUTPUT_DIR to easily find entries from different runs.

TODO: Add GRID_NODES in config instead: So can easily derive

## Pretraining on usersplit Ego4d
See [Ego4d LTA README](forecasting/LONG_TERM_ANTICIPATION.md) for a guide on how to use pretraining in general.

    python run_usersplit_ego4d_LTA.py --p_output_dir /fb-agios-acai-efs/mattdl/data/ego4d_lta_usersplits

1. Make a usersplit with the script [run_usersplit_ego4d_LTA.py](forecasting/continual_ego4d/run_usersplit_ego4d_LTA.py). 
This will generate a json split for pretraining. Use this json as input path for the config file when using pretraining for action recognition.
2. Execute the ego4d script
```
  bash tools/long_term_anticipation/ego4d_recognition.sh checkpoints/recognition/
  ```


### Pretrained model paths
- All pretrained models are copied and backed up at:
  - /fb-agios-acai-efs/mattdl/ego4d_models/continual_ego4d_pretrained_models_usersplit

USED
- Original model checkpoints on our usersplit (incl NaN users), with LR 1e-4, cosine schedule for 66 epochs, without linear warmup:
  - `/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/pretrain_slowfast/logs/2022-09-05_10-34-05_UIDd05ed672-01c5-4c3c-b790-9d0c76548825/checkpoints`  
  - Copied to: `/fb-agios-acai-efs/mattdl/ego4d_models/continual_ego4d_pretrained_models_usersplit/pretrain_148usersplit_incl_nan/2022-09-05_10-34-05_UIDd05ed672-01c5-4c3c-b790-9d0c76548825/checkpoints/best_model.ckpt`
  - This model is: 'epoch=45-step=10901.ckpt'
- Model provided by ego4d paper, pretrained on kinetics400, then on ego4d.
  - `/fb-agios-acai-efs/mattdl/ego4d_models/ego4d_pretrained_models/pretrained_models/long_term_anticipation/ego4d_slowfast8x8.ckpt`


DEPRECATED
- Model of 30 epochs on pretrain data (without NaN user):
  - /home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/pretrain_slowfast/logs/2022-07-28_17-06-20_UIDe499d926-a3ff-4a28-9632-7d01054644fe/lightning_logs/version_0/checkpoints
- Model of 30 epochs on pretrain data WITH NaN user:
  - /home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/pretrain_slowfast/logs/2022-08-13_09-41-26_UID8196dadf-1ce7-4ed5-85c1-9bd3d1e6ffe6/lightning_logs/version_0/checkpoints


### Data paths
- Usersplit including NaN-user + action sets:
  - /home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/forecasting/continual_ego4d/usersplit_data/2022-08-09_16-02-54_ego4d_LTA_usersplit/ego4d_LTA_pretrain_incl_nanusers_usersplit_148users.json

### Loggin multiple TB-dirs:
Use script tb_plot_local.py which copies from remote to local in single dir and plots for this dir.


### JSON structure

Tree of keys in the json structure:
- user_action_sets
  - user_agnostic: over all users
  - "USER-ID": single user
    - verb_to_name_dict
    - noun_to_name_dict
    - action_to_name_dict
      - {'name': "ACTION_NAME", 'count': "ACTION_COUNT"}
- users
  - "USER-ID": Flattened user-specific list of dict-entries. Each clip is an annotation entry.
- clips: Flattened user-agnostic list of dict-entries. Each clip is an annotation entry.

## Ego4d codebase
See [forecasting](forecasting) for our experimental codebase.

### Bugs
- UntrimmedClipSampler is replaced with EnhancedUntrimmedClipSampler for major bugfix. The UntrimmedClipSampler assumes the clip-end of previous o

### Data paths
- Meta-data: 
  - /fb-agios-acai-efs/Ego4D/ego4d_data/ego4d.json
- Annotations: 
  - /fb-agios-acai-efs/Ego4D/ego4d_data/v1/annotations
- LTA videos/clips: 
  - /fb-agios-acai-efs/Ego4D/lta_video_clips
- Pre-trained Ego4D models (Kinetics + full ego4d pretraining): 
  - /home/matthiasdelange/data/ego4d/ego4d_pretrained_models/pretrained_models/long_term_anticipation

### Speed-up by locally storing dataset

- Move with rsync from `/fb-agios-acai-efs/Ego4D/lta_video_clips/v1/clips` to some local dir
  - Add local private key link locally in `~/.ssh/config`
  - `rsync -chavzP --stats /fb-agios-acai-efs/Ego4D/lta_video_clips/v1/clips/* /home/matthiasdelange/data/ego4d/lta_video_clips/clips`. lta_video_clips is the video_root dir
  - `rsync -chavzP --stats /fb-agios-acai-efs/Ego4D/ego4d_data/v1/annotations/* /home/matthiasdelange/data/ego4d/annotations`
- Make symlink to local dir in `forecasting/data/long_term_anticipation/clips_root/clips_local`
  - `mkdir /home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/forecasting/data/long_term_anticipation/clips_root_local` and cd
  - `ln -s /home/matthiasdelange/data/ego4d/lta_video_clips clips_root_local`. Has e.g. `clips` and `clips_resized` dirs underneath.
  - `ln -s /home/matthiasdelange/data/ego4d/annotations annotations_local`
- In exp scripts, use `forecasting/data/long_term_anticipation/clips_root/clips_local` 


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
- UPDATE: To avoid abundant warnings on dataloader creation `[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)`.
  Install pytorch 1.9.1 or later (no other way to suppress, see [link](https://github.com/pytorch/pytorch/issues/57273)).


    conda install pytorch==1.9.1 torchvision==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

- Then install Cudatoolkit to use GPUs, install specific version for required Pytorch 1.9.0:


    conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

- To avoid known dependency bug with tensorboard (*AttributeError: module 'setuptools._distutils' has no attribute 'version'*), run this after setting up the environment:
  
  
    pip install setuptools==59.5.0


## For notebooks
Also install opencv: 

    conda install opencv

# Docs
- PytorchVideo:
  - https://pytorchvideo.readthedocs.io/en/latest/index.html
- Pytorch Lightning:
  - p



# TMP
def plot_user_histogram_grid_highlight_seen_actions_pretrain(dfs):
    """Plot grid of overview plots"""
    cols = ['verb_label','noun_label','action_label']
    col_action_sets = [verb_to_name_dict ,noun_to_name_dict,action_to_name_dict,]
    
    nb_users = len(dfs)
    fig, ax = plt.subplots(nb_users, len(cols), figsize=(15, 30), dpi=600)

    
    for row_idx, (user_id, user_df) in enumerate(dfs.items()):
        for col_idx, col in enumerate(cols):
            cnt = Counter(user_df[col].tolist())
            col_action_set = col_action_sets[col_idx]

            col_sorted = sorted([(k,v) for k,v in cnt.items() ] ,key=lambda x: x[1], reverse=True)
            vals_sorted = [x[0] for x in col_sorted]
            cnts_sorted = [x[1] for x in col_sorted]
            
            indices_in_actionset = [idx for idx,val in enumerate(vals_sorted) if val in col_action_set]
            print(f"Seen actions = {len(indices_in_actionset)}")
            
            if len(indices_in_actionset) ==0:
                import pdb;
                pdb.set_trace()


            print(f"Freqs for {col}: {cnts_sorted}")
            print(f"Labels for {col}: {vals_sorted}")

            y_axis= cnts_sorted
            x_axis= list(range(len(cnts_sorted)))
            nb_samples = sum(cnts_sorted)
            nb_mins = int(nb_samples*2.1/60)
            plot_subplot_highlight(ax[row_idx,col_idx], x_axis, y_axis, 
                                   title=f'USER {user_id} - {col}| #={nb_samples}|mins={nb_mins}',
                                  highlight_idxs=indices_in_actionset)
#     plt.suptitle(f'Stream sample histogram plot: Verb/noun/action freq {user_id}')
    
    fig.tight_layout() 
    plt.show()
    plt.clf()

    
# Barchart API: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html#matplotlib.pyplot.bar
def plot_subplot_highlight(sub_ax, x_axis, y_vals, title,ylabel=None,xlabel=None, 
                  grid=False,yerror=None,xerror=None, y_labels=None, x_labels=None,bar_align='edge',barh=False,
                 figsize=(12, 6), log=False, interactive=False,x_minor_ticks=None,highlight_idxs=None):
    max_val = max(y_vals)
    my_cmap = plt.get_cmap("plasma")
#     fig = plt.figure(figsize=figsize, dpi=600) # So all bars are visible!
#     ax=plt.subplot()
    
    barlist = sub_ax.bar(x_axis, height=y_vals,color=my_cmap.colors, align=bar_align,yerr=yerror,width=0.9,log=log)
    
    if highlight_idxs is not None:
        for idx in highlight_idxs:
            barlist[idx].set_color('r')
            
        

    if x_minor_ticks is not None:
        sub_ax.set_xticks(x_minor_ticks, minor=True)


    if x_labels:
        plt.xticks(x_axis, x_labels, rotation='vertical')
    if y_labels:
        plt.yticks(y_vals, y_labels)
    
    sub_ax.set_ylim(None,max_val*1.01)
    sub_ax.set_xlim(None,None)
#     sub_ax.set_xlabel(xlabel)
#     sub_ax.set_ylabel(ylabel)
#     plt.title(title)
    sub_ax.set_title(title)
    sub_ax.grid(grid, which='both')
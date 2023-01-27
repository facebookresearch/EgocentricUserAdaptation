# Online Egocentric User-Adaptation

# TODOs for cleanup

- <del> Instructions for requirements.
- <del> Data preprocessing: Take same users as we used
- Path-linking jsons (set PATH_TO_DATA_FILE in json with the generated ones, TODO check if works in run flow)
- Design exps dir: which ones certainly keep? -> Gather configs in parent figure/exp dir, with per-method subdirs, have general script.
  - Put configs with comments on which ones to replace
- Step 2.2, check automatically if all users are processed, and do post-processing after.
- Keep hindsight in separate step
- Clean-up notebooks + only keep those that are useful.

Final checks
- check if all internal paths gone everywhere
- 

## Installation
The repository is based on the original Ego4d codebase (see [original README](src/ego4d/README.md)).
First request access to Ego4d through [this form](), which will send an email with AWS credentials within 48 hours.
To configure your download run:

    sudo apt install awscli    # AWS command-line
    aws configure              # Config AWS
      AWS Access Key ID [None]: <ID FROM EMAIL>
      AWS Secret Access Key [None]: <KEY FROM EMAIL>
      Default region name [None]: us-east-2 # Or leave blank
      Default output format [None]:         # Leave blank

Then, continue with the actual download, this command downloads the ego4d fho (forecasting) subset in the output directory:
    
    pip install ego4d
    ego4d --output_directory='/media/mattdl/backup prim/datasets' \
    --datasets annotations clips lta_models \
    --benchmarks FHO

Follow the steps in [install.sh](install.sh) for installation of the requirements in an Anaconda environment.
This code was tested with Python=3.9 (requires python version >=3.7) and Pytorch=1.9.1.
The workflow relies on the [WandB](https://wandb.ai/site) logging platform to manage all runs.

## Results workflow

**Context:** The paper defines 2 phases (1) the pretraining phase of the population model, (2) the user-adaptation
phase.
Additionally we introduce a third phase (3) that aggregates all user results into single metric results.

### Data preprocessing

- First [download the Ego4d dataset](https://ego4d-data.org/#download). We will use the train and validation annotation JSONS to create the user splits for
  train/test/pretrain. In the forecasting LTA benchmark, download the meta-data, annotations, videos, and the SlowFast Resnet101 model pretrained on Kinetics-400.
- Run script [run_split_ego4d_on_users.py](src/continual_ego4d/processing/run_split_ego4d_on_users.py) to generate JSON files
  for our user splits. Link in the [default config](src/ego4d/config/defaults.py) to the generated json paths by setting properties *PATH_TO_DATA_SPLIT_JSON.{TRAIN,VAL,TEST}*.

### (1) Pretraining a population model
To obtain a pretrained population model. 

- First run pretraining phase in [reproduce/pretrain/learn_user_pretrain_subset](reproduce/pretrain/learn_user_pretrain_subset), 
which starts from the Kinetics-400 pretrained model (downloaded from Ego4d) and trains further on our pretrain user-split.
Make sure to adapt the *PATH_TO_DATA_FILE.{TRAIN,VAL}* in the [cfg.yaml](reproduce/pretrain/learn_user_pretrain_subset/cfg.yaml), 
with TRAIN being our pretraining JSON user-split and VAL our train JSON user-split.
  - **Config:** Once you obtained the pretrained model, use it by setting the *CHECKPOINT_FILE_PATH* in the config files.
- Then, run evaluation of the pretrain model in [reproduce/pretrain/eval_user_stream_performance](reproduce/pretrain/eval_user_stream_performance), 
both for the train and test users. This allows later metrics to be calculated as relative improvement over the pretrain model. 

### (2) Reproduce Online User-adaptation

All users start from the same pretrained model obtained from (1).
To reproduce results from the paper, run the scripts in [reproduce](reproduce), containing the config files 
[cfg.yaml]() per experiment in the subdirectories.

- [reproduce/pretrain](reproduce/pretrain): Train population model in pretraining phase
- Empirical study on train user-split: 
  - [reproduce/momentum](reproduce/momentum): Momentum strengths.
  - [reproduce/multiple_updates_per_batch](reproduce/multiple_updates_per_batch): More than 1 update per batch.
  - [reproduce/non_stationarity_analysis](reproduce/non_stationarity_analysis): Label window predictor
  - [reproduce/replay_strategies](reproduce/replay_strategies): Replay with storage strategies FIFO/Hybrid-CBRS/Reservoir.
  - [reproduce/user_feature_adaptation](reproduce/user_feature_adaptation): Fixing the feature extractor/classifier.
- [reproduce/test_user_results](reproduce/test_user_results): Final test user-split results
- [reproduce/hindsight_performance](reproduce/hindsight_performance): Get hindsight results (HAG) after learning user-streams


### (postprocess) Parsing final WandB results
**In WandB**: All results are saved in the WandB run entries and in local CSV dumps with per-update predictions. 
To get the results in WandB, group runs (1 run is 1 user-stream) on the *Group* property, and select the relevant metrics.

**Parse to Latex**: Additionally, to see which metrics are used in the paper, you can check out [the table parser script](src/continual_ego4d/processing/csv_to_latex_table_parser.py) for examples.
The script parses downloaded (grouped) runs from WandB that are exported to a CSV, and parses the CSV to Latex-formatted tables.


## Notebooks

Can be found in [notebooks](notebooks).

- [Video Player](notebooks/ego4d_OnlineActionRecog_video_player.ipynb) to display our actual user streams per
  user-split. Displays meta-data such as action (verb,noun) and user-id over time.
- [plot_classifier_weights_biases.ipynb](notebooks/): Analysis for verbs/noun on classifier weight and bias norms.
  Compares SGD on head only vs SGD on full model.
- [plot_ego4d_stats.ipynb](notebooks/): Video length in minutes (y-axis) per user (x-axis). Color codes the user splits.
- [plot_forgetting_comparison_SGD_replay.ipynb](notebooks/): Re-exposure Forgetting (RF) analysis comparing Replay and
  SGD (2 lines) for RF (y-axis) on log-binned re-exposure count (x-axis).
- [plot_heatmap_transfer.ipynb](notebooks/): Heatmap of HAG-action (of instance-based micro-loss).
- [plot_SGD_gradient_analysis.ipynb](notebooks/): Grouped-barplot comparing gradient cosine-similarity of current batch
  with previous points k steps in history of the learning trajectory.
- [plot_SGD_per_user_OAG.ipynb](notebooks/): Plots a single line per user for the instance-based micro-loss. These are
  the learning curves for the users over time.
- [plot_user_action_distribution.ipynb](notebooks/): Plots the CDF of the action-histograms in the test userset.
- [plot_user_vs_pretrain_distribution.ipynb](notebooks/): Plots the pretrain distribution ordered on frequency, and then
  overlays the test action distribution on top.
- [plot_likelihood_loss_analysis_conditional.ipynb](notebooks/): Comparison in Appendix of why we get different trends
  for loss and accuracy on multiple iterations.
- [plot_multi_iter_grouped_barplot.ipynb](notebooks/plot_multi_iter_grouped_barplot.ipynb): Lines for different
  metrics (OAG, HAG, and OAG disentangled in OAG-correlated and OAG-decorrelated) on y-axis, and number of updates on
  same batch on x-axis.
- [plot_heatmap_transfer_user_action_overlap.ipynb](notebooks/plot_heatmap_transfer_user_action_overlap.ipynb): Plot the
  number of overlapping actions betwee train-users in a heatmap.
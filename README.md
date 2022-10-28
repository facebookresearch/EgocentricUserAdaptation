# Online Egocentric User-Adaptation

## Results workflow

**Context:** The paper defines 2 phases (1) the pretraining phase of the population model, (2) the user-adaptation
phase.
Additionally we introduce a third phase (3) that aggregates all user results into single metric results.

### Data preprocessing

- First download the Ego4d dataset. We will use the train and validation annotation JSONS to create the user splits for
train/test/pretrain.
- Run script [run_usersplit_ego4d_LTA.py](forecasting/continual_ego4d/run_usersplit_ego4d_LTA.py) to generate JSON files 
for our user splits.
- Link the paths to the jsons in your configs.


### (1) Pretraining a population model

Run original Ego4d training script [forecasting/scripts/run_lta.py](forecasting/scripts/run_lta.py).
Configs can be found in the runned experiments [exps](exps), e.g. we
used [exps/ego4d_action_recog/pretrain_slowfast](exps/ego4d_action_recog/pretrain_slowfast).

### (2) Online User-adaptation

All users start from the same pretrained model obtained from (1).
Define the path to this model in your config.
The users can be processed independently, either sequential (single prcess) or concurrently (multi-processing).
To support both, the results are first collected and checkpointed for each user stream, and only if all are processed, the result are aggregated in the next step (3).

Details:
You can define how many users to run per GPU device for the scheduler.


### (3) User results aggregation


### (postprocess) From WandB to Latex


## Internal reproducing of plots/result tables

If getting permission errors of my EC2 result
directory `/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results`, this is a symlink to a shared
EFS directory for the results: `/fb-agios-acai-efs/mattdl/results`.

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

Plots not included in paper:

- [plot_forgetting_reexposure.ipynb](notebooks/): The original SGD-only re-exposure plot (barplot) for Re-expsoure
  Forgetting (RF) on y-axis, not used for the paper in the end.
- [plot_instance_counts.ipynb](notebooks/): Not used in the end. Instance count vs pretrain count KDE.


# Reproduce experiments
The experiments to reproduce are grouped in this directory.
In the following we give a brief overview over all experiments.

- **pretrain**
  - **learn_user_pretrain_subset**: Pretrain on the pretraining-user streams (U_pretrain).
  - **eval_user_stream_performance**: Keep the pretrained model fixed and obtain the online performance on the U_train and U_test user streams. Allows calculating the OAG/HAG for later experiments.

- **non_stationarity_analysis**: Label-window predictor (LWP) that gives an indication of stream correlation.
- **momentum**: Momentum ablation experiments.
- **user_feature_adaptation**: Freeze 
- **multiple_updates_per_batch**: 1 vs multiple updates per mini-batch.
- **replay_strategies**: Ablation over experience replay storage strategies and memory sizes.
- **test_user_results**: Final table with results on the 40 user streams in U_test. 
- **hindsight_performance**: Calculate the HAG performance over the final models of each user stream in a finished experiment.

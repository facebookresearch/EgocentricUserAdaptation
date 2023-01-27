All users start from the same pretrained model obtained from (1).
Define the path to this model in your config.
The users can be processed independently, either sequential (single prcess) or concurrently (multi-processing).
To support both, the results are first collected and checkpointed for each user stream, and only if all are processed,
the result are aggregated in the next step (3).

The following scripts are used for this step:

- [forecasting/continual_ego4d/run_recog_CL.py](src/continual_ego4d/run_train_user_streams.py): The main script, does data
  preprocessing, iterates over users and starts the Continual Learning Task for training.
- [forecasting/continual_ego4d/tasks/continual_action_recog_task.py](src/continual_ego4d/tasks/continual_action_recog_task.py):
  The continual learning task as Pytorch Lightning module. Defines dataloaders, builds the dataset, and calls
  training/prediction hooks for the defined method.
- [forecasting/continual_ego4d/methods/method_callbacks.py](src/continual_ego4d/methods/method_callbacks.py):
  Defines the method for which training hooks will be called. The methods define for example the training step, forward,
  etc
- [forecasting/continual_ego4d/datasets/continual_action_recog_dataset.py](src/continual_ego4d/datasets/continual_action_recog_dataset.py):
  The dataset, takes in the json that is split based on train/test/pretrain user-split, and for all action time ranges,
  samples ~2.1s consecutive video as individual samples. Uses FIFO priority policy for actions in case time ranges
  overlap.

### (2.2) Adhoc user result aggregation

To support the multi-processing, the training results are aggregated only in a second step.
[forecasting/continual_ego4d/processing/run_adhoc_metric_processing_wandb.py](src/continual_ego4d/processing/run_adhoc_metric_processing_wandb.py):
More specifically: You can call this script for aggregating the user results.
Select a single run-id or select a CSV with WandB group-names to get the final results.

TODO for final codebase: Call automatically after processing all users.

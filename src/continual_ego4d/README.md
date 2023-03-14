### Code overview
All users start from the same pretrained model obtained from (1).
Define the path to this model in your config.
The users can be processed independently, either sequential (single process) or concurrently (multi-processing).
To support both, the results are first collected and checkpointed for each user stream, and only if all are processed,
the result are aggregated in the next step.
To support the multi-processing, the training results are automatically aggregated in a second step in [src/continual_ego4d/run_train_user_streams.py](src/continual_ego4d/run_train_user_streams.py).

The following scripts are used for training on user streams:

- [src/continual_ego4d/run_recog_CL.py](src/continual_ego4d/run_train_user_streams.py): The main script, does data
  preprocessing, iterates over users and starts the Continual Learning Task for training.
- [src/continual_ego4d/tasks/continual_action_recog_task.py](src/continual_ego4d/tasks/continual_action_recog_task.py):
  The continual learning task as Pytorch Lightning module. Defines dataloaders, builds the dataset, and calls
  training/prediction hooks for the defined method.
- [src/continual_ego4d/methods/method_callbacks.py](src/continual_ego4d/methods/method_callbacks.py):
  Defines the method for which training hooks will be called. The methods define for example the training step, forward,
  etc
- [src/continual_ego4d/datasets/continual_action_recog_dataset.py](src/continual_ego4d/datasets/continual_action_recog_dataset.py):
  The dataset, takes in the json that is split based on train/test/pretrain user-split, and for all action time ranges,
  samples ~2.1s consecutive video as individual samples. Uses FIFO priority policy for actions in case time ranges
  overlap.



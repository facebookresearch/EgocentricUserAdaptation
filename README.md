# EgoAdapt: Online Egocentric User-Adaptation

## Installation

The repository is based on the original Ego4d codebase (see [original README](src/ego4d/README.md)). First request
access to Ego4d through [this form](), which will send an email with AWS credentials within 48 hours. To configure your
download run:

    sudo apt install awscli    # AWS command-line
    aws configure              # Config AWS
      AWS Access Key ID [None]: <ID FROM EMAIL>
      AWS Secret Access Key [None]: <KEY FROM EMAIL>
      Default region name [None]: us-east-2 # Or leave blank
      Default output format [None]:         # Leave blank

To proceed to the actual download, the following commands (1) download the ego4d FHO (forecasting) subset in the 
"*EGO4D_ROOT*" output directory, and (2) create a symbolic link from the project root:

    export EGO4D_ROOT='/path/to/your/download/Ego4D'

    pip install ego4d
    ego4d --output_directory="${EGO4D_ROOT}" \
    --datasets annotations clips lta_models \
    --benchmarks FHO

    # Run from project root to create symbolic link
    mkdir -p data && ln -s "${EGO4D_ROOT}" ./data/Ego4D

Follow the steps in [install.sh](install.sh) for installation of the requirements in an Anaconda environment. This code
was tested with Python=3.9 (requires python version >=3.7) and Pytorch=1.9.1. The workflow relies on
the [WandB](https://wandb.ai/site) logging platform to manage all runs.

## Results workflow

**Context:** The paper defines 2 phases (1) the pretraining phase of the population model, (2) the user-adaptation
phase. Additionally, this codebase performs a third phase (3) that aggregates all user results from phase (2) into
single metric results.

### Data preprocessing

- First [download the Ego4d dataset](https://ego4d-data.org/#download). We will use the train and validation annotation
  JSONS to create the user splits for train/test/pretrain. In the forecasting LTA benchmark, download the meta-data,
  annotations, videos, and the SlowFast Resnet101 model pretrained on Kinetics-400.
- Run script [run_split_ego4d_on_users.py](src/continual_ego4d/processing/run_split_ego4d_on_users.py) to generate JSON
  files for our user splits. The [default config](src/ego4d/config/defaults.py) automatically refers to the generated
  json paths with properties *PATH_TO_DATA_SPLIT_JSON.{TRAIN,VAL,TEST}*.

### Phase (1): Pretraining a population model

To obtain a pretrained population model.

- First run pretraining phase
  in [reproduce/pretrain/learn_user_pretrain_subset](reproduce/pretrain/learn_user_pretrain_subset), which starts from
  the Kinetics-400 pretrained model (downloaded from Ego4d) and trains further on our pretrain user-split. Make sure to
  adapt the *PATH_TO_DATA_FILE.{TRAIN,VAL}* in the [cfg.yaml](reproduce/pretrain/learn_user_pretrain_subset/cfg.yaml),
  with TRAIN being our pretraining JSON user-split and VAL our train JSON user-split.
    - To obtain the pretrained Kinetics-400 model to start our pretraining with:

          # Unzip downloaded Ego4d models (execute from project root)
          unzip ./data/v1/lta_models/lta_pretrained_models.zip 
          # Set in the config the Kinetics-400 pretrained model to start from:
          DATA.CHECKPOINT_MODULE_FILE_PATH=${PROJECT_ROOT}/data/v1/lta_models/pretrained_models/long_term_anticipation/kinetics_slowfast8x8.ckpt

    - **How to use in later experiments?** Once EgoAdapt pretraining is completed (initialized with Kinetics400
      pretraining), use the model for subsequent experiments by setting the *CHECKPOINT_FILE_PATH* in the
      experiment's [cfg.yaml]() config files or set the path as default in
      the [default config](src/ego4d/config/defaults.py).
- Then, run evaluation of the pretrain model
  in [reproduce/pretrain/eval_user_stream_performance](reproduce/pretrain/eval_user_stream_performance), both for the
  train and test users. Do this by setting the *DATA.USER_SUBSET* in the [cfg.yaml](reproduce/pretrain/eval_user_stream_performance/cfg.yaml) to 'test' or 'train'. This allows later metrics to be calculated as relative improvement over the pretrain model.
    - **How to use in later experiments?** Set in the [default config](src/ego4d/config/defaults.py) the properties 
      *TRANSFER_EVAL.PRETRAIN_{TRAIN,TEST}_USERS_GROUP_WANDB* with the corresponding WandB groupname. Later runs will use
      this to get delta results. (e.g. for OAG automatically, and for HAG in a separate run in [reproduce/hindsight_performance/user_transfer_matrix](reproduce/hindsight_performance/user_transfer_matrix/run.sh)).

### Phase (2): Reproduce Online User-adaptation

All users start from the same pretrained model by following the steps in phase (1). To reproduce results from the paper,
run the scripts in [reproduce](reproduce), containing per experiment a subdirectory with config file
[cfg.yaml]() and script to run the experiment [run.sh](). See the [README](reproduce/README.md) for more details.

- [reproduce/pretrain](reproduce/pretrain): Train population model in pretraining phase
- Empirical study on train user-split:
    - [reproduce/momentum](reproduce/momentum): Online finetuning and momentum strengths ablation.
    - [reproduce/multiple_updates_per_batch](reproduce/multiple_updates_per_batch): Online finetuning for more than 1
      update per batch.
    - [reproduce/non_stationarity_analysis](reproduce/non_stationarity_analysis): Label window predictor
    - [reproduce/replay_strategies](reproduce/replay_strategies): Replay with storage strategies
      FIFO/Hybrid-CBRS/Reservoir.
    - [reproduce/user_feature_adaptation](reproduce/user_feature_adaptation): Fixing the feature extractor/classifier.
- [reproduce/test_user_results](reproduce/test_user_results): Final results on test user-split
- [reproduce/hindsight_performance](reproduce/hindsight_performance): Get hindsight results (HAG) after learning
  user-streams


An example of an experiment to get both the OAG and HAG results:

    # Get online Finetuning results over 10 user streams in U_train
    # Postprocessing automatically aggregates the user stream results (OAG) and uploads to WandB
    cd ./reproduce/momentum/SGD && ./run.sh 

    # To get hindsight performance results on final models (HAG metrics)
    # First, set the previous experiment's group name as WANDB_GROUP_TO_EVAL in 'reproduce/hindsight_performance/cfg.yaml'
    cd ./reproduce/hindsight_performance && ./run.sh 

### (postprocess) Parsing final WandB results

**In WandB**: All results are saved in the WandB run entries and in local CSV dumps with per-update predictions. To get
the results in WandB, group runs (1 run is 1 user-stream) on the *Group* property, and select the relevant metrics.

**Parse to Latex**: Additionally, to see which metrics are used in the paper, you can check
out [the table parser script](src/continual_ego4d/processing/run_csv_to_latex_table_parser.py) for examples. The script
parses downloaded (grouped) runs from WandB that are exported to a CSV, and parses the CSV to Latex-formatted tables.


# Guide
## Tools and plots
We provide a range of tools to explore the dataset and notebooks for the analysis plots reported in the paper.
All can be found in the [notebooks](notebooks) folder.

Tools:
- [Stream meta-data collector](src/continual_ego4d/processing/run_summarize_user_streams.py) processes the user-streams action labels from the JSONS and extracts per user the actual stream samples. This is required for later use in plots and the video player.
- [Video Player](notebooks/EgoAdapt_video_player.ipynb) to display our actual user streams per
  user-split. Displays meta-data such as action (verb,noun) and user-id over time.
- [Postprocess to get decorrelated/correlated AG](src/continual_ego4d/processing/run_postprocess_metrics_dump.py) takes the CSV dumps and calculates the correlated/decorrelated AG metrics.
- [Patch WandB runs](src/continual_ego4d/processing/run_patch_wandb_finished_runs.py) by iterating the local CSV dumps and marking those runs with `finished_run=True` in WandB. This may be caused by syncing errors with WandB.

Plots:
- Dataset stats:
  - [plot_ego4d_user_split_stats.ipynb](notebooks/plot_Ego4D_user_split_stats.ipynb): Video length in minutes (y-axis) per user (x-axis). Color codes the user splits.
  - [plot_user_vs_pretrain_distribution.ipynb](notebooks/plot_EgoAdapt_user_vs_pretrain_distribution.ipynb): Plots the pretrain distribution ordered on frequency, and then
  overlays the test (or train) action distribution on top.
  - [plot_user_action_distribution.ipynb](notebooks/plot_EgoAdapt_user_action_distribution.ipynb): Plots the CDF of the action-histograms in the test user subset.

- Experiments:
  - [plot_classifier_weights_biases.ipynb](notebooks/plot_classifier_weights_biases.ipynb): Analysis for verbs/noun on classifier weight and bias norms.
    Compares SGD on head only vs SGD on full model.
  - [plot_heatmap_transfer.ipynb](notebooks/plot_heatmap_transfer.ipynb): Heatmap that visualizes the user transfer matrix, plotting the  results of user models vs user streams.
  - [plot_heatmap_transfer_user_action_overlap.ipynb](notebooks/plot_heatmap_transfer_user_action_overlap.ipynb): Plot the
  number of overlapping actions between train-users in a heatmap.
  - [plot_forgetting_comparison_SGD_replay.ipynb](notebooks/plot_forgetting_comparison_SGD_replay.ipynb): Re-exposure Forgetting (RF) analysis comparing Replay and
  SGD (2 lines) for RF (y-axis) on log-binned re-exposure count (x-axis).
  - [plot_SGD_per_user_OAG.ipynb](notebooks/plot_SGD_per_user_OAG.ipynb): Plots the cumulative Adaptation Gain (y-axis) over iterations (x-axis), for all train users (10 lines).
  - [plot_multiple_iterations_SGD.ipynb](notebooks/plot_multiple_iterations_SGD.ipynb): Lines for OAG-correlated and OAG-decorrelated metrics on y-axis, and number of updates on
  same batch on x-axis.
  - [plot_SGD_gradient_analysis.ipynb](notebooks/plot_SGD_gradient_analysis.ipynb): Grouped-barplot comparing gradient cosine-similarity of current batch
  with previous points k steps in history of the learning trajectory.

## Resources
**Resources**: The original experiments were mainly executed using 8 A100 GPU's (40G), but the code is adapted to require only around 17G GPU-memory per user stream and supports both sequential and parallel processing of the user-streams.
To further reduce the memory requirements, the batch size can be reduced from 4 (our setting) to 1.

**Parallelism**: 
To speed up experiments, a high level of multi-processing is used. In dataloaders, each process (called worker) loads all images of an entire mini-batch. Multiple workers allow to quickly iterate subsequent batches that are loaded concurrently.

Be aware that the NUM_WORKERS are defined per user-process.
If your job processes get killed because of too many processes, try to lower the NUM_WORKERS.
The NUM_WORKERS are defined for different stages in the pipeline:
- *DATA_LOADER.NUM_WORKERS*: Workers during the pretraining step.
- *CONTINUAL_EVAL.NUM_WORKERS*: Workers for loading the samples in a user stream.
- *PREDICT_PHASE.NUM_WORKERS*: Workers for hindsight performance (in [reproduce/hindsight_performance](reproduce/hindsight_performance)), or to collect pretraining results before user runs (only if *CONTINUAL_EVAL.ONLINE_OAG* is True).
- *METHOD.REPLAY.NUM_WORKERS*: Workers for the loader to load the additional Experience Replay (ER) samples

## Important Configs

We describe some of the important config parameters below.

General config:

- **DATA.PATH_PREFIX**: Ego4d videos parent path.

Ego4d Pretrain:

[//]: # (- **DATA.PATH_TO_DATA_DIR**: Default Ego4d annotations parent path, uses original Ego4d 'fho_lta_{train,val,test}.json' annotations splits.)

- **DATA.PATH_TO_DATA_FILE.{TRAIN,VAL,TEST}**: Define specific annotation file for EgoAdapt pretraining, e.g. by default
  defines TRAIN as the population user-split, and VAL as the train user-split.

EgoAdapt:

- **DATA.PATH_TO_DATA_SPLIT_JSON.{TRAIN/TEST/PRETRAIN}_SPLIT**: The user split paths used for the EgoAdapt streams.
- **CONTINUAL_EVAL.ONLINE_OAG**: Runs prediction phase on pretrained network before starting training in order to
  calculate the OAG/HAG.
    - Another option is to set this to False, calculate the pretraining results for all streams once, and calculate
      deltas in ad-hoc fashion.
- **RESUME_OUTPUT_DIR**: Resume processing of user streams and skip already processed user streams in the given output
  dir.For example, the output directory of a run would look like: *"PROJECT_ROOT"
  /results/momentum/SGD/logs/GRID_SOLVER-BASE_LR=0-1_SOLVER-MOMENTUM=0-0_SOLVER-NESTEROV=True/2023-03-10_17-24-39_UIDf3d2c6ca-6422-4fd0-a9bd-92185be24ab0*
  
## License
EgoAdapt is [MIT-licensed](https://opensource.org/license/mit/). The license applies to the pre-trained models as well.

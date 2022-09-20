"""
After training user-streams, evaluate on the final models with the full user-streams.


1) From a WandB group you are interested in copy the OUTPUT_DIR as input to this script:
/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/exps/ego4d_action_recog/exp04_01_momentum_video_reset/../../../results/ego4d_action_recog/exp04_01_momentum_video_reset/logs/GRID_SOLVER-BASE_LR=0-01_SOLVER-MOMENTUM=0-9_SOLVER-NESTEROV=True/2022-09-16_17-34-18_UID8cae3077-fb05-4368-b3d6-95cb2813e823

2) Given the parent outputdir of the run over users, this script gets for each user:
- the checkpoint: in PARENT_DIR/checkpoints/<USER>/last.ckpt
- the final dump: in PARENT_DIR/user_logs/<USER>/dumpfile.pth

3) Iterate one datastream, 1 model for simplicity. But we can run multiple of these processes concurrently.
- Init a new ContinualMultiTaskClassificationTask for each new model-stream pair:
- Run the ContinualMultiTaskClassificationTask in test-phase, which is similar to predict phase, only it will use the
test_loader attr and test_step() methods.
Start with pretrained model, collect in prediction phase: Loss action/noun/verb
-




We load the full Lightning checkpoint model, not just the weights.
TODO: Check if this contains the pretrain_loss already

Then, we iterate over the entire datastreams for the different user-models.
TODO: Can we implement this in the test-phase of the PL model?
TODO: Can we make U Trainers for the U models, and use only 1 trainer to iterate the data stream, and iterate over the other
trainers to get results for all models?
"""

import copy
import sys

from continual_ego4d.utils.checkpoint_loading import load_slowfast_model_weights, PathHandler
import multiprocessing as mp
from continual_ego4d.run_recog_CL import load_datasets_from_jsons, get_user_ids, get_device_ids

import pprint
import concurrent.futures
from collections import deque
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, DeviceStatsMonitor, GPUStatsMonitor, Timer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from continual_ego4d.utils.custom_logger_connector import CustomLoggerConnector
from pytorch_lightning.loggers import WandbLogger
import traceback

from ego4d.config.defaults import set_cfg_by_name, convert_cfg_to_flat_dict, convert_flat_dict_to_cfg
from ego4d.utils import logging
from ego4d.utils.parser import load_config, parse_args
from ego4d.tasks.long_term_anticipation import MultiTaskClassificationTask

from continual_ego4d.tasks.continual_action_recog_task import ContinualMultiTaskClassificationTask
from continual_ego4d.datasets.continual_action_recog_dataset import extract_json
from continual_ego4d.utils.scheduler import SchedulerConfig, RunConfig

from scripts.slurm import copy_and_run_with_config
import os

from continual_ego4d.utils.models import freeze_full_model, model_trainable_summary
from continual_ego4d.processing.utils import get_group_run_iterator

from fvcore.common.config import CfgNode
import argparse

import wandb

logger = logging.get_logger(__name__)


def parse_wandb_runs(project_name, group_name):
    """ For a given group name in a project, return the user-runs and get the training config. """
    user_to_config = {}

    # summary = final value (excludes NaN rows)
    # history() = gives DF of all values (includes NaN entries for multiple logs per single train/global_step)
    for user_run in get_group_run_iterator(project_name, group_name, finished_runs=True):
        user_id = user_run.config['DATA.COMPUTED_USER_ID']
        user_to_config[user_id] = copy.deepcopy(user_run.config)

    return user_to_config


def main():
    """ Iterate users and aggregate. """
    args = parse_args()
    eval_cfg = load_config(args)

    # GIVE AS INPUT THE GROUP-ID WANDB
    wandb_api = wandb.Api()
    project_name = eval_cfg.TRANSFER_EVAL.WANDB_PROJECT_NAME.strip()
    group_name = eval_cfg.TRANSFER_EVAL.WANDB_GROUP_TO_EVAL.strip()

    # Retrieve all USERS and OUTPUT_DIRS
    user_to_flat_cfg = parse_wandb_runs(project_name, group_name)

    if len(user_to_flat_cfg) == 0:
        print("No finished user runs found in group")
        exit(0)

    # Filter
    # Should all have same outputdir in same group
    outputdirs = [user_config['OUTPUT_DIR'] for user_config in user_to_flat_cfg.values()]
    assert len(set(outputdirs)) == 1, f"Users not same group dir: {set(outputdirs)}"

    train_group_outputdir = outputdirs[0]
    user_to_runid = {user: config['RUN_UID'] for user, config in user_to_flat_cfg.items()}

    # Get checkpoint paths
    user_to_checkpoint_path = {}
    train_path_handler = None
    for user in user_to_runid.keys():
        train_path_handler = PathHandler(
            main_output_dir=train_group_outputdir,
            run_group_id=group_name,
            run_uid=user_to_runid[user],
            is_resuming_run=True,
        )
        user_to_checkpoint_path[user] = train_path_handler.get_user_checkpoints_dir(
            user_id=user, include_ckpt_file='last.ckpt'
        )

    # Set current config with this config: Used to make trainer with same hyperparams on stream
    train_cfg: CfgNode = convert_flat_dict_to_cfg(
        next(iter(user_to_flat_cfg.values())),
        key_exclude_set={'DATA.COMPUTED_USER_ID',
                         'DATA.COMPUTED_USER_DS_ENTRIES',
                         'COMPUTED_USER_DUMP_FILE'}
    )
    pretrain_modelpath = train_cfg['CHECKPOINT_FILE_PATH']

    # SET EVAL OUTPUT_DIR
    eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_OUTPUTDIR = os.path.join(train_group_outputdir, 'transfer_eval')
    logging.setup_logging(eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_OUTPUTDIR, host_name='MASTER', overwrite_logfile=False)
    logger.info(f"Starting main script with OUTPUT_DIR={eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_OUTPUTDIR}")

    # Dataset lists from json / users to process
    user_datasets = load_datasets_from_jsons(train_cfg)
    processed_user_ids, all_user_ids = get_user_ids(train_cfg, user_datasets, train_path_handler)
    assert len(processed_user_ids) == len(all_user_ids), \
        f"Not all users were processed in training: {set(all_user_ids) - set(processed_user_ids)}"

    # CFG overwrites and setup
    overwrite_config_transfer_eval(eval_cfg)
    logger.info("Run with eval_cfg:")
    logger.info(pprint.pformat(eval_cfg))

    # Sequential/parallel execution user jobs
    available_device_ids = get_device_ids(eval_cfg)
    assert len(available_device_ids) >= 1

    # Get run entries
    run_entries = []
    for user_id in all_user_ids:
        user_train_cfg = copy.deepcopy(train_cfg)
        user_train_cfg.DATA.COMPUTED_USER_ID = user_id
        user_train_cfg.DATA.COMPUTED_USER_DS_ENTRIES = user_datasets[user_id]
        user_train_cfg.COMPUTED_USER_DUMP_FILE = train_path_handler.get_user_streamdump_file(user_id)

        run_entries.append(
            RunConfig(
                run_id=user_id,
                target_fn=eval_single_model_single_stream,
                fn_args=(
                    user_train_cfg,  # To recreate data stream
                    eval_cfg,  # For
                    user_to_checkpoint_path[user_id],
                    user_id,
                )
            )
        )

    scheduler_cfg = SchedulerConfig(
        run_entries=run_entries,
        processed_run_ids=[],
        available_device_ids=available_device_ids,
        max_runs_per_device=eval_cfg.NUM_USERS_PER_DEVICE,
    )

    if scheduler_cfg.is_all_runs_processed():
        logger.info("All users already processed, skipping execution. "
                    f"All users={scheduler_cfg.all_run_ids}, "
                    f"processed={scheduler_cfg.processed_run_ids}")
        return

    scheduler_cfg.schedule()

    logger.info("Finished processing all users")
    logger.info(f"All results over users can be found in OUTPUT-DIR={eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_OUTPUTDIR}")


def overwrite_config_transfer_eval(eval_cfg):
    overwrite_dict = {
        "SOLVER.ACCELERATOR": "gpu",
        "NUM_SHARDS": 1,  # no DDP supported
        "CHECKPOINT_LOAD_MODEL_HEAD": True,  # From pretrain we also load model head
    }

    for hierarchy_k, v in overwrite_dict.items():
        set_cfg_by_name(eval_cfg, hierarchy_k, v)

    logger.debug(f"OVERWRITING CFG attributes for transfer eval:\n{pprint.pformat(overwrite_dict)}")


def eval_single_model_single_stream(
        # Scheduling args
        mp_queue: mp.Queue,
        device_id: int,
        run_id: str,

        # additional args
        user_train_cfg: CfgNode,  # To recreate data stream
        eval_cfg: CfgNode,  # For
        load_model_path: str,

        user_id: str,
) -> (bool, str, str):
    """ Run single user sequentially. Returns path to user results and interruption status."""
    seed_everything(eval_cfg.RNG_SEED)

    # TODO make compat for pretrain stream

    run_outputdir = os.path.join(eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_OUTPUTDIR, 'results', user_id)
    run_logdir = os.path.join(eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_OUTPUTDIR, 'logs', user_id)

    # Loggers
    logging.setup_logging(  # Stdout logging
        [run_logdir],
        host_name=f'RUN-{run_id}|GPU-{device_id}|PID-{os.getpid()}',
        overwrite_logfile=False,
    )

    # Choose task type based on config.
    logger.info("Starting init Task")
    if user_train_cfg.DATA.TASK == "continual_classification":
        task = ContinualMultiTaskClassificationTask(user_train_cfg)

        # Add outputhead masker so checkpoint can load the params
        task.configure_head()

    else:
        raise ValueError(f"cfg.DATA.TASK={user_train_cfg.DATA.TASK} not supported")
    logger.info(f"Initialized task as {task}")

    # LOAD PRETRAINED
    load_slowfast_model_weights(
        load_model_path, task, eval_cfg.CHECKPOINT_LOAD_MODEL_HEAD,
    )

    # Freeze model fully
    freeze_full_model(task.model)
    model_trainable_summary(task.model)  # Print summary

    # There are no validation/testing phases!
    logger.info("Initializing Trainer")
    trainer = Trainer(
        # default_root_dir=main_output_dir,  # Default path for logs and weights when no logger/ckpt_callback passed
        accelerator="gpu",  # cfg.SOLVER.ACCELERATOR, only single device for now
        benchmark=True,
        replace_sampler_ddp=False,  # Disable to use own custom sampler
        fast_dev_run=False,  # For CL Should NOT define fast_dev_run in lightning! Doesn't log results then

        # Devices/distributed
        devices=[device_id],
        gpus=None,  # Determined by devices
        num_nodes=1,  # DDP specific

        callbacks=[GPUStatsMonitor()],
        logger=False,
    )

    logger.info("PREDICTING (INFERENCE)")
    sample_idx_to_loss_dict_list: list[dict] = trainer.predict(task)

    # TODO FLATTEN

    sample_idx_to_losses_dict = {k: v for iter_dict in sample_idx_to_loss_dict_list for k, v in iter_dict.items()}

    # Get array based on sample_idx
    # sample_idx_to_loss = [None] * len(sample_idx_to_losses_dict)
    # for sample_idx,
    import pandas as pd
    df = pd.DataFrame(sample_idx_to_losses_dict)

    import pdb;
    pdb.set_trace()

    # TODO SAVE
    torch.save({}, os.path.join(run_outputdir, 'results.pth'))

    # Cleanup process GPU-MEM allocation (Only process context will remain allocated)
    torch.cuda.empty_cache()

    ret = (False, device_id, run_id)
    if mp_queue is not None:
        mp_queue.put(ret)

    return ret  # For multiprocessing indicate which resources are free now


if __name__ == "__main__":
    main()

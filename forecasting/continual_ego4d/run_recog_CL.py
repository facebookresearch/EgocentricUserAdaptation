"""

Train vs test
---------------------------------------------
TRAIN/EVAL are separate, set via: cfg.TRAIN.ENABLE: True
Can do both:     if cfg.TRAIN.ENABLE and cfg.TEST.ENABLE:



Pretraining
---------------------------------------------
cfg:
    CHECKPOINT_LOAD_MODEL_HEAD False \ -> Useful for using pretrained feats.
    MODEL.FREEZE_BACKBONE True \ -> Useful for using pretrained feat-net and freezing it
    DATA.CHECKPOINT_MODULE_FILE_PATH "" \ -> Checkpoint for the MVIT or SlowFAST video net
    FORECASTING.AGGREGATOR "" \
    FORECASTING.DECODER


Flow
---------------------------------------------
run_lta.py: trainer.fit(task: LongTermAnticipationTask)
-> Select task (LTA/short-term recognition/...), initialize it with the config, restore checkpoint,
-> and launch pytorch ligthning trainer to fit the Task.

\ego4d\tasks\long_term_anticipation.py: class LongTermAnticipationTask(VideoTask):
-> Define {train/val/test}_{step/epoch_end}

\ego4d\tasks\video_task.py: class VideoTask(LightningModule)
-> The cfg is used all the way: cfg.MODEL.NUM_CLASSES
-> Build model/optimizer/dataloaders for the task/additional hooks (e.g. on_backwards)


"""
from continual_ego4d.utils.checkpoint_loading import load_pretrain_model

import pprint

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from ego4d.utils import logging
from ego4d.utils.parser import load_config, parse_args
from ego4d.tasks.long_term_anticipation import MultiTaskClassificationTask

from continual_ego4d.tasks.continual_action_recog_task import ContinualMultiTaskClassificationTask
from continual_ego4d.datasets.continual_action_recog_dataset import get_user_to_dataset_dict

from scripts.slurm import copy_and_run_with_config
import os
import os.path as osp

logger = logging.get_logger(__name__)


def main(cfg):
    """ Iterate users and aggregate. """
    resuming_run = len(cfg.RESUME_OUTPUT_DIR) > 0
    if resuming_run:
        cfg.OUTPUT_DIR = cfg.RESUME_OUTPUT_DIR  # Resume run if specified, and output to same output dir

    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info(f"Starting main script with OUTPUT_DIR={cfg.OUTPUT_DIR}")

    # CFG overwrites and setup
    overwrite_config_continual_learning(cfg)
    logger.info("Run with config:")
    logger.info(pprint.pformat(cfg))

    # Select user-split file based on config: Either train or test:
    assert cfg.DATA.USER_SUBSET in ['train', 'test'], \
        "Choose either 'train' or 'test' mode, TRAIN is the user-subset for hyperparam tuning, TEST is held-out final eval"
    data_paths = {
        'train': cfg.DATA.PATH_TO_DATA_SPLIT_JSON.TRAIN_SPLIT,
        'test': cfg.DATA.PATH_TO_DATA_SPLIT_JSON.TEST_SPLIT
    }
    data_path = data_paths[cfg.DATA.USER_SUBSET]

    usersplit_annotations = get_user_to_dataset_dict(data_path)
    logger.info(f'Running JSON USER SPLIT "{cfg.DATA.USER_SUBSET}" in path: {data_path}')

    # Load Meta-loop state checkpoint (Only 1 checkpoint per user, after user-stream finished)
    meta_checkpoint_path = osp.join(cfg.OUTPUT_DIR, 'meta_checkpoint.pth')
    if resuming_run:
        logger.info(f"Resuming run from {cfg.OUTPUT_DIR}")
        assert osp.isfile(meta_checkpoint_path), \
            f"Can't resume run, because no meta checkpoint missing: {meta_checkpoint_path}"
        meta_checkpoint = torch.load(meta_checkpoint_path)
        processed_user_ids = meta_checkpoint['processed_user_ids']
        logger.debug(f"LOADED META CHECKPOINT: {meta_checkpoint}, from path: {meta_checkpoint_path}")
    else:
        processed_user_ids = []

    # Iterate user datasets
    user_result_paths = {}
    user_ids_s = sorted([u for u in usersplit_annotations.keys()])  # Deterministic user order
    for user_id in user_ids_s:
        cfg.DATA.USER_ID = user_id
        cfg.DATA.USER_DS_ENTRIES = usersplit_annotations[user_id]

        user_result_path, interrupted = online_adaptation_single_user(cfg, user_id, processed_user_ids)
        if interrupted:
            logger.debug(f"Shutting down on USER {user_id}, because of Trainer being Interrupted")
            raise Exception()

        # Update and save state
        user_result_paths[user_id] = user_result_path
        processed_user_ids.append(user_id)
        torch.save({'processed_user_ids': processed_user_ids}, meta_checkpoint_path)

    # TODO aggregate metrics over user dumps
    logger.info(f"All results over users can be found in OUTPUT-DIR={cfg.OUTPUT_DIR}")


def overwrite_config_continual_learning(cfg):
    overwrite_dict = {
        "SOLVER.ACCELERATOR": "gpu",
        "NUM_GPUS": 1,
        "NUM_SHARDS": 1,
        "SOLVER.MAX_EPOCH": 1,
        "SOLVER.LR_POLICY": "constant",
        "CHECKPOINT_LOAD_MODEL_HEAD": True,  # From pretrain we also load model head
    }

    for hierarchy_k, v in overwrite_dict.items():
        keys = hierarchy_k.split('.')
        target_obj = cfg
        for idx, key in enumerate(keys):  # If using dots, set in hierarchy of objects, not as single dotted-key
            if idx == len(keys) - 1:  # is last
                setattr(target_obj, key, v)
            else:
                target_obj = getattr(target_obj, key)
    logger.debug(f"OVERWRITING CFG attributes for continual learning:\n{pprint.pformat(overwrite_dict)}")


def online_adaptation_single_user(cfg, user_id, processed_user_ids) -> (str, bool):
    """ Run single user sequentially. Returns path to user results and interruption status."""
    seed_everything(cfg.RNG_SEED)
    logger.info(f"{'*' * 20} USER {user_id} (seed={cfg.RNG_SEED}) {'*' * 20}")

    # Paths
    main_output_dir = cfg.OUTPUT_DIR
    experiment_version = f"user_{user_id.replace('.', '-')}"

    # Loggers
    assert cfg.ENABLE_LOGGING, "Need CSV logging to aggregate results afterwards."
    tb_logger = TensorBoardLogger(save_dir=main_output_dir, name=f"tb", version=experiment_version)
    csv_logger = CSVLogger(save_dir=main_output_dir, name="user_logs", version=experiment_version,
                           flush_logs_every_n_steps=1)
    trainer_loggers = [tb_logger, csv_logger]
    cfg.USER_RESULT_PATH = csv_logger.log_dir  # Use for CSV and other dumps
    cfg.USER_DUMP_FILE = osp.join(cfg.USER_RESULT_PATH, 'stream_info_dump.pth')  # Dump-path for Trainer stream info

    # SKIP PROCESSED USER
    if user_id in processed_user_ids:
        logger.info(f"Skipping USER {user_id} as already processed, result_path={cfg.USER_RESULT_PATH}")
        return cfg.USER_RESULT_PATH, False

    # Callbacks
    # Save model on end of stream for possibly ad-hoc usage of the model
    checkpoint_dirpath = os.path.join(main_output_dir, 'checkpoints', experiment_version)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        every_n_epochs=1, save_on_train_epoch_end=True, save_last=True, save_top_k=1,
    )
    trainer_callbacks = [checkpoint_callback, DeviceStatsMonitor()]  # LearningRateMonitor(),

    # Choose task type based on config.
    logger.info("Starting init Task")
    assert cfg.DATA.TASK == "classification", "Only action recognition supported, no LTA"
    task = ContinualMultiTaskClassificationTask(cfg)

    # LOAD PRETRAINED
    ckpt_task_types = [MultiTaskClassificationTask, ContinualMultiTaskClassificationTask]
    load_pretrain_model(cfg, cfg.CHECKPOINT_FILE_PATH, task, ckpt_task_types)

    # GPU DEVICE
    # Make sure it's an array to define the GPU-ids. A single int indicates the number of GPUs instead.
    if cfg.GPU_IDS is not None:
        cfg.NUM_GPUS = None  # Need to disable
        # /accelerator_connector.py:266: UserWarning: The flag `devices=[7]` will be ignored, as you have set `gpus=1`
        if isinstance(cfg.GPU_IDS, int):
            cfg.GPU_IDS = [cfg.GPU_IDS]
        elif isinstance(cfg.GPU_IDS, str):
            cfg.GPU_IDS = list(map(int, cfg.GPU_IDS.split(',')))

    # There are no validation/testing phases!
    logger.info("Initializing Trainer")
    trainer = Trainer(
        # default_root_dir=main_output_dir,  # Default path for logs and weights when no logger/ckpt_callback passed
        accelerator=cfg.SOLVER.ACCELERATOR,  # cfg.SOLVER.ACCELERATOR, only single device for now
        max_epochs=cfg.SOLVER.MAX_EPOCH,
        num_sanity_val_steps=0,  # Sanity check before starting actual training to make sure validation works
        benchmark=True,
        replace_sampler_ddp=False,  # Disable to use own custom sampler
        fast_dev_run=False,  # For CL Should NOT define fast_dev_run in lightning! Doesn't log results then

        # Devices/distributed
        devices=cfg.GPU_IDS,
        gpus=cfg.NUM_GPUS,
        # auto_select_gpus=True,
        # plugins=DDPPlugin(find_unused_parameters=False), # DDP specific
        num_nodes=cfg.NUM_SHARDS,  # DDP specific

        callbacks=trainer_callbacks,
        logger=trainer_loggers,
        log_every_n_steps=1,  # Required to allow per-step log-cals for evaluation
    )

    logger.info("Starting Trainer fitting")
    trainer.fit(task, val_dataloaders=None)  # Skip validation

    return cfg.USER_RESULT_PATH, trainer.interrupted


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    if args.on_cluster:
        copy_and_run_with_config(
            main,
            cfg,
            args.working_directory,
            job_name=args.job_name,
            time="72:00:00",
            partition="devlab,learnlab,learnfair",
            gpus_per_node=cfg.NUM_GPUS,
            ntasks_per_node=cfg.NUM_GPUS,
            cpus_per_task=10,
            mem="470GB",
            nodes=cfg.NUM_SHARDS,
            constraint="volta32gb",
        )
    else:  # local
        main(cfg)

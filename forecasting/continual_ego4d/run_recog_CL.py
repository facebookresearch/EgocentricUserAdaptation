import sys

from continual_ego4d.utils.checkpoint_loading import load_pretrain_model, load_meta_state, save_meta_state, PathHandler

import pprint
import concurrent.futures
from collections import deque
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, DeviceStatsMonitor, GPUStatsMonitor, Timer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from continual_ego4d.utils.custom_logger_connector import CustomLoggerConnector

from ego4d.utils import logging
from ego4d.utils.parser import load_config, parse_args
from ego4d.tasks.long_term_anticipation import MultiTaskClassificationTask

from continual_ego4d.tasks.continual_action_recog_task import ContinualMultiTaskClassificationTask
from continual_ego4d.tasks.iid_action_recog_task import IIDMultiTaskClassificationTask
from continual_ego4d.datasets.continual_action_recog_dataset import get_user_to_dataset_dict

from scripts.slurm import copy_and_run_with_config
import os
import os.path as osp
import shutil

logger = logging.get_logger(__name__)


def main(cfg):
    """ Iterate users and aggregate. """
    resuming_run = len(cfg.RESUME_OUTPUT_DIR) > 0
    if resuming_run:
        cfg.OUTPUT_DIR = cfg.RESUME_OUTPUT_DIR  # Resume run if specified, and output to same output dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    logging.setup_logging(cfg.OUTPUT_DIR, host_name='MASTER', overwrite_logfile=False)
    logger.info(f"Starting main script with OUTPUT_DIR={cfg.OUTPUT_DIR}")

    # CFG overwrites and setup
    overwrite_config_continual_learning(cfg)
    logger.info("Run with config:")
    logger.info(pprint.pformat(cfg))

    # Copy files to output dir for reproducing
    for reproduce_path in [cfg.PARENT_SCRIPT_FILE_PATH, cfg.CONFIG_FILE_PATH]:
        shutil.copy2(reproduce_path, cfg.OUTPUT_DIR)

    # Select user-split file based on config: Either train or test:
    assert cfg.DATA.USER_SUBSET in ['train', 'test'], \
        "Choose either 'train' or 'test' mode, TRAIN is the user-subset for hyperparam tuning, TEST is held-out final eval"
    data_paths = {
        'train': cfg.DATA.PATH_TO_DATA_SPLIT_JSON.TRAIN_SPLIT,
        'test': cfg.DATA.PATH_TO_DATA_SPLIT_JSON.TEST_SPLIT
    }
    data_path = data_paths[cfg.DATA.USER_SUBSET]

    user_datasets = get_user_to_dataset_dict(data_path)
    all_user_ids_s = sorted([u for u in user_datasets.keys()])  # Deterministic user order
    logger.info(f'Running JSON USER SPLIT "{cfg.DATA.USER_SUBSET}" in path: {data_path}')

    # Load Meta-loop state checkpoint (Only 1 checkpoint per user, after user-stream finished)
    processed_user_ids = []
    path_handler = PathHandler(cfg)
    if resuming_run:
        logger.info(f"Resuming run from {cfg.OUTPUT_DIR}")
        processed_user_ids = path_handler.get_processed_users_from_final_dumps()
        logger.debug(f"LOADED META CHECKPOINT: Processed users = {processed_user_ids}")

        # If ONLY TESTING, then assume all users have been processed
        if cfg.TEST.ENABLE and not cfg.TRAIN.ENABLE:
            logger.info(f"TEST-ONLY MODE: assuming all users have been processed")

            # Checks
            assert len(processed_user_ids) == len(all_user_ids_s), \
                f"Only {len(processed_user_ids)}/{len(all_user_ids_s)} users processed for test: {processed_user_ids}"
            assert len(cfg.CHECKPOINT_FILE_PATH) > 0, "Need a model path to load for testing"
            assert cfg.CHECKPOINT_LOAD_MODEL_HEAD, f"Need to load head for testing mode"

            processed_user_ids = []  # For testing, all still have to be processed

    # Sequential/parallel execution user jobs
    device_ids = get_device_ids(cfg)
    assert len(device_ids) >= 1

    if len(device_ids) == 1:
        process_users_sequentially(user_datasets, processed_user_ids, path_handler, device_ids, all_user_ids_s)
    else:
        process_users_parallel(user_datasets, processed_user_ids, path_handler, device_ids, all_user_ids_s)
    logger.info("Finished processing all users")


def process_users_parallel(
        user_datasets: dict[list[tuple]],
        processed_user_ids: list[str],
        path_handler: PathHandler,
        device_ids: list[int],
        all_user_ids: list[str],
):
    """ This process is master process that spawn user-specific processes on free GPU devices once they are free. """
    users_to_process = deque([u for u in all_user_ids if u not in processed_user_ids])
    nb_available_devices = len(device_ids)

    # Multi-process env
    with concurrent.futures.ProcessPoolExecutor(max_workers=nb_available_devices) as executor:
        futures = []

        def submit_userprocesses_on_free_devices(free_device_ids):
            for device_id in free_device_ids:
                if len(users_to_process) == 0:
                    logger.info(f"All users processed, skipping allocation to new devices")
                    return
                user_id = users_to_process.popleft()
                logger.info(f"{'*' * 20} USER {user_id} (device_id={device_id}) {'*' * 20}")
                futures.append(
                    executor.submit(
                        online_adaptation_single_user,
                        cfg, user_id, user_datasets[user_id], [device_id], path_handler
                    )
                )

        # Initially fill all devices with user-processes
        submit_userprocesses_on_free_devices(device_ids)

        for future in concurrent.futures.as_completed(futures):  # Firs completed in async process pool
            interrupted, device_ids, user_id = future.result()
            logger.info(f"Finished processing user {user_id}")

            if interrupted:
                logger.exception(f"Process for USER {user_id} failed because of Trainer being Interrupted")
                continue

            # Save results
            # processed_user_ids.append(user_id)
            # save_meta_state(path_handler.meta_checkpoint_path, user_id)

            # Start new user-processes on free devices
            if len(users_to_process) > 0:
                logger.info(f"Submitting processes for free devices {len(device_ids)}")
                submit_userprocesses_on_free_devices(device_ids)


def process_users_sequentially(
        user_datasets: dict[list[tuple]],
        processed_user_ids: list[str],
        path_handler: PathHandler,
        device_ids: list[int],
        all_user_ids: list[str],
):
    """ Sequentially iterate over users and process on single device.
    All processing happens in master process. """

    # Iterate user datasets
    for user_id in all_user_ids:
        if user_id in processed_user_ids:  # SKIP PROCESSED USER
            logger.info(f"Skipping USER {user_id} as already processed, result_path={cfg.OUTPUT_DIR}")

        interrupted, *_ = online_adaptation_single_user(
            cfg,
            user_id,
            user_datasets[user_id],
            device_ids,
            path_handler
        )
        if interrupted:
            logger.exception(f"Shutting down on USER {user_id}, because of Trainer being Interrupted")
            raise Exception()

        # Update and save state
        # processed_user_ids.append(user_id)
        # save_meta_state(path_handler.meta_checkpoint_path, user_id)

    logger.info(f"All results over users can be found in OUTPUT-DIR={cfg.OUTPUT_DIR}")


def overwrite_config_continual_learning(cfg):
    overwrite_dict = {
        "SOLVER.ACCELERATOR": "gpu",
        "NUM_SHARDS": 1,  # no DDP supported
        # "SOLVER.MAX_EPOCH": 1, # Allow multiple for IID
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


def get_device_ids(cfg) -> list[int]:
    """
    Make sure it's an array to define the GPU-ids. A single int indicates the number of GPUs instead.
    :return:
    """
    if cfg.GPU_IDS is None:
        assert isinstance(cfg.NUM_GPUS, int) and cfg.NUM_GPUS >= 1
        device_ids = list(range(cfg.NUM_GPUS))  # Select first devices
    else:
        cfg.NUM_GPUS = None  # Need to disable
        if isinstance(cfg.GPU_IDS, int):
            device_ids = [cfg.GPU_IDS]
        elif isinstance(cfg.GPU_IDS, str):
            device_ids = list(map(int, cfg.GPU_IDS.split(',')))
        else:
            raise ValueError(f"cfg.GPU_IDS wrong format: {cfg.GPU_IDS}")

    return device_ids


def online_adaptation_single_user(
        cfg,
        user_id: str,
        user_dataset: list[tuple],
        device_ids: list[int],
        path_handler: PathHandler,
) -> (str, bool):
    """ Run single user sequentially. Returns path to user results and interruption status."""
    seed_everything(cfg.RNG_SEED)

    # Set user configs
    cfg.DATA.USER_ID = user_id
    cfg.DATA.USER_DS_ENTRIES = user_dataset

    # Paths
    cfg.USER_DUMP_FILE = path_handler.get_user_streamdump_file(user_id)  # Dump-path for Trainer stream info

    # Loggers
    logging.setup_logging(  # Stdout logging
        [path_handler.get_user_results_dir(user_id)],
        host_name=f'GPU-{device_ids}|USER-{user_id}',
        overwrite_logfile=False,
    )

    assert cfg.ENABLE_LOGGING, "Need CSV logging to aggregate results afterwards."
    tb_logger = TensorBoardLogger(
        save_dir=path_handler.main_output_dir,
        name=path_handler.tb_dirname,
        version=path_handler.get_experiment_version(user_id)
    )
    csv_logger = CSVLogger(
        save_dir=path_handler.main_output_dir,
        name=path_handler.csv_dirname,
        version=path_handler.get_experiment_version(user_id),
        flush_logs_every_n_steps=1
    )
    trainer_loggers = [tb_logger, csv_logger]

    # Callbacks
    # Save model on end of stream for possibly ad-hoc usage of the model
    checkpoint_callback = ModelCheckpoint(
        dirpath=path_handler.get_user_checkpoints_dir(user_id),
        every_n_epochs=1, save_on_train_epoch_end=True, save_last=True, save_top_k=1,
    )
    trainer_callbacks = [
        checkpoint_callback,
        # DeviceStatsMonitor(), # Way too detailed
        GPUStatsMonitor(),
        Timer(duration=None, interval='epoch')
        # LearningRateMonitor(), # Cst LR by default
    ]

    # Choose task type based on config.
    logger.info("Starting init Task")
    if cfg.DATA.TASK == "continual_classification":
        task = ContinualMultiTaskClassificationTask(cfg)

    elif cfg.DATA.TASK == "iid_classification":
        task = IIDMultiTaskClassificationTask(cfg)

    else:
        raise ValueError(f"cfg.DATA.TASK={cfg.DATA.TASK} not supported")
    logger.info(f"Initialized task as {task}")

    # LOAD PRETRAINED
    ckpt_task_types = [MultiTaskClassificationTask,
                       ContinualMultiTaskClassificationTask,
                       IIDMultiTaskClassificationTask]
    load_pretrain_model(cfg, cfg.CHECKPOINT_FILE_PATH, task, ckpt_task_types)

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
        devices=device_ids,
        gpus=None,  # Determined by devices
        # auto_select_gpus=True,
        # plugins=DDPPlugin(find_unused_parameters=False), # DDP specific
        num_nodes=cfg.NUM_SHARDS,  # DDP specific

        callbacks=trainer_callbacks,
        logger=trainer_loggers,
        log_every_n_steps=cfg.TRAIN.CONTINUAL_EVAL_FREQ,  # Required to allow per-step log-cals for evaluation
    )

    # Overwrite (Always log on first step)
    trainer.logger_connector = CustomLoggerConnector(trainer, trainer.logger_connector.log_gpu_memory)

    interrupted = False
    if cfg.TRAIN.ENABLE:
        logger.info("Starting Trainer fitting")

        trainer.fit(task, val_dataloaders=None)  # Skip validation

        interrupted = trainer.interrupted
        logger.info(f"Trainer interrupted signal = {interrupted}")

    if not interrupted and cfg.TEST.ENABLE:
        logger.info("Starting Trainer testing")  # Logs test-metrics using the same loggers
        trainer.test(task)

        interrupted = trainer.interrupted
        logger.info(f"Trainer interrupted signal during testing = {interrupted}")

    return interrupted, device_ids, user_id  # For multiprocessing indicate which resources are free now


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

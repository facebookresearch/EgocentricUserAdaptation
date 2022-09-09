import copy
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
from pytorch_lightning.loggers import WandbLogger
import traceback

from ego4d.config.defaults import set_cfg_by_name, convert_cfg_to_dict
from ego4d.utils import logging
from ego4d.utils.parser import load_config, parse_args
from ego4d.tasks.long_term_anticipation import MultiTaskClassificationTask

from continual_ego4d.tasks.continual_action_recog_task import ContinualMultiTaskClassificationTask
from continual_ego4d.datasets.continual_action_recog_dataset import extract_json

from scripts.slurm import copy_and_run_with_config
import os

from fvcore.common.config import CfgNode

logger = logging.get_logger(__name__)


def main(cfg: CfgNode):
    """ Iterate users and aggregate. """
    path_handler = PathHandler(cfg)

    logging.setup_logging(path_handler.main_output_dir, host_name='MASTER', overwrite_logfile=False)
    logger.info(f"Starting main script with OUTPUT_DIR={path_handler.main_output_dir}")

    # CFG overwrites and setup
    overwrite_config_continual_learning(cfg)
    logger.info("Run with config:")
    logger.info(pprint.pformat(cfg))

    # Dataset lists from json / users to process
    user_datasets = load_datasets_from_jsons(cfg)
    processed_user_ids, all_user_ids = get_user_ids(cfg, user_datasets, path_handler)

    # Sequential/parallel execution user jobs
    available_device_ids = get_device_ids(cfg)
    assert len(available_device_ids) >= 1

    scheduler_cfg = SchedulerConfig(
        available_device_ids=available_device_ids,
        all_user_ids=all_user_ids,
        processed_user_ids=processed_user_ids,
        userprocesses_per_device=cfg.NUM_USERS_PER_DEVICE,
    )

    if len(scheduler_cfg.available_device_ids) == 1:
        process_users_sequentially(cfg, scheduler_cfg, user_datasets, path_handler)
    else:
        process_users_parallel(cfg, scheduler_cfg, user_datasets, path_handler)
    logger.info("Finished processing all users")


def load_datasets_from_jsons(cfg):
    """
    Load the train OR test json.
    The Pretrain action sets are always loaded.
    """
    # Select user-split file based on config: Either train or test:
    assert cfg.DATA.USER_SUBSET in ['train', 'test'], \
        "Choose either 'train' or 'test' mode, TRAIN is the user-subset for hyperparam tuning, TEST is held-out final eval"
    data_paths = {
        'train': cfg.DATA.PATH_TO_DATA_SPLIT_JSON.TRAIN_SPLIT,
        'test': cfg.DATA.PATH_TO_DATA_SPLIT_JSON.TEST_SPLIT,
        'pretrain': cfg.DATA.PATH_TO_DATA_SPLIT_JSON.PRETRAIN_SPLIT,
    }
    data_path = data_paths[cfg.DATA.USER_SUBSET]
    logger.info(f'Running JSON USER SPLIT "{cfg.DATA.USER_SUBSET}" in path: {data_path}')

    # Current training data (for all users)
    datasets_holder = extract_json(data_path)
    user_datasets = datasets_holder['users']  # user-specific datasets

    # Pretraining stats (e.g. action sets), cfg requires COMPUTED_ for dynamically added nodes
    pretrain_dataset_holder = extract_json(data_paths['pretrain'])
    cfg.COMPUTED_PRETRAIN_ACTION_SETS = copy.deepcopy(pretrain_dataset_holder['user_action_sets']['user_agnostic'])
    del pretrain_dataset_holder

    return user_datasets


def get_user_ids(cfg, user_datasets, path_handler):
    """
    Get all user_ids and the subset that is processed.
    """
    # Order users on dataset length (nb annotations as proxy)
    user_to_ds_len = sorted(
        [(user_id, len(user_ds)) for user_id, user_ds in user_datasets.items()],
        key=lambda x: x[1], reverse=True
    )
    all_user_ids = [x[0] for x in user_to_ds_len]

    if cfg.USER_SELECTION is not None:  # Apply user-filter
        user_selection = cfg.USER_SELECTION.split(',')
        for user_id in user_selection:
            assert user_id in all_user_ids, f"Config user-id '{user_id}' is invalid. Define one in {all_user_ids}"
        all_user_ids = list(filter(lambda x: x in user_selection, all_user_ids))

    logger.info(f"Processing users in order: {all_user_ids}, with sizes {user_to_ds_len}")

    # Load Meta-loop state checkpoint (Only 1 checkpoint per user, after user-stream finished)
    processed_user_ids = []
    if path_handler.is_resuming_run:
        logger.info(f"Resuming run from {path_handler.main_output_dir}")
        processed_user_ids = path_handler.get_processed_users_from_final_dumps()
        logger.debug(f"LOADED META CHECKPOINT: Processed users = {processed_user_ids}")

    return processed_user_ids, all_user_ids


class SchedulerConfig:
    """ Config attributes used for scheduling the job either sequentially or parallel."""

    def __init__(self, all_user_ids, processed_user_ids, available_device_ids, userprocesses_per_device):
        self.all_user_ids: list[str] = all_user_ids
        self.processed_user_ids: list[str] = processed_user_ids

        self.userprocesses_per_device = userprocesses_per_device
        assert self.userprocesses_per_device >= 1
        self.available_device_ids: list[int] = available_device_ids * self.userprocesses_per_device


def process_users_parallel(
        cfg: CfgNode,
        scheduler_cfg: SchedulerConfig,
        user_datasets: dict[str, list[tuple]],
        path_handler: PathHandler,
):
    """ This process is master process that spawn user-specific processes on free GPU devices once they are free. """
    users_to_process = deque([u for u in scheduler_cfg.all_user_ids if u not in scheduler_cfg.processed_user_ids])
    nb_available_devices = len(scheduler_cfg.available_device_ids)

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
        submit_userprocesses_on_free_devices(scheduler_cfg.available_device_ids)

        for future in concurrent.futures.as_completed(futures):  # Firs completed in async process pool

            try:
                interrupted, device_ids, user_id = future.result()
                logger.info(f"Finished processing user {user_id}")

            except Exception as e:
                traceback.print_exc()
                logger.info(f'Future {future} terminated and generated an exception: {e}')
                continue

            if interrupted:
                logger.exception(f"Process for USER {user_id} failed because of Trainer being Interrupted."
                                 f"Not releasing GPU as might give Out-of-Memory exception.")
                continue

            # Start new user-processes on free devices
            if len(users_to_process) > 0:
                logger.info(f"Submitting processes for free devices {len(device_ids)}")
                submit_userprocesses_on_free_devices(device_ids)


def process_users_sequentially(
        cfg: CfgNode,
        scheduler_cfg: SchedulerConfig,
        user_datasets: dict[str, list[tuple]],
        path_handler: PathHandler,
):
    """ Sequentially iterate over users and process on single device.
    All processing happens in master process. """

    # Iterate user datasets
    for user_id in scheduler_cfg.all_user_ids:
        if user_id in scheduler_cfg.processed_user_ids:  # SKIP PROCESSED USER
            logger.info(f"Skipping USER {user_id} as already processed, result_path={path_handler.main_output_dir}")

        interrupted, *_ = online_adaptation_single_user(
            copy.deepcopy(cfg),  # Cfg doesn't allow resetting COMPUTED_ attributes
            user_id,
            user_datasets[user_id],
            scheduler_cfg.available_device_ids,
            path_handler
        )
        if interrupted:
            logger.exception(f"Shutting down on USER {user_id}, because of Trainer being Interrupted")
            raise Exception()

    logger.info(f"All results over users can be found in OUTPUT-DIR={path_handler.main_output_dir}")


def overwrite_config_continual_learning(cfg):
    overwrite_dict = {
        "SOLVER.ACCELERATOR": "gpu",
        "NUM_SHARDS": 1,  # no DDP supported
        "SOLVER.MAX_EPOCH": 1,
        "SOLVER.LR_POLICY": "constant",
        "CHECKPOINT_LOAD_MODEL_HEAD": True,  # From pretrain we also load model head
    }

    for hierarchy_k, v in overwrite_dict.items():
        set_cfg_by_name(cfg, hierarchy_k, v)

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
        cfg: CfgNode,
        user_id: str,
        user_dataset: list[tuple],
        device_ids: list[int],
        path_handler: PathHandler,
) -> (bool, str, str):
    """ Run single user sequentially. Returns path to user results and interruption status."""
    seed_everything(cfg.RNG_SEED)

    # Set user configs
    cfg.DATA.COMPUTED_USER_ID = user_id
    cfg.DATA.COMPUTED_USER_DS_ENTRIES = user_dataset

    # Paths
    cfg.COMPUTED_USER_DUMP_FILE = path_handler.get_user_streamdump_file(user_id)  # Dump-path for Trainer stream info

    # Loggers
    logging.setup_logging(  # Stdout logging
        [path_handler.get_user_results_dir(user_id)],
        host_name=f'USER-{user_id}|GPU-{device_ids}|PID-{os.getpid()}',
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
    wandb_logger = WandbLogger(
        project=path_handler.wandb_project_name,
        save_dir=path_handler.get_user_wandb_dir(user_id),  # Make user-specific dir
        name=path_handler.exp_uid,  # Display name for run
        group=path_handler.exp_uid,
        tags=cfg.WANDB.TAGS.split(',') if cfg.WANDB.TAGS is not None else None,
        config=convert_cfg_to_dict(cfg)  # Load full config to wandb setting
    )

    trainer_loggers = [tb_logger, csv_logger, wandb_logger]

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
        Timer(duration=None, interval='epoch'),
        Timer(duration=None, interval='step')
        # LearningRateMonitor(), # Cst LR by default
    ]

    # Choose task type based on config.
    logger.info("Starting init Task")
    if cfg.DATA.TASK == "continual_classification":
        task = ContinualMultiTaskClassificationTask(cfg)

    else:
        raise ValueError(f"cfg.DATA.TASK={cfg.DATA.TASK} not supported")
    logger.info(f"Initialized task as {task}")

    # LOAD PRETRAINED
    ckpt_task_types = [MultiTaskClassificationTask,
                       ContinualMultiTaskClassificationTask]
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
        log_every_n_steps=1,  # Required to allow per-step log-cals for evaluation
    )

    # Overwrite (Always log on first step)
    trainer.logger_connector = CustomLoggerConnector(trainer, trainer.logger_connector.log_gpu_memory)

    interrupted = False
    if cfg.TRAIN.ENABLE:
        # Dependent on task: Might need to run prediction first (gather per-sample results for init model)
        if task.run_predict_before_train:
            logger.info("Starting Trainer Prediction round before fitting")
            trainer.predict(task)  # Skip validation

        logger.info("Starting Trainer fitting")
        trainer.fit(task, val_dataloaders=None)  # Skip validation

        interrupted = trainer.interrupted
        logger.info(f"Trainer interrupted signal = {interrupted}")

    # Cleanup process GPU-MEM allocation (Only process context will remain allocated)
    torch.cuda.empty_cache()
    wandb_logger.experiment.finish()

    return interrupted, device_ids, user_id  # For multiprocessing indicate which resources are free now


if __name__ == "__main__":
    args = parse_args()
    parsed_cfg = load_config(args)
    if args.on_cluster:
        copy_and_run_with_config(
            main,
            parsed_cfg,
            args.working_directory,
            job_name=args.job_name,
            time="72:00:00",
            partition="devlab,learnlab,learnfair",
            gpus_per_node=parsed_cfg.NUM_GPUS,
            ntasks_per_node=parsed_cfg.NUM_GPUS,
            cpus_per_task=10,
            mem="470GB",
            nodes=parsed_cfg.NUM_SHARDS,
            constraint="volta32gb",
        )
    else:  # local
        main(parsed_cfg)

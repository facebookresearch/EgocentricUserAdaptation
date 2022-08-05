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
from continual_ego4d.datasets.continual_action_recog_dataset import Ego4dContinualRecognition


def main(cfg):
    """ Iterate users and aggregate. """
    resuming_run = len(cfg.RESUME_OUTPUT_DIR) > 0
    if resuming_run:
        cfg.OUTPUT_DIR = cfg.RESUME_OUTPUT_DIR  # Resume run if specified, and output to same output dir
    print(f"Output is redirected to: {cfg.OUTPUT_DIR}")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    path_handler = PathHandler(cfg)

    # Assertion bypassing
    cfg.SOLVER.ACCELERATOR = "gpu"

    # Select user-split file based on config: Either train or test:
    assert cfg.DATA.USER_SUBSET in ['train', 'test'], "Usersplits should be train or test"
    data_paths = {
        'train': cfg.DATA.PATH_TO_DATA_SPLIT_JSON.TRAIN_SPLIT,
        'test': cfg.DATA.PATH_TO_DATA_SPLIT_JSON.TEST_SPLIT
    }
    data_path = data_paths[cfg.DATA.USER_SUBSET]

    user_datasets = get_user_to_dataset_dict(data_path)
    all_user_ids_s = sorted([u for u in user_datasets.keys()])  # Deterministic user order
    print(f'Running JSON USER SPLIT "{cfg.DATA.USER_SUBSET}" in path: {data_path}')

    # Iterate user datasets
    checkpoint_filename = f"dataset_entries_{cfg.DATA.USER_SUBSET}_{osp.basename(data_path).split('.')[0]}.ckpt"
    checkpoint_path = osp.join(cfg.OUTPUT_DIR, checkpoint_filename)
    print(f"Dataset checkpoint path={checkpoint_path}")

    if osp.isfile(checkpoint_path):
        datasets = torch.load(checkpoint_path)
    else:
        datasets = {}
        for user_id in all_user_ids_s:
            print(f"Collecting dataset for user {user_id}")
            datasets[user_id] = collect_user_dataset(cfg, user_id, user_datasets[user_id], path_handler)
        print(f"Collected all user datasets, USERS={list(datasets.keys())}")

        torch.save(datasets, checkpoint_path)



def collect_user_dataset(
        cfg,
        user_id: str,
        user_dataset,
        path_handler: PathHandler,
) -> (str, bool):
    """ Run single user sequentially. Returns path to user results and interruption status."""
    seed_everything(cfg.RNG_SEED)

    # Set user configs
    cfg.DATA.USER_ID = user_id
    cfg.DATA.USER_DS_ENTRIES = user_dataset
    cfg.USER_DUMP_FILE = path_handler.get_user_streamdump_file(user_id)  # Dump-path for Trainer stream info
    cfg.USER_RESULT_PATH = path_handler.get_user_results_dir(user_id)

    # Choose task type based on config.
    if cfg.DATA.TASK == "continual_classification":
        task = ContinualMultiTaskClassificationTask(cfg)

    elif cfg.DATA.TASK == "iid_classification":
        task = IIDMultiTaskClassificationTask(cfg)

    else:
        raise ValueError(f"cfg.DATA.TASK={cfg.DATA.TASK} not supported")
    print(f"Initialized task as {type(task)}")

    dataset_obj: Ego4dContinualRecognition = task.train_dataloader().dataset
    dataset_tuple_entries = dataset_obj.seq_input_list

    # Unpack entries and leave out video_paths in (video_path, entry) tuples
    dataset_entries = [t[1] for t in dataset_tuple_entries]

    return dataset_entries


# import pandas as pd
# def plot_user_summary(user_id: str, user_entries: list):
#     """"""
#     # Do all for actions/verbs/nouns
#     user_df = pd.json_normalize(user_entries)  # Convert to DF
#
#     # Plot accum 2s-clips per action + total length over all actions
#     # plot_total_action_length_histogram()
#
#     # Plot Number of actions instead of length + mention max per-clip length (e.g. 2s)
#
#     # Plot like EPIC KITCHENS a normalized bar with same action same color
#     # Also do for verbs/nouns separately


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    main(cfg)

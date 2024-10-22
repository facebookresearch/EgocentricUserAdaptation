# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
As sequential sampling within the action label time-ranges occurs at runtime for the user-streams,
this script collects all samples and pickles them for further analysis in the notebooks.

Run this script with a config file that defines;
    cfg.DATA.USER_SUBSET, # 'train' or 'test'
    cfg.DATA.PATH_TO_DATA_SPLIT_JSON.TRAIN_SPLIT, # The train JSON split
    cfg.DATA.PATH_TO_DATA_SPLIT_JSON.TEST_SPLIT, # The test JSON split
    cfg.DATA.PATH_TO_DATA_SPLIT_JSON.PRETRAIN_SPLIT # The pretrain JSON split
"""
import copy
import os.path as osp
import pickle

from pytorch_lightning import seed_everything

from continual_ego4d.datasets.continual_dataloader import construct_trainstream_loader
from continual_ego4d.run_train_user_streams import load_datasets_from_jsons
from continual_ego4d.tasks.continual_action_recog_task import PretrainState
from continual_ego4d.utils.misc import makedirs
from ego4d.utils import logging
from ego4d.utils.parser import load_config, parse_args

logger = logging.get_logger(__name__)


def main(cfg):
    """ Iterate users and aggregate. """

    resuming_run = len(cfg.RESUME_OUTPUT_DIR) > 0
    if resuming_run:
        cfg.OUTPUT_DIR = cfg.RESUME_OUTPUT_DIR  # Resume run if specified, and output to same output dir
    print(f"Output is redirected to: {cfg.OUTPUT_DIR}")
    makedirs(cfg.OUTPUT_DIR, exist_ok=True, mode=0o777)

    # Logger for dataset
    logging.setup_logging(cfg.OUTPUT_DIR, host_name='MASTER', overwrite_logfile=False)

    # Assertion bypassing
    cfg.SOLVER.ACCELERATOR = "gpu"

    # Dataset loading
    data_paths = {
        'train': cfg.DATA.PATH_TO_DATA_SPLIT_JSON.TRAIN_SPLIT,
        'test': cfg.DATA.PATH_TO_DATA_SPLIT_JSON.TEST_SPLIT,
        'pretrain': cfg.DATA.PATH_TO_DATA_SPLIT_JSON.PRETRAIN_SPLIT,
    }
    data_path = data_paths[cfg.DATA.USER_SUBSET]

    user_datasets = load_datasets_from_jsons(cfg)
    all_user_ids_s = sorted([u for u in user_datasets.keys()])  # Deterministic user order

    # Iterate user datasets
    checkpoint_filename = f"dataset_entries_{cfg.DATA.USER_SUBSET}_{osp.basename(data_path).split('.')[0]}.ckpt"
    checkpoint_path = osp.join(cfg.OUTPUT_DIR, checkpoint_filename)
    print(f"Dataset checkpoint path={checkpoint_path}")
    assert not osp.isfile(checkpoint_path), "Not overwriting summary checkpoint"

    datasets = {}
    for user_id in all_user_ids_s:
        print(f"Collecting dataset for user {user_id}")
        datasets[user_id] = collect_user_dataset(copy.deepcopy(cfg), user_id, user_datasets[user_id])
    print(f"Collected all user datasets, USERS={list(datasets.keys())}")

    # torch.save(datasets, checkpoint_path)
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(datasets, f)

    return checkpoint_path


def collect_user_dataset(
        cfg,
        user_id: str,
        user_dataset,
) -> (str, bool):
    """ Run single user sequentially with sequential ClipSampler, return data list of annotation entries returned by the clip sampler."""
    seed_everything(cfg.RNG_SEED)

    # Set user configs
    cfg.DATA.COMPUTED_USER_ID = user_id
    cfg.DATA.COMPUTED_USER_DS_ENTRIES = user_dataset

    # Set pretrain stats
    cfg.COMPUTED_PRETRAIN_STATE = PretrainState(cfg.COMPUTED_PRETRAIN_ACTION_SETS)

    loader = construct_trainstream_loader(cfg)
    dataset_obj = loader.dataset
    dataset_tuple_entries = dataset_obj.seq_input_list

    # Unpack entries and leave out video_paths in (video_path, entry) tuples
    final_entries = []
    for video_path, video_entry_dict in dataset_tuple_entries:
        video_entry_dict['video_path'] = video_path
        final_entries.append(video_entry_dict)

    return final_entries


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    main(cfg)

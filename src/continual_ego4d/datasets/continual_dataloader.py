# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler

import itertools

from ego4d.datasets.build import build_dataset
from collections import defaultdict
from ego4d.utils import logging

logger = logging.get_logger(__name__)


def construct_trainstream_loader(cfg, shuffle=False):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            ego4d/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    dataset_name = cfg.TRAIN.DATASET  # e.g. Ego4dContinualRecognition
    if cfg.SOLVER.ACCELERATOR not in ["dp", "gpu"]:
        batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
    else:
        batch_size = cfg.TRAIN.BATCH_SIZE
    logger.info(f"Train stream dataloader has batch_size: {batch_size}")
    drop_last = False  # Always false

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split="continual")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=None,
    )
    return loader


def construct_predictstream_loader(trainloader, cfg, subset_idxes):
    """
    Constructs data loader similar to trainloader, but using max batch_size.
    Samples that are NOT in one of the pretrain verb/noun sets are excluded.
    """
    num_workers = cfg.PREDICT_PHASE.NUM_WORKERS  # Optimized for inference
    total_mem_batch_size = cfg.PREDICT_PHASE.BATCH_SIZE
    if cfg.SOLVER.ACCELERATOR not in ["dp", "gpu"]:
        batch_size = int(total_mem_batch_size / cfg.NUM_GPUS)
    else:
        batch_size = total_mem_batch_size
    logger.info(f"Prediction dataloader has batch_size: {batch_size}, and subset idxes: {subset_idxes}")

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainloader.dataset, indices=subset_idxes),
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        num_workers=num_workers,
        pin_memory=trainloader.pin_memory,
        drop_last=trainloader.drop_last,
        collate_fn=None,
    )
    return loader

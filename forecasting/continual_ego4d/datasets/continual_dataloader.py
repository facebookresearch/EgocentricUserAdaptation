import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler

import itertools

from ego4d.datasets.build import build_dataset
from collections import defaultdict


def construct_trainstream_loader(cfg, shuffle=False):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            ego4d/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    dataset_name = cfg.TRAIN.DATASET
    if cfg.SOLVER.ACCELERATOR not in ["dp", "gpu"]:
        batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
    else:
        batch_size = cfg.TRAIN.BATCH_SIZE
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

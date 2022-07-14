from __future__ import annotations

import torch.utils.data

import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import torch.utils.data
from iopath.common.file_io import g_pathmgr

from pytorchvideo.data.clip_sampling import ClipSampler

import itertools
import os

import torch
import torch.utils.data
from pytorchvideo.data import make_clip_sampler, UniformClipSampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample,
)
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import (
    Compose,
    Lambda,
)

from ego4d.datasets.build import DATASET_REGISTRY
from ego4d.utils import logging, video_transformer
from ego4d.datasets.ptv_dataset_helper import LabeledVideoDataset, UntrimmedClipSampler

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Ego4dContinualRecognition(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        assert mode in ["continual"], \
            "Split '{}' not supported for Continual Ego4d ".format(mode)

        # Video sampling
        assert cfg.SOLVER.ACCELERATOR == "gpu", "Online per-sample processing only allows single-device training"
        video_sampler = SequentialSampler
        if cfg.SOLVER.ACCELERATOR not in ["dp", "gpu"] and cfg.NUM_GPUS > 1:
            video_sampler = DistributedSampler

        # Clip sampling
        clip_duration = (self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE) / self.cfg.DATA.TARGET_FPS
        clip_stride_seconds = self.cfg.DATA.SEQ_OBSERVED_FRAME_STRIDE / self.cfg.DATA.TARGET_FPS  # Seconds
        clip_sampler = UniformClipSampler(
            clip_duration=clip_duration,
            stride=clip_stride_seconds,
        )

        self.dataset = clip_user_recognition_dataset(
            user_id=cfg.DATA.USER_ID,
            user_annotations=cfg.DATA.USER_DS_ENTRIES,
            clip_sampler=clip_sampler,
            video_sampler=video_sampler,
            decode_audio=False,
            transform=self._make_transform(mode, cfg),
            video_path_prefix=self.cfg.DATA.PATH_PREFIX,
        )

        # Make iterable dataset
        self._dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(self.dataset), 2)
        )

    @property
    def sampler(self):
        return self.dataset.video_sampler

    def _make_transform(self, mode: str, cfg):
        return Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(cfg.DATA.NUM_FRAMES),
                            Lambda(lambda x: x / 255.0),
                            Normalize(cfg.DATA.MEAN, cfg.DATA.STD),
                        ]
                        + video_transformer.random_scale_crop_flip(mode, cfg)
                        + [video_transformer.uniform_temporal_subsample_repeated(cfg)]
                    ),
                ),
                Lambda(
                    lambda x: (
                        x["video"],
                        torch.tensor([x["verb_label"], x["noun_label"]]),
                        str(x["video_name"]) + "_" + str(x["video_index"]),
                        {},
                    )
                ),
            ]
        )

    def __getitem__(self, index):
        value = next(self._dataset_iter)
        return value

    def __len__(self):
        return self.dataset.num_videos


def get_user_to_dataset_dict(data_path):
    """
    :return: Dictionary that holds the annotation entries in a list per user. <User,Dataset-entry-list>
    """
    # Set json path for dataloader
    assert os.path.exists(data_path), f'Please run user datasplit script first to create: {data_path}'

    if g_pathmgr.isfile(data_path):
        try:
            with g_pathmgr.open(data_path, "r") as f:
                dict_holder = json.load(f)
        except Exception:
            raise FileNotFoundError(f"{data_path} must be json for Ego4D dataset")

    usersplit_annotations = dict_holder['users']

    return usersplit_annotations


def clip_user_recognition_dataset(
        user_id: str,  # Ego4d 'fb_participant_id'
        user_annotations: List,
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        video_path_prefix: str = "",
        decode_audio: bool = True,
        decoder: str = "pyav",
):
    # LabeledVideoDataset requires the data to be list of tuples with format:
    # (video_paths, annotation_dict). For recognition, the annotation_dict contains
    # the verb and noun label, and the annotation boundaries.
    untrimmed_clip_annotations = []
    for entry in user_annotations:
        untrimmed_clip_annotations.append(
            (
                os.path.join(video_path_prefix, f'{entry["clip_uid"]}.mp4'),
                {
                    "clip_start_sec": entry['action_clip_start_sec'],
                    "clip_end_sec": entry['action_clip_end_sec'],
                    "noun_label": entry['noun_label'],
                    "verb_label": entry['verb_label'],
                    "action_idx": entry['action_idx'],
                    "user_id": user_id,  # Grouped by user
                },
            )
        )

    dataset = LabeledVideoDataset(
        untrimmed_clip_annotations,
        UntrimmedClipSampler(clip_sampler),
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )
    return dataset


def construct_seq_loader(
        cfg,
        dataset_name,
        usersplit,
        batch_size,
        subset_indices: Union[List, Tuple] = None):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            ego4d/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, usersplit)  # TODO import

    # TODO make subset
    if subset_indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices=subset_indices)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=dataset.sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=False,
        collate_fn=None,
    )
    return loader



from __future__ import annotations

import copy

import torch.utils.data

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import torch.utils.data

import itertools
import pandas as pd

from fractions import Fraction
import torch.utils.data
from pytorchvideo.data import UniformClipSampler
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
from ego4d.datasets.build import build_dataset

import gc

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr

from pytorchvideo.data.clip_sampling import ClipSampler, ClipInfo
from pytorchvideo.data.video import VideoPathHandler
from pytorchvideo.transforms.functional import uniform_temporal_subsample
from pytorchvideo.data.utils import MultiProcessSampler

from ego4d.datasets.ptv_dataset_helper import UntrimmedClipSampler  # TODO MAKE SURE NOT USING THIS ONE!

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Ego4dContinualRecognition(torch.utils.data.Dataset):
    """Torch compatible Wrapper dataset with video transforms."""

    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(self, cfg, mode):
        self.cfg = cfg
        assert mode in ["continual"], \
            "Split '{}' not supported for Continual Ego4d ".format(mode)

        # Video sampling
        assert cfg.SOLVER.ACCELERATOR == "gpu", \
            f"Online per-sample processing only allows single-device training, not '{cfg.SOLVER.ACCELERATOR}'"
        video_sampler = SequentialSampler
        if cfg.SOLVER.ACCELERATOR not in ["dp", "gpu"] and cfg.NUM_GPUS > 1:
            video_sampler = DistributedSampler

        # Clip sampling
        clip_duration = Fraction((self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE), self.cfg.DATA.TARGET_FPS)
        clip_stride_seconds = Fraction(self.cfg.DATA.SEQ_OBSERVED_FRAME_STRIDE, self.cfg.DATA.TARGET_FPS)  # Seconds
        clip_stride_seconds = None

        logger.debug(f"CLIP SAMPLER: Clip duration={clip_duration}, clip_stride_seconds={clip_stride_seconds}")
        clip_sampler = EnhancedUntrimmedClipSampler(
            UniformClipSampler(
                clip_duration=clip_duration,
                stride=clip_stride_seconds,
            )
        )

        self.miniclip_sampler = SequentialSampler  # Visit all miniclips (with annotation) sequentially

        self.seq_input_list = get_seq_annotated_clip_input_list(
            user_id=cfg.DATA.USER_ID,
            user_annotations=cfg.DATA.USER_DS_ENTRIES,
            clip_sampler=clip_sampler,
            video_sampler=video_sampler,
            video_path_prefix=self.cfg.DATA.PATH_PREFIX,
        )

    @property
    def sampler(self):
        return self.miniclip_sampler

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
        """
        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            try:
                video_path, miniclip_info_dict = self.seq_input_list[index]
                video = self.path_handler.video_from_path(
                    video_path,
                    decode_audio=self._decode_audio,
                    decoder=self._decoder,
                )
                # logger.debug(f'video={os.path.basename(video_path)}, info={info_dict}')
            except Exception as e:
                logger.debug(
                    "Failed to load video with error: {}; trial {}".format(e, i_try)
                )
                continue

            clip_start = miniclip_info_dict['clip_start_sec']
            clip_end = miniclip_info_dict['clip_end_sec']
            clip_index = miniclip_info_dict['clip_index']
            aug_index = miniclip_info_dict['aug_index']

            # DECODE
            decoded_clip = video.get_clip(clip_start, clip_end)
            video_is_null = decoded_clip is None or decoded_clip["video"] is None

            if video_is_null:
                # Close the loaded encoded video and reset the last sampled clip time ready
                # to sample a new video on the next iteration.
                video.close()

                # Force garbage collection to release video container immediately
                # otherwise memory can spike.
                gc.collect()

                logger.debug("Failed to load clip {}; trial {}".format(video.name, i_try))
                continue

            frames = decoded_clip[0]["video"]
            audio_samples = decoded_clip[0]["audio"]
            logger.info(f"decoded miniclip: {miniclip_info_dict}")

            sample_dict = {
                **miniclip_info_dict,
                **({"audio": audio_samples} if audio_samples is not None else {}),
                "video": frames,
                "video_name": video.name,
                "video_index": self._last_video_index,
                "clip_index": clip_index,
                "aug_index": aug_index,
            }
            if self._transform is not None:
                sample_dict = self._transform(sample_dict)

            return sample_dict
        else:
            raise RuntimeError(
                f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )

    def __len__(self):
        return len(self.seq_input_list)


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
    logger.debug(f"Got usersplit with users: {list(usersplit_annotations.keys())}")
    return usersplit_annotations


def get_seq_annotated_clip_input_list(
        user_id: str,  # Ego4d 'fb_participant_id'
        user_annotations: List,
        clip_sampler: Union[ClipSampler, EnhancedUntrimmedClipSampler],
        video_sampler: Type[torch.utils.data.Sampler],
        video_path_prefix: str = "",
        decoder: str = "pyav",
):
    # Sort clips to match sequence
    # Per video (>>5miin): origin_video_id (user collected by same collector-instance, with possibly time-like video id)
    # Per 5min clip in video: 'clip_parent_start_sec'
    # Per action annotation in 5min clip: action_idx
    user_df = pd.DataFrame(user_annotations)
    seq_user_df = user_df.sort_values(
        ['origin_video_id', 'clip_parent_start_sec', 'action_idx'],
        ascending=[True, True, True]
    )
    seq_user_df = seq_user_df.reset_index(drop=True)

    # Collect sorted annotation entries. And applies a policy for overlapping annotation entries in the same clip.
    # e.g. FIFO uses the end-time of the earliest action, as a new start time for subsequent overlapping actions.
    untrimmed_video_annotation_list = get_seq_annotations_by_policy(
        seq_user_df, clip_sampler, video_path_prefix, user_id,
        policy='fifo'
    )

    # Convert the annotation-level entries to actual visited 2-s input clips
    # This allows to make a Dataset instead of an IterableDataset, which is required for efficient slicing
    # For this, we will apply the ClipSampler and VideoSampler in pre-processing.
    # The result will give an index-able list of input-level entries with according annotation.
    untrimmed_miniclip_annotation_list = get_preprocessed_clips(
        untrimmed_video_annotation_list, video_sampler, clip_sampler, decoder)

    # LabeledVideoDataset requires the data to be list of tuples with format:
    # (video_paths, annotation_dict). For recognition, the annotation_dict contains
    # the verb and noun label, and the annotation boundaries.
    return untrimmed_miniclip_annotation_list


def get_seq_annotations_by_policy(seq_user_df, clip_sampler, video_path_prefix, user_id, policy='fifo'):
    """Take the user data frame, and return the annotations as a list of (video_path, annotation_info_dict) entries."""
    if policy == 'fifo':
        return annotation_fifo_policy(seq_user_df, clip_sampler, video_path_prefix, user_id)
    else:
        raise NotImplementedError(f"Policy '{policy}' is unkown")


def annotation_fifo_policy(seq_user_df, clip_sampler, video_path_prefix, user_id):
    # Keep final time, if start_time of new fragment < end_time, readjust start_time.
    # If the new_time >= end_time, just cut the fragment.
    # If subclip is < min_clip_len, disregard
    untrimmed_clip_annotations = []
    last_clip_uid = None
    last_clip_action_time = 0
    num_skipped_entries = 0
    num_trimmed_entries = 0
    for entry_idx in range(seq_user_df.shape[0]):
        row = seq_user_df.iloc[entry_idx]

        # Action in new clip: reset
        if row['clip_uid'] != last_clip_uid:
            last_clip_uid = row['clip_uid']
            last_clip_action_time = 0

        # Clip info
        clip_start_sec = row['action_clip_start_sec']
        clip_end_sec = row['action_clip_end_sec']
        clip_old_len_sec = clip_end_sec - clip_start_sec
        clip_new_len_sec = clip_end_sec - last_clip_action_time

        # No problem, take full clip
        if clip_start_sec >= last_clip_action_time:
            last_clip_action_time = clip_end_sec
            ret_entry = get_formatted_entry(row, video_path_prefix, user_id)

        # The current one is entirely behind current timestep, or is too short to be sampled
        elif clip_end_sec <= last_clip_action_time \
                or clip_new_len_sec < clip_sampler._clip_duration:
            num_skipped_entries += 1
            continue

        else:  # Trim current one
            num_trimmed_entries += 1
            ret_entry = get_formatted_entry(row, video_path_prefix, user_id)
            ret_entry[1]['clip_start_sec'] = last_clip_action_time  # Overwrite the start time
            last_clip_action_time = clip_end_sec

        # Add entry
        untrimmed_clip_annotations.append(ret_entry)

    logger.info(f"In FIFO single-action-policy for the dataset, "
                f"entries DROPPED={num_skipped_entries}, TRIMMED={num_trimmed_entries}")

    return untrimmed_clip_annotations


def get_preprocessed_clips(video_annotation_list, video_sampler, clip_sampler: ClipSampler, decoder: str = "pyav"):
    """

    Videos are only decoded when using video.get_clip(). Loading the videos without decoding calls,
    we can access just the meta-data, such as duration.

    :param video_annotation_list: list of annotation-level entries, full annotation range for the video.
    The annotation info dicts are formatted entries, see function 'get_formatted_entry'.
    Video paths are not unique as multiple annotations may refer to the same video, e.g.:
            [(video_path, annotation1_info_dict),
            (video_path, annotation2_info_dict),...]
    :param video_sampler: sample over the video_annotation_list
    :param clip_sampler: sample clips within 1 entry
    :return: List of miniclip level clips, e.g.:
                [(video_path, annotation1_sampledclip1_info_dict),
                (video_path, annotation1_sampledclip2_info_dict),
                ...,
                (video_path, annotation2_sampledclip1_info_dict),
                (video_path, annotation2_sampledclip2_info_dict),
                ... ]
    """

    miniclip_annotation_list = []
    next_clip_start_time = 0.0
    path_handler = VideoPathHandler()

    annotation_sampler = video_sampler(video_annotation_list)
    assert isinstance(annotation_sampler, SequentialSampler), \
        f"Preprocessing only supports a sequential (deterministic) annotation sampler, not {annotation_sampler}"
    annotation_index = next(annotation_sampler)

    # Iterate all annotations
    while annotation_index < len(video_annotation_list):

        video_path, info_dict = video_annotation_list[annotation_index]
        assert os.path.isfile(video_path), f"Non-existing video: {video_path}"
        logger.debug(f'video={os.path.basename(video_path)}, info={info_dict}')

        # Meta-data holder object (not decoded yet)
        video = path_handler.video_from_path(
            video_path,
            decode_audio=False,
            decoder=decoder,
        )
        clip: ClipInfo = clip_sampler(
            next_clip_start_time, video.duration, info_dict
        )
        assert not isinstance(clip, list), f"No multi-clip sampling supported"

        # Untrimmed refers to the 5-min clip, with the clip start-time possibly != 0
        untrimmed_clip_start_time = clip.clip_start_sec
        untrimmed_clip_end_time = clip.clip_end_sec

        # BE CAREFUL, this will be the UNTRIMMED TIME. WHEN PASSING TO THE SAMPLER ON NEXT ITERATION, SHOULD
        # SUBTRACT THE START TIME!
        next_clip_start_time = untrimmed_clip_end_time

        # Add annotation entry with adapted times
        new_cliplevel_entry = copy.deepcopy(video_annotation_list[annotation_index])
        new_cliplevel_entry[1]['clip_start_sec'] = untrimmed_clip_start_time
        new_cliplevel_entry[1]['clip_end_sec'] = untrimmed_clip_end_time
        new_cliplevel_entry[1]['aug_index'] = clip.aug_index
        new_cliplevel_entry[1]['clip_index'] = clip.clip_index
        miniclip_annotation_list.append(new_cliplevel_entry)

        if clip.is_last_clip:  # This annotation ran out, sample next
            next_clip_start_time = 0.0
            annotation_index = next(annotation_sampler)
            logger.debug(f"NEW ANNOTATION INDEX: {annotation_index}")

    return miniclip_annotation_list


def get_formatted_entry(entry, video_path_prefix, user_id):
    """Return (video_path, annotation_info) format."""
    return (
        os.path.join(video_path_prefix, f'{entry["clip_uid"]}.mp4'),
        {
            "clip_start_sec": entry['action_clip_start_sec'],
            "clip_end_sec": entry['action_clip_end_sec'],
            "noun_label": entry['noun_label'],
            "verb_label": entry['verb_label'],
            "action_idx": entry['action_idx'],
            "parent_video_scenarios": entry['parent_video_scenarios'],
            "user_id": user_id,  # Grouped by user
        }
    )


class EnhancedUntrimmedClipSampler:
    """
    A wrapper for adapting untrimmed annotated clips from the json_dataset to the
    standard `pytorchvideo.data.ClipSampler` expected format. Specifically, for each
    clip it uses the provided `clip_sampler` to sample between "clip_start_sec" and
    "clip_end_sec" from the json_dataset clip annotation.
    """

    def __init__(self, clip_sampler: ClipSampler) -> None:
        """
        Args:
            clip_sampler (`pytorchvideo.data.ClipSampler`): Strategy used for sampling
                between the untrimmed clip boundary.
        """
        self._trimmed_clip_sampler = clip_sampler

    def __call__(
            self, untrim_last_clip_time: float, video_duration: float, clip_info: Dict[str, Any]
    ) -> ClipInfo:
        clip_start_boundary = clip_info["clip_start_sec"]
        clip_end_boundary = clip_info["clip_end_sec"]
        duration = clip_end_boundary - clip_start_boundary

        # Important to avoid out-of-bounds when sampling multiple times (when untrim_last_clip_time>0)
        trim_last_clip_time = untrim_last_clip_time - clip_start_boundary

        # Sample between 0 and duration of untrimmed clip, then add back start boundary.
        clip_info = self._trimmed_clip_sampler(trim_last_clip_time, duration, clip_info)
        return ClipInfo(
            clip_info.clip_start_sec + clip_start_boundary,
            clip_info.clip_end_sec + clip_start_boundary,
            clip_info.clip_index,
            clip_info.aug_index,
            clip_info.is_last_clip,
        )

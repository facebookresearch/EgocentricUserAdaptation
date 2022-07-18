from __future__ import annotations

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
        return self.dataset.video_sampler  # TODO get a sequential sampler of the mini-clip level entries

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

    def __getitem__(self, index):  # TODO Take actual index
        value = next(self._dataset_iter)
        return value

    def __len__(self):
        return self.dataset.num_videos  # TODO make per-clip samples!


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


def clip_user_recognition_dataset(
        user_id: str,  # Ego4d 'fb_participant_id'
        user_annotations: List,
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler],
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        video_path_prefix: str = "",
        decode_audio: bool = True,
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

    # LabeledVideoDataset requires the data to be list of tuples with format:
    # (video_paths, annotation_dict). For recognition, the annotation_dict contains
    # the verb and noun label, and the annotation boundaries.
    untrimmed_clip_annotations = get_seq_annotations_by_policy(
        seq_user_df, clip_sampler, video_path_prefix, user_id, policy='fifo')

    dataset = ContinualLabeledVideoDataset(
        untrimmed_clip_annotations,
        # clip_sampler,
        EnhancedUntrimmedClipSampler(clip_sampler),
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )
    return dataset


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


class ContinualLabeledVideoDataset(torch.utils.data.IterableDataset):
    """
    Continual part makes sure each video is observed until finished all clips.

    LabeledVideoDataset handles the storage, loading, decoding and clip sampling for a
    video dataset. It assumes each video is stored as either an encoded video
    (e.g. mp4, avi) or a frame video (e.g. a folder of jpg, or png)
    """

    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
            self,
            labeled_video_paths: List[Tuple[str, Optional[dict]]],
            clip_sampler: ClipSampler,
            video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
            transform: Optional[Callable[[dict], Any]] = None,
            decode_audio: bool = True,
            decoder: str = "pyav",
    ) -> None:
        """
        Args:
            labeled_video_paths (List[Tuple[str, Optional[dict]]]): List containing
                    video file paths and associated labels. If video paths are a folder
                    it's interpreted as a frame video, otherwise it must be an encoded
                    video.

            clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

            transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations on the clips. The clip output format is described in __next__().

            decode_audio (bool): If True, also decode audio from video.

            decoder (str): Defines what type of decoder used to decode a video. Not used for
                frame videos.
        """
        self._decode_audio = decode_audio
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._labeled_videos = labeled_video_paths
        self._decoder = decoder
        self.path_handler = VideoPathHandler()

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clips = None
        self._next_clip_start_time = 0.0

        # CONTINUAL LEARNING STATES
        self._last_video_index = None

    @property
    def video_sampler(self):
        """
        Returns:
            The video sampler that defines video sample order. Note that you'll need to
            use this property to set the epoch for a torch.utils.data.DistributedSampler.
        """
        return self._video_sampler

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self.video_sampler)

    def num_clips(self):
        """
        Returns:
            Number of videos in dataset.
        """

        # TODO for each video try out
        self._last_video_index = next(self._video_sampler_iter)

        try:
            video_path, info_dict = self._labeled_videos[self._last_video_index]
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

        clips = self._clip_sampler(
            self._next_clip_start_time, video.duration, info_dict
        )

        if not isinstance(clips, list):
            clips = [clips]

        decoded_clips = []
        video_is_null = False
        for clip_start, clip_end, clip_index, aug_index, is_last_clip in clips:
            clip = video.get_clip(clip_start, clip_end)
            video_is_null = clip is None or clip["video"] is None
            if video_is_null:
                break
            decoded_clips.append(clip)

        # Next is entirely new observed batch of input samples
        self._next_clip_start_time = clip_end
        # BE CAREFUL, this will be the UNTRIMMED TIME. WHEN PASSING TO THE SAMPLER ON NEXT ITERATION, SHOULD
        # SUBTRACT THE START TIME!

        logger.debug(
            f"2s-CLIP: video={os.path.basename(video_path)},video_idx={self._last_video_index}, nb_clips={len(clips)}, is_last_clip={is_last_clip}, clip_start={clip_start},clip_end={clip_end}, video_len={video.duration} "
            f"  self._next_clip_start_time={self._next_clip_start_time}")  # Always 1 clip exactly
        logger.debug(
            f"5min-CLIP: video={os.path.basename(video_path)}, video_idx={self._last_video_index},clip_info_dict={info_dict}")

        if is_last_clip or video_is_null:
            self._next_clip_start_time = 0.0
            if video_is_null:
                continue

        if is_last_clip:  # This video ran out, sample next
            self._last_video_index = next(self._video_sampler_iter)
            logger.debug(f"NEW VIDEO INDEX: {self._last_video_index}")

        return len(self.video_sampler)

    def __next__(self) -> dict:
        """
        Keep sampling the

        Retrieves the next clip based on the clip sampling strategy and video sampler.

        video_sampler: The next video-level entry may be the same video, but with other annotation entry (other action).
        The boundaries within the same video hence may progress, or it may be another video_path instead.


        clip_sampler: within the action_boundaries in a video, sample a subvideo (a clip).
        The single annotation video bounary should be processed entirely in shifting window style.
        One iteration only considers a single clip of these boundaries. The next timestep the start time is shifted.

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
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        if self._last_video_index is None:  # Load first video
            self._last_video_index = next(self._video_sampler_iter)

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            try:
                video_path, info_dict = self._labeled_videos[self._last_video_index]
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

            clips = self._clip_sampler(
                self._next_clip_start_time, video.duration, info_dict
            )

            if not isinstance(clips, list):
                clips = [clips]

            decoded_clips = []
            video_is_null = False
            for clip_start, clip_end, clip_index, aug_index, is_last_clip in clips:
                clip = video.get_clip(clip_start, clip_end)
                video_is_null = clip is None or clip["video"] is None
                if video_is_null:
                    break
                decoded_clips.append(clip)

            # Next is entirely new observed batch of input samples
            self._next_clip_start_time = clip_end
            # BE CAREFUL, this will be the UNTRIMMED TIME. WHEN PASSING TO THE SAMPLER ON NEXT ITERATION, SHOULD
            # SUBTRACT THE START TIME!

            logger.debug(
                f"2s-CLIP: video={os.path.basename(video_path)},video_idx={self._last_video_index}, nb_clips={len(clips)}, is_last_clip={is_last_clip}, clip_start={clip_start},clip_end={clip_end}, video_len={video.duration} "
                f"  self._next_clip_start_time={self._next_clip_start_time}")  # Always 1 clip exactly
            logger.debug(
                f"5min-CLIP: video={os.path.basename(video_path)}, video_idx={self._last_video_index},clip_info_dict={info_dict}")

            if is_last_clip or video_is_null:
                # Close the loaded encoded video and reset the last sampled clip time ready
                # to sample a new video on the next iteration.
                video.close()
                self._next_clip_start_time = 0.0

                # Force garbage collection to release video container immediately
                # otherwise memory can spike.
                gc.collect()

                if video_is_null:
                    logger.debug(
                        "Failed to load clip {}; trial {}".format(video.name, i_try)
                    )
                    continue

            if len(decoded_clips) == 1:
                frames = decoded_clips[0]["video"]
                audio_samples = decoded_clips[0]["audio"]
            else:
                clip_frames = [
                    uniform_temporal_subsample(x["video"], num_samples=64)
                    for x in decoded_clips
                ]
                frames = torch.stack(clip_frames, dim=0)

                clip_audio = [x["audio"] for x in decoded_clips]
                audio_samples = None
                if None not in clip_audio:
                    audio_samples = torch.stack(clip_audio, dim=0)

            sample_dict = {
                "video": frames,
                "video_name": video.name,
                "video_index": self._last_video_index,
                "clip_index": clip_index,
                "aug_index": aug_index,
                **info_dict,
                **({"audio": audio_samples} if audio_samples is not None else {}),
            }
            if self._transform is not None:
                sample_dict = self._transform(sample_dict)

            if is_last_clip:  # This video ran out, sample next
                self._last_video_index = next(self._video_sampler_iter)
                logger.debug(f"NEW VIDEO INDEX: {self._last_video_index}")

            return sample_dict
        else:
            raise RuntimeError(
                f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self


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

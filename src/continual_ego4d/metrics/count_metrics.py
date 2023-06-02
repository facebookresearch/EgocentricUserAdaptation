# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from typing import Dict, Union
from continual_ego4d.metrics.metric import Metric, get_metric_tag, TAG_BATCH
from collections import Counter
import numpy as np
from continual_ego4d.datasets.continual_action_recog_dataset import verbnoun_to_action, verbnoun_format
from continual_ego4d.metrics.meters import AverageMeter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from continual_ego4d.tasks.continual_action_recog_task import StreamStateTracker


class HistoryCountMetric(Metric):
    """
    For actions/verbs/nouns in current batch, returns the avg count of these actions/verbs/nouns in the observed
    (history) part of the stream.
    May optionally also include the pretrain count.
    """
    reset_before_batch = True
    action_modes = ["verb", "noun", "action"]
    count_modes = ["history+pretrain", "history", "pretrain"]

    def __init__(
            self,
            history_action_instance_count: dict = None,
            pretrain_action_instance_count: dict = None,
            action_mode="action",
    ):
        self.history_action_instance_count = history_action_instance_count
        self.pretrain_action_instance_count = pretrain_action_instance_count

        if self.history_action_instance_count is not None \
                and self.pretrain_action_instance_count is not None:
            self.count_mode = 'history+pretrain'
            self.counter_dicts = [self.history_action_instance_count, self.pretrain_action_instance_count]

        elif self.history_action_instance_count is not None:
            self.count_mode = 'history'
            self.counter_dicts = [self.history_action_instance_count]

        elif self.pretrain_action_instance_count is not None:
            self.count_mode = 'pretrain'
            self.counter_dicts = [self.pretrain_action_instance_count]

        else:
            raise ValueError("At least one counter should be defined.")
        assert self.count_mode in self.count_modes

        self.action_mode = action_mode  # Action/verb/noun
        assert self.action_mode in self.action_modes
        if self.action_mode == 'verb':
            self.label_idx = 0
        elif self.action_mode == 'noun':
            self.label_idx = 1
        elif self.action_mode == 'action':
            self.label_idx = None  # Not applicable
        else:
            raise NotImplementedError()

        base_name = f"batch_{self.count_mode}_count"
        self.name = get_metric_tag(TAG_BATCH, action_mode=self.action_mode, base_metric_name=base_name)

        # Keep all results
        self.avg_meter = AverageMeter()
        self.iter_to_result = {}

    @torch.no_grad()
    def update(self, preds, labels, stream_sample_idxs, **kwargs):
        """Update metric from predictions and labels."""

        # Verb/noun errors
        if self.action_mode in ['verb', 'noun']:
            label_list = [verbnoun_format(x) for x in labels[:, self.label_idx].tolist()]

        elif self.action_mode in ['action']:
            label_batch_axis = 0
            label_list = []
            for verbnoun_t in torch.unbind(labels, dim=label_batch_axis):
                label_list.append(verbnoun_to_action(*verbnoun_t.tolist()))

        else:
            raise ValueError()

        # Avg over labels how many times counted in counter_dicts
        for label in label_list:
            label_history_count = 0

            for counter_dict in self.counter_dicts:
                if label in counter_dict:
                    label_history_count += counter_dict[label]

            self.avg_meter.update(label_history_count, weight=1)

    @torch.no_grad()
    def reset(self):
        """Reset the metric."""
        self.avg_meter.reset()

    @torch.no_grad()
    def result(self, current_batch_idx: int, **kwargs) -> Dict:
        """Get the metric(s) with name in dict format."""
        result = self.avg_meter.avg  # Avg over counts
        self.iter_to_result[current_batch_idx] = result
        return {self.name: result}

    @torch.no_grad()
    def dump(self) -> Dict:
        """Optional: after training stream, a dump of states could be returned."""
        return {f"{self.name}_PER_BATCH": self.iter_to_result}


class SetCountMetric(Metric):
    """Count how many elements in a set. When a reference set is defined, counts also intersection, and
    subtracted leftover parts of the two sets.
    """
    reset_before_batch = False

    modes = ["verb", "noun", "action"]

    def __init__(
            self,
            observed_set_name: str,
            observed_set: Union[set, dict],
            ref_set_name: str = None,
            ref_set: Union[set, dict] = None,
            mode="action"
    ):
        self.observed_set_name = observed_set_name
        self.observed_set = observed_set

        self.ref_set = ref_set
        if self.ref_set is not None:
            assert ref_set_name is not None
            self.ref_set_name = ref_set_name

        self.mode = mode  # Action/verb/noun
        assert self.mode in self.modes

    @torch.no_grad()
    def update(self, preds, labels, stream_sample_idxs, **kwargs):
        """Update metric from predictions and labels."""
        pass

    @torch.no_grad()
    def reset(self):
        """Reset the metric."""
        pass

    @torch.no_grad()
    def result(self, current_batch_idx: int, **kwargs) -> Dict:
        """Get the metric(s) with name in dict format."""
        ret = []

        # Count in observed set
        ret.append(
            (get_metric_tag(TAG_BATCH, action_mode=self.mode, base_metric_name=f"count_{self.observed_set_name}"),
             len(self.observed_set))
        )

        if self.ref_set is not None:
            intersect_set = {x for x in self.observed_set if x in self.ref_set}

            # Count in reference set
            ret.append(
                (get_metric_tag(TAG_BATCH, action_mode=self.mode, base_metric_name=f"count_{self.ref_set_name}"),
                 len(self.ref_set))
            )

            # Count of intersection
            ret.append(
                (get_metric_tag(TAG_BATCH, action_mode=self.mode,
                                base_metric_name=f"count_intersect_{self.observed_set_name}_VS_{self.ref_set_name}"),
                 len(intersect_set))
            )

            # Fraction intersection/observed set
            if len(self.observed_set) > 0:
                ret.append((
                    get_metric_tag(
                        TAG_BATCH, action_mode=self.mode, base_metric_name=
                        f"count_intersect_{self.observed_set_name}_VS_{self.ref_set_name}_{self.observed_set_name}-fract"),
                    len(intersect_set) / len(self.observed_set)
                ))

            # Fraction intersection/ref set
            if len(self.ref_set) > 0:
                ret.append((
                    get_metric_tag(
                        TAG_BATCH, action_mode=self.mode, base_metric_name=
                        f"count_intersect_{self.observed_set_name}_VS_{self.ref_set_name}_{self.ref_set_name}-fract"),
                    len(intersect_set) / len(self.ref_set)
                ))

        return {k: v for k, v in ret}

    @torch.no_grad()
    def dump(self) -> Dict:
        """Optional: after training stream, a dump of states could be returned."""
        return {}


class WindowedUniqueCountMetric(Metric):
    """ For a window preceding the current timestep, how many actions/verbs/nouns are unique. """
    reset_before_batch = True
    action_modes = ["verb", "noun", "action"]

    def __init__(
            self,
            preceding_window_size: int,
            sample_idx_to_action_list: list,
            action_mode="action"
    ):
        assert preceding_window_size > 0
        self.preceding_window_size = preceding_window_size
        self.sample_idx_to_action_list = sample_idx_to_action_list
        self.name_prefix = f"count_window_{self.preceding_window_size}"

        self.action_mode = action_mode  # Action/verb/noun
        assert self.action_mode in self.action_modes

        if self.action_mode == 'action':
            self.label_map_fn = lambda x: x
        elif self.action_mode == 'verb':
            self.label_map_fn = lambda x: x[0]
        elif self.action_mode == 'noun':
            self.label_map_fn = lambda x: x[1]
        else:
            raise ValueError()

        # State vars
        self._current_batch_min_idx: int = None

    @torch.no_grad()
    def update(self, preds, labels, stream_sample_idxs, stream_state: 'StreamStateTracker' = None, **kwargs):
        """Update metric from predictions and labels."""
        current_batch_min_idx = min(stream_state.stream_batch_sample_idxs)
        if self._current_batch_min_idx is None:
            self._current_batch_min_idx = current_batch_min_idx
        else:
            self._current_batch_min_idx = min(self._current_batch_min_idx, current_batch_min_idx)

    @torch.no_grad()
    def reset(self):
        """Reset the metric."""
        self._current_batch_min_idx = None

    @torch.no_grad()
    def result(self, current_batch_idx: int, **kwargs) -> Dict:
        """Get the metric(s) with name in dict format."""
        assert self._current_batch_min_idx is not None, "Assign self._current_batch_min_idx first with update()"

        label_window_subset = self.sample_idx_to_action_list[
                              max(0, self._current_batch_min_idx - self.preceding_window_size):
                              self._current_batch_min_idx]
        window_size = len(label_window_subset)
        if window_size == 0:
            return {}

        # Map to verb/noun/action
        label_window_subset_m = list(map(self.label_map_fn, label_window_subset))

        # Count per label
        window_label_counter = Counter(label_window_subset_m)

        # Uniques
        nb_uniques = len(window_label_counter)  # nb of action keys

        # Histogram stats
        window_action_counts = list(window_label_counter.values())
        histogram_mean = np.mean(window_action_counts)
        histogram_std = np.std(window_action_counts)
        histogram_max = max(window_action_counts)
        histogram_min = min(window_action_counts)

        return {
            # Unique counts
            get_metric_tag(
                TAG_BATCH, action_mode=self.action_mode,
                base_metric_name=f"{self.name_prefix}_unique"): nb_uniques,
            get_metric_tag(
                TAG_BATCH, action_mode=self.action_mode,
                base_metric_name=f"{self.name_prefix}_unique_frac"): nb_uniques / window_size,

            # Window stats
            get_metric_tag(
                TAG_BATCH, action_mode=self.action_mode,
                base_metric_name=f"{self.name_prefix}_freq_mean"): histogram_mean,
            get_metric_tag(
                TAG_BATCH, action_mode=self.action_mode,
                base_metric_name=f"{self.name_prefix}_freq_std"): histogram_std,
            get_metric_tag(
                TAG_BATCH, action_mode=self.action_mode,
                base_metric_name=f"{self.name_prefix}_freq_min"): histogram_min,
            get_metric_tag(
                TAG_BATCH, action_mode=self.action_mode,
                base_metric_name=f"{self.name_prefix}_freq_max"): histogram_max,
        }

    @torch.no_grad()
    def dump(self) -> Dict:
        """Optional: after training stream, a dump of states could be returned."""
        return {}

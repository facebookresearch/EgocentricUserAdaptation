import torch
from typing import Dict, Set, Union, Tuple
from continual_ego4d.metrics.metric import AvgMeterMetric, Metric, get_metric_tag, TAG_BATCH
from collections import Counter
import numpy as np


class SetCountMetric(Metric):
    reset_before_batch = False

    modes = ["verb", "noun", "action"]

    def __init__(
            self,
            observed_set_name: str,
            observed_set: set,
            ref_set_name: str = None,
            ref_set: set = None,
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
    def update(self, current_batch_idx: int, preds, labels, *args, **kwargs):
        """Update metric from predictions and labels."""
        pass

    @torch.no_grad()
    def reset(self):
        """Reset the metric."""
        pass

    @torch.no_grad()
    def result(self, current_batch_idx: int, *args, **kwargs) -> Dict:
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
    def update(self, current_batch_idx: int, preds, labels, current_batch_sample_idxs: list):
        """Update metric from predictions and labels."""
        current_batch_min_idx = min(current_batch_sample_idxs)
        if self._current_batch_min_idx is None:
            self._current_batch_min_idx = current_batch_min_idx
        else:
            self._current_batch_min_idx = min(self._current_batch_min_idx, current_batch_min_idx)

    @torch.no_grad()
    def reset(self):
        """Reset the metric."""
        self._current_batch_min_idx = None

    @torch.no_grad()
    def result(self, current_batch_idx: int, *args, **kwargs) -> Dict:
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

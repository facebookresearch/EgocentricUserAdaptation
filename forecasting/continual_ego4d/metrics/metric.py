import logging
from abc import ABC, abstractmethod
import torch
from typing import Dict, Set, Union, Tuple
from continual_ego4d.utils.meters import AverageMeter
from ego4d.evaluation import lta_metrics as metrics
from collections import defaultdict

TAG_BATCH = 'batch'
TAG_FUTURE = 'future'
TAG_PAST = 'past'

ACTION_MODES = ('action', 'verb', 'noun')
TRAIN_MODES = ('train', 'pred')


def get_metric_tag(main_parent_tag, train_mode='train', action_mode=None, base_metric_name=None):
    """Get logging metric name."""

    # PARENT
    assert train_mode in TRAIN_MODES
    parent_tag = [train_mode]

    if action_mode is not None:
        assert action_mode in ACTION_MODES
        parent_tag.append(action_mode)

    parent_tag.append(main_parent_tag)

    # CHILD
    assert base_metric_name is not None
    child_tag = [base_metric_name]

    return f"{'_'.join(parent_tag)}/{'_'.join(child_tag)}"


class Metric(ABC):

    @property
    @abstractmethod
    def reset_before_batch(self):
        """Reset before each batch (True), or keep accumulating (False)."""

    @abstractmethod
    @torch.no_grad()
    def update(self, current_batch_idx: int, preds: list, labels, *args, **kwargs):
        """Update metric from predictions and labels.
        preds: (2 x batch_size x input_shape) -> first dim = verb/noun
        labels: (batch_size x 2) -> second dim = verb/noun
        """

    @abstractmethod
    @torch.no_grad()
    def reset(self):
        """Reset the metric."""

    @abstractmethod
    @torch.no_grad()
    def result(self, current_batch_idx: int, *args, **kwargs) -> Dict:
        """Get the metric(s) with name in dict format."""

    @torch.no_grad()
    def save_result_to_history(self, current_batch_idx: int, *args, **kwargs):
        """Optional: after getting the result, we may also want to re-use this result later on."""
        pass

    @torch.no_grad()
    def plot(self) -> Dict:
        """Optional: during training stream, plot state of metric."""
        return {}

    @abstractmethod
    @torch.no_grad()
    def dump(self) -> Dict:
        """Optional: after training stream, a dump of states could be returned."""


class AvgMeterMetric(Metric):
    reset_before_batch = True
    modes = ACTION_MODES

    def __init__(self, mode="action"):
        self.name = None  # Overwrite
        self.avg_meter = AverageMeter()

        self.mode = mode
        assert self.mode in self.modes
        if self.mode == 'verb':
            self.label_idx = 0
        elif self.mode == 'noun':
            self.label_idx = 1
        elif self.mode == 'action':
            self.label_idx = None  # Not applicable
        else:
            raise NotImplementedError()

        # Keep all results
        self.iter_to_result = {}

    @torch.no_grad()
    def update(self, current_batch_idx: int, preds, labels, *args, **kwargs):
        """Update metric from predictions and labels."""
        raise NotImplementedError()

    @torch.no_grad()
    def reset(self):
        """Reset the metric."""
        self.avg_meter.reset()

    @torch.no_grad()
    def result(self, current_batch_idx: int, *args, meter_attr='avg', **kwargs) -> Dict:
        """Get the metric(s) with name in dict format.
        :param: meter_attr: can be set to avg or sum. sum is useful for cumulative metrics.
        """
        if self.avg_meter.count == 0:
            return {}
        assert self.name is not None, "Define name for metric"

        result = getattr(self.avg_meter, meter_attr)
        self.iter_to_result[current_batch_idx] = result
        return {self.name: result}

    @torch.no_grad()
    def dump(self) -> Dict:
        """Optional: after training stream, a dump of states could be returned."""
        return {f"{self.name}_BATCH": self.iter_to_result}

# class AvgMeterDictMetric(Metric):
#     reset_before_batch = True
#     modes = ACTION_MODES
#
#     def __init__(self, mode="action"):
#         self.name = None  # Overwrite
#         self.avg_meter_dict = defaultdict(AverageMeter)
#
#         self.mode = mode
#         assert self.mode in self.modes
#         if self.mode == 'verb':
#             self.label_idx = 0
#         elif self.mode == 'noun':
#             self.label_idx = 1
#         elif self.mode == 'action':
#             self.label_idx = None  # Not applicable
#         else:
#             raise NotImplementedError()
#
#     @torch.no_grad()
#     def update(self, current_batch_idx: int, preds, labels, *args, **kwargs):
#         """Update metric from predictions and labels."""
#         raise NotImplementedError()
#
#     @torch.no_grad()
#     def reset(self):
#         """Reset the metric."""
#         for meter in self.avg_meter_dict.values():
#             meter.reset()
#
#     @torch.no_grad()
#     def result(self, current_batch_idx: int, *args, **kwargs) -> Dict:
#         """Get the metric(s) with name in dict format."""
#         if len(self.avg_meter_dict) == 0:
#             return {}
#         if sum(meter.count for meter in self.avg_meter_dict.values()) == 0:
#             return {}
#         assert self.name is not None, "Define name for metric"
#
#         balanced_avg = torch.mean(
#             torch.tensor([meter.avg for meter in self.avg_meter_dict.values()])
#         ).item()  # List of all avgs, equally weighed
#         return {self.name: balanced_avg}
#
#     @torch.no_grad()
#     def dump(self) -> Dict:
#         """Optional: after training stream, a dump of states could be returned."""
#         raise NotImplementedError()

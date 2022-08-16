import logging
from abc import ABC, abstractmethod
import torch
from typing import Dict, Set, Union, Tuple
from continual_ego4d.utils.meters import AverageMeter
from ego4d.evaluation import lta_metrics as metrics

ACTION_MODES = ('action', 'verb', 'noun')
TRAIN_MODES = ('train', 'pred')


def get_metric_tag(parent_tag, train_mode='train', action_mode=None, base_metric_name=None):
    """Get logging metric name."""
    assert train_mode in TRAIN_MODES
    child_tag = [train_mode]

    if action_mode is not None:
        assert action_mode in ACTION_MODES
        child_tag.append(action_mode)

    if base_metric_name is not None:
        child_tag.append(base_metric_name)

    return f"{parent_tag}/{'_'.join(child_tag)}"


class Metric(ABC):

    @property
    @abstractmethod
    def reset_before_batch(self):
        """Reset before each batch (True), or keep accumulating (False)."""

    @abstractmethod
    @torch.no_grad()
    def update(self, preds: list, labels, *args, **kwargs):
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
    def result(self) -> Dict:
        """Get the metric(s) with name in dict format."""

    @torch.no_grad()
    def save_result_to_history(self):
        """Optional: after getting the result, we may also want to re-use this result later on."""
        pass


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

    @torch.no_grad()
    def update(self, preds, labels, *args, **kwargs):
        """Update metric from predictions and labels."""
        raise NotImplementedError()

    @torch.no_grad()
    def reset(self):
        """Reset the metric."""
        self.avg_meter.reset()

    @torch.no_grad()
    def result(self) -> Dict:
        """Get the metric(s) with name in dict format."""
        if self.avg_meter.count == 0:
            return {}
        assert self.name is not None, "Define name for metric"
        return {self.name: self.avg_meter.avg}

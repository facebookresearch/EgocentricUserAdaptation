from abc import ABC, abstractmethod
import torch
from typing import Dict
from continual_ego4d.utils.meters import AverageMeter
from ego4d.evaluation import lta_metrics as metrics


class Metric(ABC):

    @property
    @abstractmethod
    def reset_before_batch(self):
        """Reset before each batch (True), or keep accumulating (False)."""

    @abstractmethod
    @torch.no_grad()
    def update(self, preds, labels):
        """Update metric from predictions and labels.
        preds: (2 x batch_size x input_shape) -> first dim = verb/noun
        labels: (2 x batch_size ) -> first dim = verb/noun
        """

    @abstractmethod
    @torch.no_grad()
    def reset(self):
        """Reset the metric."""

    @abstractmethod
    @torch.no_grad()
    def result(self) -> Dict:
        """Get the metric(s) with name in dict format."""


class OnlineTopkVerbAccMetric(Metric):
    reset_before_batch = True

    def __init__(self, k: int = 1):
        self.k = k
        self.avg_meter = AverageMeter()
        self.label_idx = 0
        self.name = f"top{self.k}_verb_err"

    @torch.no_grad()
    def update(self, preds, labels):
        """Update metric from predictions and labels."""
        assert preds.shape[1] == labels.shape[1], f"Batch sizes not matching!"
        batch_size = labels.shape[1]

        # Verb/noun errors
        topk_acc_verb = metrics.distributed_topk_errors(
            preds[0], labels[self.label_idx, :], [self.k], acc=True
        )

        self.avg_meter.update(topk_acc_verb, weight=batch_size)

    @torch.no_grad()
    def reset(self):
        """Reset the metric."""
        self.avg_meter.reset()

    @torch.no_grad()
    def result(self) -> Dict:
        """Get the metric(s) with name in dict format."""
        return {self.name: self.avg_meter.avg}


class OnlineTopkNounAccMetric(Metric):
    reset_before_batch = True

    def __init__(self, k: int = 1):
        self.k = k
        self.avg_meter = AverageMeter()
        self.label_idx = 1
        self.name = f"top{self.k}_noun_err"

    @torch.no_grad()
    def update(self, preds, labels):
        """Update metric from predictions and labels."""
        assert preds.shape[1] == labels.shape[1], f"Batch sizes not matching!"
        batch_size = labels.shape[1]

        # Verb/noun errors
        topk_acc_verb = metrics.distributed_topk_errors(
            preds[0], labels[self.label_idx, :], [self.k], acc=True
        )

        self.avg_meter.update(topk_acc_verb, weight=batch_size)

    @torch.no_grad()
    def reset(self):
        """Reset the metric."""
        self.avg_meter.reset()

    @torch.no_grad()
    def result(self) -> Dict:
        """Get the metric(s) with name in dict format."""
        return {self.name: self.avg_meter.avg}


class OnlineTop1ActionAccMetric(Metric):
    reset_before_batch = True

    def __init__(self, reset_before_batch=True):
        self.avg_meter = AverageMeter()
        self.name = "top1_action_err"
        self.reset_before_batch = reset_before_batch

    @torch.no_grad()
    def update(self, preds, labels):
        """Update metric from predictions and labels."""
        assert preds.shape[1] == labels.shape[1], f"Batch sizes not matching!"
        batch_size = labels.shape[1]

        # Combine
        top1_acc_action = metrics.distributed_twodistr_top1_errors(preds, labels, acc=True)

        self.avg_meter.update(top1_acc_action, weight=batch_size)

    @torch.no_grad()
    def reset(self):
        """Reset the metric."""
        self.avg_meter.reset()

    @torch.no_grad()
    def result(self) -> Dict:
        """Get the metric(s) with name in dict format."""
        return {self.name: self.avg_meter.avg}


class AvgTopkVerbAccMetric(OnlineTopkVerbAccMetric):
    reset_before_batch = False


class AvgTopkNounAccMetric(OnlineTopkNounAccMetric):
    reset_before_batch = False


class AvgTop1ActionAccMetric(OnlineTop1ActionAccMetric):
    reset_before_batch = False


# class OnlineForgetting(Metric):
# class ReexposureForgetting(Metric):
# class OnlineForgetting(Metric):
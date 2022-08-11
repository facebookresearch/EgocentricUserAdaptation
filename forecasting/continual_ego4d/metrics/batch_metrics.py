import logging
from abc import ABC, abstractmethod
import torch
from typing import Dict, Set, Union, Tuple
from continual_ego4d.utils.meters import AverageMeter
from ego4d.evaluation import lta_metrics as metrics


class Metric(ABC):

    @property
    @abstractmethod
    def reset_before_batch(self):
        """Reset before each batch (True), or keep accumulating (False)."""

    @abstractmethod
    @torch.no_grad()
    def update(self, preds: list, labels):
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


class OnlineTopkAccMetric(Metric):
    reset_before_batch = True

    modes = ["verb", "noun", "action"]

    def __init__(self, k: int = 1, mode="action"):
        self.k = k
        self.name = f"top{self.k}_{mode}_acc"
        self.avg_meter = AverageMeter()

        self.mode = mode
        assert self.mode in self.modes
        if self.mode == 'verb':
            self.label_idx = 0
        elif self.mode == 'noun':
            self.label_idx = 1
        elif self.mode == 'action':
            assert self.k == 1, f"Action mode only supports top1, not top-{self.k}"
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def update(self, preds, labels):
        """Update metric from predictions and labels."""
        assert preds[0].shape[0] == labels.shape[0], f"Batch sizes not matching!"
        batch_size = labels.shape[0]

        # Verb/noun errors
        if self.mode in ['verb', 'noun']:
            topk_acc: torch.FloatTensor = metrics.distributed_topk_errors(
                preds[self.label_idx], labels[:, self.label_idx], [self.k], acc=True
            )[0]  # Unpack self.k
        elif self.mode in ['action']:
            topk_acc: torch.FloatTensor = metrics.distributed_twodistr_top1_errors(
                preds[0], preds[1], labels[:, 0], labels[:, 1], acc=True)

        self.avg_meter.update(topk_acc.item(), weight=batch_size)

    @torch.no_grad()
    def reset(self):
        """Reset the metric."""
        self.avg_meter.reset()

    @torch.no_grad()
    def result(self) -> Dict:
        """Get the metric(s) with name in dict format."""
        return {self.name: self.avg_meter.avg}


class RunningAvgOnlineTopkAccMetric(OnlineTopkAccMetric):
    reset_before_batch = False

    def __init__(self, k: int = 1, mode="action"):
        super().__init__(k, mode)
        self.name = f"running_avg_{self.name}"


class CountMetric(Metric):
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
    def update(self, preds, labels):
        """Update metric from predictions and labels."""
        pass

    @torch.no_grad()
    def reset(self):
        """Reset the metric."""
        pass

    @torch.no_grad()
    def result(self) -> Dict:
        """Get the metric(s) with name in dict format."""
        ret = []

        # Count in observed set
        ret.append(
            (f"{self.observed_set_name}_{self.mode}_count",
             len(self.observed_set))
        )

        if self.ref_set is not None:
            intersect_set = {x for x in self.observed_set if x in self.ref_set}

            # Count in reference set
            ret.append(
                (f"{self.ref_set_name}_{self.mode}_count",
                 len(self.ref_set))
            )

            # Count of intersection
            ret.append(
                (f"intersect_{self.observed_set_name}_VS_{self.ref_set_name}_{self.mode}_count",
                 len(intersect_set))
            )

            # Fraction intersection/observed set
            if len(self.observed_set) > 0:
                ret.append((
                    f"intersect_{self.observed_set_name}_VS_{self.ref_set_name}_{self.mode}_{self.observed_set_name}-fract",
                    len(intersect_set) / len(self.observed_set)
                ))

            # Fraction intersection/ref set
            if len(self.ref_set) > 0:
                ret.append((
                    f"intersect_{self.observed_set_name}_VS_{self.ref_set_name}_{self.mode}_{self.ref_set_name}-fract",
                    len(intersect_set) / len(self.ref_set)
                ))

        return {k: v for k, v in ret}

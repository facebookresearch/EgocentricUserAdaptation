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
    def update(self, preds: list, labels, stream_sample_ids=None):
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
    def update(self, preds, labels, stream_sample_ids=None):
        """Update metric from predictions and labels."""
        assert preds[0].shape[0] == labels.shape[0], f"Batch sizes not matching!"
        batch_size = labels.shape[0]

        # Verb/noun errors
        if self.mode in ['verb', 'noun']:
            topk_acc: float = metrics.distributed_topk_errors(
                preds[self.label_idx], labels[:, self.label_idx], [self.k], acc=True
            )[0]
        elif self.mode in ['action']:
            topk_acc: float = metrics.distributed_twodistr_top1_errors(
                preds[0], preds[1], labels[:, 0], labels[:, 1], acc=True)

        self.avg_meter.update(topk_acc, weight=batch_size)

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


class ConditionalOnlineTopkAccMetric(OnlineTopkAccMetric):
    reset_before_batch = True

    def __init__(self, k, mode, name_prefix, cond_set: Union[Set[int], Set[Tuple[int]], Dict[int], Dict[Tuple[int]]],
                 in_cond_set=True):
        """
        Mask out predictions, but keep those that are in (in_cond_set=True) or not in (in_cond_set=False) the
        given label set (cond_set).
        :param name_prefix: Indicates what the metric stands for
        :param cond_set: Is a set or dict either with int's indicating the labels, or tuples of 2 ints indicating the pairs of
        labels. e.g. for actions we need (verb,noun) labels).
        """
        super().__init__(k, mode)
        self.name = f"{name_prefix}_{self.name}"
        self.cond_set: Union[Set, Dict] = cond_set  # Conditional set: Which samples to look at
        self.in_cond_set = in_cond_set

    @torch.no_grad()
    def update(self, preds, labels, stream_sample_ids=None):
        """Update metric from predictions and labels."""
        assert preds[0].shape[0] == labels.shape[0], f"Batch sizes not matching!"

        if self.in_cond_set and len(self.cond_set) <= 0:
            return

        # Verb/noun errors
        if self.mode in ['verb', 'noun']:
            topk_acc, subset_batch_size = self._get_verbnoun_topk_acc(preds, labels)

        # Action errors
        elif self.mode in ['action']:
            topk_acc, subset_batch_size = self._get_action_topk_acc(preds, labels)

        else:
            raise NotImplementedError()

        # Update
        self.avg_meter.update(topk_acc, weight=subset_batch_size)

    def _get_verbnoun_topk_acc(self, preds, labels):
        target_preds = preds[self.label_idx]
        target_labels = labels[:, self.label_idx]

        label_mask = sum(target_labels == el for el in self.cond_set).bool()
        if not self.in_cond_set:  # Reverse
            label_mask = ~label_mask

        subset_batch_size = sum(label_mask)
        if subset_batch_size <= 0:  # No selected
            topk_acc = 0
        else:
            subset_preds = target_preds[label_mask]
            subset_labels = target_labels[label_mask]

            topk_acc: float = metrics.distributed_topk_errors(
                subset_preds, subset_labels, [self.k], acc=True)[0]

        return topk_acc, subset_batch_size

    def _get_action_topk_acc(self, preds, labels):
        # Get mask
        label_batch_axis = 0
        label_mask = torch.stack([
            torch.BoolTensor([tuple((verbnoun_t.tolist())) in self.cond_set])  # Is (verb,noun) tuple in cond_set?
            for verbnoun_t in torch.unbind(labels, dim=label_batch_axis)
        ], dim=label_batch_axis)
        label_mask.squeeze_()  # Get rid of stack dim

        if not self.in_cond_set:  # Reverse
            label_mask = ~label_mask

        subset_batch_size = sum(label_mask)
        if subset_batch_size <= 0:  # No selected
            topk_acc = 0
        else:
            # Subset
            preds1, preds2 = preds[0][label_mask], preds[1][label_mask]
            subset_labels = labels[label_mask]
            labels1, labels2 = subset_labels[:, 0], subset_labels[:, 1]

            topk_acc: float = metrics.distributed_twodistr_top1_errors(
                preds1, preds2, labels1, labels2, acc=True)

        return topk_acc, subset_batch_size


class GeneralizationTopkAccMetric(ConditionalOnlineTopkAccMetric):
    """ Measure on future data for actions that have been seen in history data."""

    def __init__(self, seen_action_set, k, mode):
        super().__init__(
            k=k, mode=mode,
            name_prefix='GEN', cond_set=seen_action_set, in_cond_set=True  # Fixed
        )


class FWTTopkAccMetric(ConditionalOnlineTopkAccMetric):
    """ Measure on future data for actions that have NOT been seen in history data."""

    def __init__(self, seen_action_set, k, mode):
        super().__init__(
            k=k, mode=mode,
            name_prefix='FWT', cond_set=seen_action_set, in_cond_set=False  # Fixed
        )


class ConditionalOnlineForgettingMetric(OnlineTopkAccMetric):
    reset_before_batch = True  # Resets forgetting Avg metrics, but history is kept

    def __init__(self, k, mode, name_prefix, cond_set: Union[Set[int], Set[Tuple[int]]], in_cond_set=True):
        """
        Mask out predictions, but keep those that are in (in_cond_set=True) or not in (in_cond_set=False) the
        given label set (cond_set).
        :param name_prefix: Indicates what the metric stands for
        :param cond_set: Is a set either with int's indicating the labels, or tuples of 2 ints indicating the pairs of
        labels. e.g. for actions we need (verb,noun) labels).
        """
        super().__init__(k, mode)
        self.name = f"{name_prefix}_{self.name}"
        self.cond_set: set = cond_set  # Conditional set: Which samples to look at
        self.in_cond_set = in_cond_set

        """ Set with observed actions (seen set), and for each observed action, we keep:
        - The previous accuracy (at prev time t_a)
        - The sample_idx of the prev instance of this action: (represents t_a)
        - Current AverageMeter for current Accuracy (Stop counting for t > t_a).
        
        Then avg over all these average meters their results to get the final result.        
        """

    @torch.no_grad()
    def update(self, preds, labels, stream_sample_ids):
        """Update metric from predictions and labels."""
        assert preds[0].shape[0] == labels.shape[0], f"Batch sizes not matching!"

        """
        Use sampler! Keep all actions that have been seen, and keep last observation sample_idx.
        Get 1 metric per seen action, and iterate over all history samples with max(all_seen_actions_observation_sample_idxs).
        So basically the latest one.
        
        -> This can also be conditional that they are in the condiitonal set (like future)
        
        Each action-metric only accumulates until its 'seen'-index.
        
        Afterwards the average over all is returned.
        
        """
        if self.in_cond_set and len(self.cond_set) <= 0:
            return

        # Verb/noun errors
        if self.mode in ['verb', 'noun']:
            topk_acc, subset_batch_size = self._get_verbnoun_topk_acc(preds, labels)

        # Action errors
        elif self.mode in ['action']:
            topk_acc, subset_batch_size = self._get_action_topk_acc(preds, labels)

        else:
            raise NotImplementedError()

        # Update
        self.avg_meter.update(topk_acc, weight=subset_batch_size)

    def _get_verbnoun_topk_acc(self, preds, labels):
        target_preds = preds[self.label_idx]
        target_labels = labels[:, self.label_idx]

        label_mask = sum(target_labels == el for el in self.cond_set).bool()
        if not self.in_cond_set:  # Reverse
            label_mask = ~label_mask

        subset_batch_size = sum(label_mask)
        if subset_batch_size <= 0:  # No selected
            topk_acc = 0
        else:
            subset_preds = target_preds[label_mask]
            subset_labels = target_labels[label_mask]

            topk_acc: float = metrics.distributed_topk_errors(
                subset_preds, subset_labels, [self.k], acc=True)[0]

        return topk_acc, subset_batch_size

    def _get_action_topk_acc(self, preds, labels):

        import pdb;
        pdb.set_trace()

        # Get mask
        label_batch_axis = 0
        label_mask = torch.stack([
            torch.BoolTensor([tuple((verbnoun_t.tolist())) in self.cond_set])  # Is (verb,noun) tuple in cond_set?
            for verbnoun_t in torch.unbind(labels, dim=label_batch_axis)
        ], dim=label_batch_axis)
        label_mask.squeeze_()  # Get rid of stack dim

        if not self.in_cond_set:  # Reverse
            label_mask = ~label_mask

        subset_batch_size = sum(label_mask)
        if subset_batch_size <= 0:  # No selected
            topk_acc = 0
        else:
            # Subset
            preds1, preds2 = preds[0][label_mask], preds[1][label_mask]
            subset_labels = labels[label_mask]
            labels1, labels2 = subset_labels[:, 0], subset_labels[:, 1]

            topk_acc: float = metrics.distributed_twodistr_top1_errors(
                preds1, preds2, labels1, labels2, acc=True)

        return topk_acc, subset_batch_size

# class OnlineForgetting(Metric):
# class ReexposureForgetting(Metric):
# class OnlineForgetting(Metric):

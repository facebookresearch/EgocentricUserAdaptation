import logging
from abc import ABC, abstractmethod
import torch
from typing import Dict, Set, Union, Tuple
from continual_ego4d.utils.meters import AverageMeter
from ego4d.evaluation import lta_metrics as metrics
from collections import defaultdict
from continual_ego4d.datasets.continual_action_recog_dataset import verbnoun_to_action
from continual_ego4d.metrics.batch_metrics import OnlineTopkAccMetric


class ConditionalOnlineTopkAccMetric(OnlineTopkAccMetric):
    reset_before_batch = True

    def __init__(self, k, mode, name_prefix, cond_set: Union[Set[int], Set[Tuple[int]]], in_cond_set=True):
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
    def update(self, preds, labels):
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
        self.avg_meter.update(topk_acc.item(), weight=subset_batch_size)

    def _get_verbnoun_topk_acc(self, preds, labels):
        target_preds = preds[self.label_idx]
        target_labels = labels[:, self.label_idx]

        try:
            if len(self.cond_set) == 0:
                label_mask = torch.zeros_like(target_labels).bool()
            else:
                label_mask = sum(target_labels == el for el in self.cond_set).bool()
        except Exception as e:
            print(e)
            import pdb;
            pdb.set_trace()

        if not self.in_cond_set:  # Reverse
            label_mask = ~label_mask

        subset_batch_size = sum(label_mask)
        if subset_batch_size <= 0:  # No selected
            topk_acc = torch.FloatTensor([0])
        else:
            subset_preds = target_preds[label_mask]
            subset_labels = target_labels[label_mask]

            topk_acc: torch.FloatTensor = metrics.distributed_topk_errors(
                subset_preds, subset_labels, [self.k], acc=True)[0]

        return topk_acc, subset_batch_size

    def _get_action_topk_acc(self, preds, labels):
        # Get mask
        label_batch_axis = 0
        label_mask = torch.cat([
            torch.BoolTensor([verbnoun_to_action(*verbnoun_t.tolist()) in self.cond_set])
            for verbnoun_t in torch.unbind(labels, dim=label_batch_axis)
        ], dim=label_batch_axis)

        if not self.in_cond_set:  # Reverse
            label_mask = ~label_mask

        subset_batch_size = sum(label_mask)
        if subset_batch_size <= 0:  # No selected
            topk_acc = torch.FloatTensor([0])
        else:
            # Subset
            preds1, preds2 = preds[0][label_mask], preds[1][label_mask]
            subset_labels = labels[label_mask]
            labels1, labels2 = subset_labels[:, 0], subset_labels[:, 1]

            topk_acc: torch.FloatTensor = metrics.distributed_twodistr_top1_errors(
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

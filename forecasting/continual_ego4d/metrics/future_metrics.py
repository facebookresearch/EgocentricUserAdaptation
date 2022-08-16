import logging
from abc import ABC, abstractmethod
import torch
from typing import Dict, Set, Union, Tuple
from continual_ego4d.utils.meters import AverageMeter
from ego4d.evaluation import lta_metrics as metrics
from collections import defaultdict
from continual_ego4d.datasets.continual_action_recog_dataset import verbnoun_to_action
from continual_ego4d.metrics.metric import AvgMeterMetric, Metric, get_metric_tag

from ego4d.utils import logging

logger = logging.get_logger(__name__)

TAG_FUTURE = 'future'


class ConditionalOnlineMetric(AvgMeterMetric):
    reset_before_batch = True
    base_metric_modes = ['loss', 'acc']

    def __init__(self, base_metric_mode: str, action_mode: str, main_metric_name: str,
                 cond_set: Union[Set[int], Set[Tuple[int]]], in_cond_set=True,
                 loss_fun=None, k=None
                 ):
        """
        Mask out predictions, but keep those that are in (in_cond_set=True) or not in (in_cond_set=False) the
        given label set (cond_set).
        :param main_metric_name: Indicates what the metric stands for. e.g. FWT, GEN,...
        :param cond_set: Is a set or dict either with int's indicating the labels, or tuples of 2 ints indicating the pairs of
        labels. e.g. for actions we need (verb,noun) labels).
        """
        super().__init__(action_mode)
        self.name = get_metric_tag(TAG_FUTURE, action_mode=self.mode,
                                   base_metric_name=f"{main_metric_name}_{base_metric_mode}")
        self.cond_set: Union[Set, Dict] = cond_set  # Conditional set: Which samples to look at
        self.in_cond_set = in_cond_set

        self.base_metric_mode = base_metric_mode
        assert self.base_metric_mode in self.base_metric_modes

        if self.base_metric_mode == 'loss':
            assert loss_fun is not None, f"Need to pass loss_fun to metric if in loss mode."
            self.loss_fun = loss_fun
        elif self.base_metric_mode == 'acc':
            assert k is not None, f"Need to pass loss_fun to metric if in loss mode."

    @torch.no_grad()
    def update(self, preds, labels, *args, **kwargs):
        """Update metric from predictions and labels."""
        assert preds[0].shape[0] == labels.shape[0], f"Batch sizes not matching!"

        if self.in_cond_set and len(self.cond_set) <= 0:
            return

        # Verb/noun errors
        if self.mode in ['verb', 'noun']:
            metric_result, subset_batch_size = self._get_verbnoun_metric_result(preds, labels)

        # Action errors
        elif self.mode in ['action']:
            metric_result, subset_batch_size = self._get_action_metric_result(preds, labels)

        else:
            raise NotImplementedError()

        # Update
        if metric_result is not None:
            self.avg_meter.update(metric_result.item(), weight=subset_batch_size)

    def _get_verbnoun_metric_result(self, preds, labels):
        target_preds = preds[self.label_idx]
        target_labels = labels[:, self.label_idx]

        try:
            # Type check
            if len(self.cond_set) > 0:
                assert isinstance(next(iter(self.cond_set)), int)
                label_mask = sum(target_labels == el for el in self.cond_set).bool()  # Match tensors
            else:
                label_mask = torch.zeros_like(target_labels).bool()
        except Exception as e:  # TODO FIXME Remove this
            import pdb
            import traceback
            logger.debug(traceback.format_exc())
            logger.debug(f"target_labels={target_labels}, cond_set={self.cond_set}")
            logger.debug(f"expr={sum(target_labels == el for el in self.cond_set)}")

        if not self.in_cond_set:  # Reverse
            label_mask = ~label_mask

        subset_batch_size = sum(label_mask)
        metric_result = None
        if subset_batch_size > 0:  # When samples selected
            subset_preds = target_preds[label_mask]
            subset_labels = target_labels[label_mask]

            if self.base_metric_mode == 'acc':
                metric_result: torch.FloatTensor = metrics.distributed_topk_errors(
                    subset_preds, subset_labels, [self.k], return_mode='acc')[0]
            elif self.base_metric_mode == 'loss':
                metric_result: torch.FloatTensor = self.loss_fun(subset_preds, subset_labels)

        return metric_result, subset_batch_size

    def _get_action_metric_result(self, preds, labels):
        # Get mask
        label_batch_axis = 0
        label_mask = torch.cat([
            torch.BoolTensor([verbnoun_to_action(*verbnoun_t.tolist()) in self.cond_set])
            for verbnoun_t in torch.unbind(labels, dim=label_batch_axis)
        ], dim=label_batch_axis)

        if not self.in_cond_set:  # Reverse
            label_mask = ~label_mask

        subset_batch_size = sum(label_mask)
        metric_result = None
        if subset_batch_size > 0:  # When samples selected
            # Subset
            preds1, preds2 = preds[0][label_mask], preds[1][label_mask]
            subset_labels = labels[label_mask]
            labels1, labels2 = subset_labels[:, 0], subset_labels[:, 1]

            if self.base_metric_mode == 'acc':
                metric_result: torch.FloatTensor = metrics.distributed_twodistr_top1_errors(
                    preds1, preds2, labels1, labels2, return_mode='acc')
            elif self.base_metric_mode == 'loss':
                verb_loss: torch.FloatTensor = self.loss_fun(preds1, labels1)
                noun_loss: torch.FloatTensor = self.loss_fun(preds2, labels2)
                metric_result: torch.FloatTensor = verb_loss + noun_loss

        return metric_result, subset_batch_size


class GeneralizationTopkAccMetric(ConditionalOnlineMetric):
    """ Measure on future data for actions that have been seen in history data."""

    def __init__(self, seen_action_set, k, action_mode):
        super().__init__(
            base_metric_mode='acc', main_metric_name='GEN',
            k=k, loss_fun=None,  # Acc
            action_mode=action_mode,
            cond_set=seen_action_set, in_cond_set=True  # Fixed
        )


class FWTTopkAccMetric(ConditionalOnlineMetric):
    """ Measure on future data for actions that have NOT been seen in history data."""

    def __init__(self, seen_action_set, k, action_mode):
        super().__init__(
            base_metric_mode='acc', main_metric_name='FWT',
            k=k, loss_fun=None,  # Acc
            action_mode=action_mode,
            cond_set=seen_action_set, in_cond_set=False  # Fixed
        )


class GeneralizationLossMetric(ConditionalOnlineMetric):
    """ Measure on future data for actions that have been seen in history data."""

    def __init__(self, seen_action_set, action_mode, loss_fun):
        super().__init__(
            base_metric_mode='loss', main_metric_name='GEN',
            k=None, loss_fun=loss_fun,  # Loss
            action_mode=action_mode,
            cond_set=seen_action_set, in_cond_set=True  # Fixed
        )


class FWTLossMetric(ConditionalOnlineMetric):
    """ Measure on future data for actions that have NOT been seen in history data."""

    def __init__(self, seen_action_set, action_mode, loss_fun):
        super().__init__(
            base_metric_mode='loss', main_metric_name='FWT',
            k=None, loss_fun=loss_fun,  # Loss
            action_mode=action_mode,
            cond_set=seen_action_set, in_cond_set=False  # Fixed
        )

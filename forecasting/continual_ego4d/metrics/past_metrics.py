import logging
from abc import ABC, abstractmethod
import torch
from typing import Dict, Set, Union, Tuple
from continual_ego4d.utils.meters import AverageMeter
from ego4d.evaluation import lta_metrics as metrics
from collections import defaultdict
from continual_ego4d.datasets.continual_action_recog_dataset import verbnoun_to_action
from continual_ego4d.metrics.metric import Metric, ACTION_MODES, get_metric_tag

from ego4d.utils import logging

logger = logging.get_logger(__name__)

TAG_PAST = 'past'


class ConditionalOnlineForgettingMetric(Metric):
    reset_before_batch = True  # Resets forgetting Avg metrics, but history is kept
    modes = ACTION_MODES
    base_metric_modes = ['loss', 'acc']

    def __init__(self, base_metric_mode: str, action_mode: str, main_metric_name: str,
                 current_batch_cond_set=False, in_cond_set=True,
                 k: int = None, loss_fun=None,
                 ):
        """
        Takes in the history stream, and calculates forgetting per action.
        The result gives the average over all action-accuracies deltas, each minus their previous stored accuracy.
        This delta equals to Forgetting.

        The action_cond_set can be used to only consider action in the update() method that are in the cond_set
        (if in_cond_set=True) or are NOT in the cond_set (in_cond_set=False).

        For each observed action, we keep:
        - The sample_id of the prev instance of this action: (represents t_a)
        - The previous accuracy (at last observed time t_a)
        - Current AverageMeter for current Accuracy (Stops counting for t > t_a).

        Notes:
        The metric stores the accuracies for previous actions on save_result_to_history.
        The reset only resets the running average meters, not the history.

        :param: current_batch_cond_set: Use current observed batch as conditional set for constraints?
        :param: in_cond_set: If current_batch_cond_set=True, should consider samples in or out this set?
        """
        self.action_mode = action_mode
        self.base_metric_mode = base_metric_mode

        # Base Metric-mode specific
        if self.base_metric_mode == 'loss':
            assert loss_fun is not None, f"Need to pass loss_fun to metric if in loss mode."
            self.loss_fun = loss_fun
            self.delta_fun = lambda current, prev: current - prev  # Reverse of accuracy because lower is better
            basic_metric_name = f"{main_metric_name}_{self.base_metric_mode}"

        elif self.base_metric_mode == 'acc':
            assert k is not None, f"Need to pass loss_fun to metric if in loss mode."
            self.k = k
            self.delta_fun = lambda current, prev: prev - current
            basic_metric_name = f"{main_metric_name}_top{self.k}{self.base_metric_mode}"

            if self.action_mode == 'action':
                assert self.k == 1, f"Action mode only supports top1, not top-{self.k}"
        else:
            raise ValueError()

        # Action mode specific
        assert self.action_mode in self.modes
        if self.action_mode == 'verb':
            self.label_idx = 0
        elif self.action_mode == 'noun':
            self.label_idx = 1

        # Forgetting states
        self.action_to_prev_perf = {}
        self.action_to_current_perf = defaultdict(AverageMeter)

        # Filtering in Update-method
        self.in_cond_set = in_cond_set  # Only consider actions in the cond set if True, Outside if False.
        self.current_batch_cond_set = current_batch_cond_set

        self.name = get_metric_tag(TAG_PAST, action_mode=self.action_mode, base_metric_name=basic_metric_name)

    @torch.no_grad()
    def update(self, preds, labels, *args, stream_batch_labels=None, **kwargs):
        """
        ASSUMPTION: The preds,labels are all from the history stream only, forwarded on the CURRENT model.
        Update per-action accuracy metric from predictions and labels.
        """
        assert preds[0].shape[0] == labels.shape[0], f"Batch sizes not matching!"
        assert stream_batch_labels is not None

        # Verb/noun errors
        cond_set = None
        if self.action_mode in ['verb', 'noun']:

            # Use current batch as conditional set
            if self.current_batch_cond_set:
                # TODO: This is NOT Condition Set, Should be the actual current batch in stream, this is the batch in history!!
                cond_set = set(stream_batch_labels[:, self.label_idx].tolist())

            self._get_verbnoun_metric_result(preds, labels, cond_set)

        # Action errors
        elif self.action_mode in ['action']:

            # Use current batch as conditional set
            if self.current_batch_cond_set:
                cond_set = set()
                for (verb, noun) in stream_batch_labels.tolist():
                    action = verbnoun_to_action(verb, noun)
                    cond_set.add(action)
            self._get_action_metric_result(preds, labels, cond_set)

        else:
            raise NotImplementedError()

    def _satisfies_conditions(self, target, cond_set):
        if cond_set is not None and (
                (self.in_cond_set and target not in cond_set) or
                (not self.in_cond_set and target in cond_set)
        ):
            return False
        return True

    def _get_verbnoun_metric_result(self, preds, labels, cond_set):
        # Iterate actions in batch, continue if not satisfying filter constraint
        obs_actions_batch = set()
        label_batch_axis = 0
        for idx, verbnoun_t in enumerate(torch.unbind(labels, dim=label_batch_axis)):
            verbnoun = verbnoun_t[self.label_idx].item()  # Verb or noun
            verbnoun_labels = labels[:, self.label_idx]
            verbnoun_preds = preds[self.label_idx]

            # Filter out based on conditional set
            if not self._satisfies_conditions(verbnoun, cond_set):
                continue

            # Already processed, so skip
            if verbnoun in obs_actions_batch:
                continue
            obs_actions_batch.add(verbnoun)

            # Mask
            label_mask = (verbnoun_labels == verbnoun).bool()
            subset_batch_size = sum(label_mask)

            try:
                subset_preds = verbnoun_preds[label_mask]
                subset_labels = verbnoun_labels[label_mask]
            except:
                import traceback
                logger.debug(traceback.format_exc())

            # Acc metric
            if self.base_metric_mode == 'acc':
                metric_result: torch.FloatTensor = metrics.distributed_topk_errors(
                    subset_preds, subset_labels, [self.k], return_mode='acc')[0]
            elif self.base_metric_mode == 'loss':
                metric_result: torch.FloatTensor = self.loss_fun(subset_preds, subset_labels)

            self.action_to_current_perf[verbnoun].update(metric_result.item(), weight=subset_batch_size)

    def _get_action_metric_result(self, preds, labels, cond_set):
        # Iterate actions in batch, continue if not satisfying filter constraint
        obs_actions_batch = set()
        label_batch_axis = 0
        for idx, verbnoun_t in enumerate(torch.unbind(labels, dim=label_batch_axis)):
            action = verbnoun_to_action(*verbnoun_t.tolist())

            # Filter out based on conditional set
            if not self._satisfies_conditions(action, cond_set):
                continue

            # Already processed, so skip
            if action in obs_actions_batch:
                continue
            obs_actions_batch.add(action)

            # Process all samples in batch for this action at once
            label_mask = torch.cat([
                torch.BoolTensor([verbnoun_to_action(*verbnoun_t.tolist()) == action])
                for verbnoun_t in torch.unbind(labels, dim=label_batch_axis)
            ], dim=label_batch_axis)

            # Subset
            subset_batch_size = sum(label_mask)
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

            self.action_to_current_perf[action].update(metric_result.item(), weight=subset_batch_size)

    @torch.no_grad()
    def reset(self):
        """Reset the metric."""
        self.action_to_current_perf = defaultdict(AverageMeter)

    @torch.no_grad()
    def result(self) -> Dict:
        """Get the metric(s) with name in dict format.
        Averages with equal weight over all actions with a meter.
        """
        # avg acc results over all actions, and make relative to previous acc (forgetting)
        avg_forg_over_actions = AverageMeter()
        for action, action_current_acc in self.action_to_current_perf.items():
            if action not in self.action_to_prev_perf:  # First time the acc is measured on the model for this action
                logger.debug(f"{self.action_mode} {action} acc measured for first time in history stream.")
                continue
            current_acc = action_current_acc.avg
            assert action_current_acc.count > 0, f"Acc {current_acc} has zero count."

            forg = self.delta_fun(current_acc, self.action_to_prev_perf[action])
            avg_forg_over_actions.update(forg, weight=1)

        if avg_forg_over_actions.count == 0:
            return {}
        return {self.name: avg_forg_over_actions.avg}

    @torch.no_grad()
    def save_result_to_history(self):
        """ Save acc on current model results per action in the dict."""
        for action, action_current_perf in self.action_to_current_perf.items():
            if action_current_perf.count == 0:
                continue
            self.action_to_prev_perf[action] = action_current_perf.avg


class FullOnlineForgettingAccMetric(ConditionalOnlineForgettingMetric):
    """ Measure on history data over all actions that have already been observed."""

    def __init__(self, k, action_mode):
        super().__init__(
            base_metric_mode='acc', k=k, loss_fun=None,  # Acc
            action_mode=action_mode, main_metric_name="FORG_full",
            # current_batch_cond_set=False, in_cond_set=False  # Fixed
        )


class FullOnlineForgettingLossMetric(ConditionalOnlineForgettingMetric):
    """ Measure on history data over all actions that have already been observed."""

    def __init__(self, loss_fun, action_mode):
        super().__init__(
            base_metric_mode='loss', k=None, loss_fun=loss_fun,  # Acc
            action_mode=action_mode, main_metric_name="FORG_full",
            # current_batch_cond_set=False, in_cond_set=False  # Fixed
        )


class ReexposureForgettingAccMetric(ConditionalOnlineForgettingMetric):
    """ Measure on history data over actions that have already been observed AND are observed in current mini-batch."""

    def __init__(self, k, action_mode):
        super().__init__(
            base_metric_mode='acc', k=k, loss_fun=None,  # Acc
            action_mode=action_mode, main_metric_name="FORG_EXPOSE",
            current_batch_cond_set=True, in_cond_set=True,
        )


class ReexposureForgettingLossMetric(ConditionalOnlineForgettingMetric):
    """ Measure on history data over actions that have already been observed AND are observed in current mini-batch."""

    def __init__(self, loss_fun, action_mode):
        super().__init__(
            base_metric_mode='loss', k=None, loss_fun=loss_fun,  # Acc
            action_mode=action_mode, main_metric_name="FORG_EXPOSE",
            current_batch_cond_set=True, in_cond_set=True,
        )


class CollateralForgettingAccMetric(ConditionalOnlineForgettingMetric):
    """ Measure on history data over actions that have already been observed
    AND are NOTobserved in current mini-batch."""

    def __init__(self, k, action_mode):
        super().__init__(
            base_metric_mode='acc', k=k, loss_fun=None,  # Acc
            action_mode=action_mode, main_metric_name="FORG_COLLAT",
            current_batch_cond_set=True, in_cond_set=False
        )


class CollateralForgettingLossMetric(ConditionalOnlineForgettingMetric):
    """ Measure on history data over actions that have already been observed
    AND are NOTobserved in current mini-batch."""

    def __init__(self, loss_fun, action_mode):
        super().__init__(
            base_metric_mode='loss', k=None, loss_fun=loss_fun,  # Acc
            action_mode=action_mode, main_metric_name="FORG_COLLAT",
            current_batch_cond_set=True, in_cond_set=False
        )

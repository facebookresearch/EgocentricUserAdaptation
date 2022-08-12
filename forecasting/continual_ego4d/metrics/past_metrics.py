import logging
from abc import ABC, abstractmethod
import torch
from typing import Dict, Set, Union, Tuple
from continual_ego4d.utils.meters import AverageMeter
from ego4d.evaluation import lta_metrics as metrics
from collections import defaultdict
from continual_ego4d.datasets.continual_action_recog_dataset import verbnoun_to_action
from continual_ego4d.metrics.batch_metrics import Metric


class ConditionalOnlineForgettingMetric(Metric):
    reset_before_batch = True  # Resets forgetting Avg metrics, but history is kept
    modes = ["verb", "noun", "action"]

    def __init__(self, k: int = 1, mode="action", name_prefix="",
                 current_batch_cond_set=False, in_cond_set=True):
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
        """
        self.k = k
        if len(name_prefix) > 0:
            name_prefix = f"{name_prefix}_"
        self.name = f"{name_prefix}top{self.k}_{mode}_forg"

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

        # Forgetting states
        self.action_to_prev_acc = {}
        self.action_to_current_acc = defaultdict(AverageMeter)

        # Filtering in Update-method
        self.in_cond_set = in_cond_set  # Only consider actions in the cond set if True, Outside if False.
        self.current_batch_cond_set = current_batch_cond_set

    @torch.no_grad()
    def update(self, preds, labels, *args, **kwargs):
        """
        ASSUMPTION: The preds,labels are all from the history stream only, forwarded on the CURRENT model.
        Update per-action accuracy metric from predictions and labels.
        """
        assert preds[0].shape[0] == labels.shape[0], f"Batch sizes not matching!"

        # Verb/noun errors
        cond_set = None
        if self.mode in ['verb', 'noun']:

            # Use current batch as conditional set
            if self.current_batch_cond_set:
                cond_set = set(labels[:, self.label_idx].tolist())

            self._get_verbnoun_topk_acc(preds, labels, cond_set)

        # Action errors
        elif self.mode in ['action']:

            # Use current batch as conditional set
            if self.current_batch_cond_set:
                cond_set = set()
                for (verb, noun) in labels.tolist():
                    action = verbnoun_to_action(verb, noun)
                    cond_set.add(action)
            self._get_action_topk_acc(preds, labels, cond_set)

        else:
            raise NotImplementedError()

    def _satisfies_conditions(self, target, cond_set):
        if cond_set is not None and (
                (self.in_cond_set and target not in cond_set) or
                (not self.in_cond_set and target in cond_set)
        ):
            return False
        return True

    def _get_verbnoun_topk_acc(self, preds, labels, cond_set):
        # Iterate actions in batch, continue if not satisfying filter constraint
        obs_actions_batch = set()
        label_batch_axis = 0
        for idx, verbnoun_t in enumerate(torch.unbind(labels, dim=label_batch_axis)):
            verbnoun = verbnoun_t[self.label_idx].item()  # Verb or noun

            # Filter out based on conditional set
            if not self._satisfies_conditions(verbnoun, cond_set):
                continue

            # Already processed, so skip
            if verbnoun in obs_actions_batch:
                continue
            obs_actions_batch.add(verbnoun)

            # Mask
            label_mask = (labels == verbnoun).bool()
            subset_batch_size = sum(label_mask)

            subset_preds = preds[label_mask]
            subset_labels = labels[label_mask]

            # Acc metric
            topk_acc: torch.FloatTensor = metrics.distributed_topk_errors(
                subset_preds, subset_labels, [self.k], acc=True)[0]

            self.action_to_current_acc[verbnoun].update(topk_acc.item(), weight=subset_batch_size)

    def _get_action_topk_acc(self, preds, labels, cond_set):
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

            action_topk_acc: torch.FloatTensor = metrics.distributed_twodistr_top1_errors(
                preds1, preds2, labels1, labels2, acc=True)

            self.action_to_current_acc[action].update(action_topk_acc.item(), weight=subset_batch_size)

    @torch.no_grad()
    def reset(self):
        """Reset the metric."""
        self.action_to_current_acc = defaultdict(AverageMeter)

    @torch.no_grad()
    def result(self) -> Dict:
        """Get the metric(s) with name in dict format.
        Averages with equal weight over all actions with a meter.
        """

        # avg acc results over all actions, and make relative to previous acc (forgetting)
        avg_forg_over_actions = AverageMeter()
        for action, action_current_acc in self.action_to_current_acc.items():
            if action not in self.action_to_prev_acc:  # First time the acc is measured on the model for this action
                logging.debug(f"Action {action} acc measured for first time in history stream.")
                continue
            current_acc = action_current_acc.avg
            forg = current_acc - self.action_to_prev_acc[action]
            avg_forg_over_actions.update(forg, weight=1)

        return {self.name: avg_forg_over_actions.avg}

    @torch.no_grad()
    def save_result_to_history(self):
        """ Save acc on current model results per action in the dict."""
        for action, action_current_acc in self.action_to_current_acc.items():
            self.action_to_prev_acc[action] = action_current_acc.avg


class FullOnlineForgettingMetric(ConditionalOnlineForgettingMetric):
    """ Measure on history data over all actions that have already been observed."""

    def __init__(self, k, mode):
        super().__init__(
            k=k, mode=mode,
            name_prefix='', current_batch_cond_set=False,  # Fixed
        )


class ReexposureForgettingMetric(ConditionalOnlineForgettingMetric):
    """ Measure on history data over actions that have already been observed AND are observed in current mini-batch."""

    def __init__(self, k, mode):
        super().__init__(
            k=k, mode=mode,
            name_prefix='EXPOSE', current_batch_cond_set=True, in_cond_set=True
        )


class CollateralForgettingMetric(ConditionalOnlineForgettingMetric):
    """ Measure on history data over actions that have already been observed
    AND are NOTobserved in current mini-batch."""

    def __init__(self, k, mode):
        super().__init__(
            k=k, mode=mode,
            name_prefix='COLLAT', current_batch_cond_set=True, in_cond_set=False
        )

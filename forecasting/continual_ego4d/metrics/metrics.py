import logging
from abc import ABC, abstractmethod
import torch
from typing import Dict, Set, Union, Tuple
from continual_ego4d.utils.meters import AverageMeter
from ego4d.evaluation import lta_metrics as metrics
from collections import defaultdict
from continual_ego4d.datasets.continual_action_recog_dataset import verbnoun_to_action


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
        label_mask = torch.cat([
            torch.BoolTensor([verbnoun_to_action(*verbnoun_t.tolist()) in self.cond_set])
            for verbnoun_t in torch.unbind(labels, dim=label_batch_axis)
        ], dim=label_batch_axis)

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


class ConditionalOnlineForgettingMetric(Metric):
    reset_before_batch = True  # Resets forgetting Avg metrics, but history is kept
    modes = ["verb", "noun", "action"]

    def __init__(self, k: int = 1, mode="action", name_prefix="",
                 action_cond_set: Union[Set[int], Set[Tuple[int]]] = None, in_cond_set=True):
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
        """
        self.k = k
        if len(name_prefix) > 0:
            name_prefix = f"{name_prefix}_"
        self.name = f"{name_prefix}top{self.k}_{mode}_forg"

        assert mode == 'action', f"Only action mode supported for now"
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
        self.action_cond_set: set = action_cond_set  # If None, not considering cond_set

    @torch.no_grad()
    def update(self, preds, labels):
        """
        ASSUMPTION: The preds,labels are all from the history stream only, forwarded on the CURRENT model.
        Update per-action accuracy metric from predictions and labels.
        """
        assert preds[0].shape[0] == labels.shape[0], f"Batch sizes not matching!"

        # Verb/noun errors
        if self.mode in ['verb', 'noun']:
            self._get_verbnoun_topk_acc(preds, labels)

        # Action errors
        elif self.mode in ['action']:
            self._get_action_topk_acc(preds, labels)

        else:
            raise NotImplementedError()

    def _get_verbnoun_topk_acc(self, preds, labels):
        raise NotImplementedError()

    def _get_action_topk_acc(self, preds, labels):
        # Iterate actions in batch, continue if not satisfying filter constraint
        obs_actions_batch = set()
        label_batch_axis = 0
        for idx, verbnoun_t in enumerate(torch.unbind(labels, dim=label_batch_axis)):
            action = verbnoun_to_action(*verbnoun_t.tolist())

            # Filter out based on conditional set
            if self.action_cond_set is not None:
                if (self.in_cond_set and action not in self.action_cond_set) or \
                        (not self.in_cond_set and action in self.action_cond_set):
                    continue

            # Already processed, so skip
            if action in obs_actions_batch:
                continue
            obs_actions_batch.add(action)

            # Process all samples in batch for this action at once + check (verb,noun) tuple in cond_set?
            label_mask = torch.cat([
                torch.BoolTensor([verbnoun_to_action(*verbnoun_t.tolist()) == action])
                for verbnoun_t in torch.unbind(labels, dim=label_batch_axis)
            ], dim=label_batch_axis)

            # Subset
            subset_batch_size = sum(label_mask)
            preds1, preds2 = preds[0][label_mask], preds[1][label_mask]
            subset_labels = labels[label_mask]
            labels1, labels2 = subset_labels[:, 0], subset_labels[:, 1]

            action_topk_acc: float = metrics.distributed_twodistr_top1_errors(
                preds1, preds2, labels1, labels2, acc=True)

            self.action_to_current_acc[action].update(action_topk_acc, weight=subset_batch_size)

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
            name_prefix='', action_cond_set=None, in_cond_set=True  # Fixed
        )


class ReexposureForgettingMetric(ConditionalOnlineForgettingMetric):
    """ Measure on history data over actions that have already been observed AND are observed in current mini-batch."""

    def __init__(self, current_batch_actions: set, k, mode):
        super().__init__(
            k=k, mode=mode,
            name_prefix='EXPOSE', action_cond_set=current_batch_actions, in_cond_set=True
        )


class CollateralForgettingMetric(ConditionalOnlineForgettingMetric):
    """ Measure on history data over actions that have already been observed
    AND are NOTobserved in current mini-batch."""

    def __init__(self, current_batch_actions: set, k, mode):
        super().__init__(
            k=k, mode=mode,
            name_prefix='COLLAT', action_cond_set=current_batch_actions, in_cond_set=False
        )

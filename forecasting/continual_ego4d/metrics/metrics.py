from abc import ABC, abstractmethod
import torch
from typing import Dict, Set, Union, Tuple
from continual_ego4d.utils.meters import AverageMeter
from ego4d.evaluation import lta_metrics as metrics
from collections import defaultdict


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
            torch.BoolTensor([tuple((verbnoun_t.tolist())) in self.cond_set])  # Is (verb,noun) tuple in cond_set?
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

    def __init__(self, k: int = 1, mode="action"):
        """
        Takes in the history stream, and calculates accuracy per action.
        The result gives the average over all action-accuracies, each minus their previous stored accuracy.
        The metric stores the accuracies for previous actions on save_result_to_history.

        The reset only resets the running average meters, not the history.
        """

        self.k = k
        self.name = f"top{self.k}_{mode}_forg"

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
        self.action_to_avgmeter = defaultdict(AverageMeter)
        """ Set with observed actions (seen set), and for each observed action, we keep:
        - The previous accuracy (at prev time t_a)
        - The sample_idx of the prev instance of this action: (represents t_a)
        - Current AverageMeter for current Accuracy (Stop counting for t > t_a).
        
        Then avg over all these average meters their results to get the final result.        
        """

        # TODO: Add conditional set (Current learning batch, NOT EVAL BATCH in metric.update())
        #  to get disentangled forgetting metrics

    @torch.no_grad()
    def update(self, preds, labels):
        """
        ASSUMPTION: The preds,labels are all from the history stream only, forwarded on the CURRENT model.
        Update per-action accuracy metric from predictions and labels.
        """
        assert preds[0].shape[0] == labels.shape[0], f"Batch sizes not matching!"

        """
        Use sampler! Keep all actions that have been seen, and keep last observation sample_idx.
        Get 1 metric per seen action, and iterate over all history samples with max(all_seen_actions_observation_sample_idxs).
        So basically the latest one.
        
        -> This can also be conditional that they are in the condiitonal set (like future)
        
        Each action-metric only accumulates until its 'seen'-index.
        
        Afterwards the average over all is returned.
        
        """

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

        # TODO: Select current batch on 2 conditions,
        # For each batch we have to check:
        # 1) Which samples belong to action a_i? (for each a_i we count acc for the action separately)

        # 2) Is sample_idx < lates_action_idx? -> Easiest to check: Skip actions that have t_a > min(all_t's in batch)
        # Get top-acc results for Meter per action for the subset that satisfies.
        # -> ALWAYS SATISFIED FOR HISTORY STREAM OF DATA: all data will be action data until last time seen that action
        # This is because we measure acc separately per action.

        # first_sample_id = min(stream_sample_ids)
        # Filter out actions for which current min sample id (t) in batch > last action observation id (t_a)
        # seen_actions_sat = {a for a, t_a_list in self.seen_action_to_obs_ids.items()
        #                     if t_a_list[-1] > first_sample_id}

        # Iterate actions in batch, continue if not satisfying filter constraint
        obs_actions_batch = set()
        label_batch_axis = 0
        for idx, verbnoun_t in enumerate(torch.unbind(labels, dim=label_batch_axis)):
            action = tuple((verbnoun_t.tolist()))
            if action in obs_actions_batch:  # Already processed, so skip
                continue
            obs_actions_batch.add(action)

            # Process all samples in batch for this action at once
            label_mask = torch.cat([
                torch.BoolTensor([tuple((verbnoun_t.tolist())) == action])  # Is (verb,noun) tuple in cond_set?
                for verbnoun_t in torch.unbind(labels, dim=label_batch_axis)
            ], dim=label_batch_axis)

            # Subset
            subset_batch_size = sum(label_mask)
            preds1, preds2 = preds[0][label_mask], preds[1][label_mask]
            subset_labels = labels[label_mask]
            labels1, labels2 = subset_labels[:, 0], subset_labels[:, 1]

            action_topk_acc: float = metrics.distributed_twodistr_top1_errors(
                preds1, preds2, labels1, labels2, acc=True)

            self.action_to_avgmeter[action].update(action_topk_acc, weight=subset_batch_size)

    @torch.no_grad()
    def reset(self):
        """Reset the metric."""
        self.action_to_avgmeter = defaultdict(AverageMeter)

    @torch.no_grad()
    def result(self) -> Dict:
        """Get the metric(s) with name in dict format.
        Averages with equal weight over all actions with a meter (observed in history stream)."""

        # avg acc results over all actions, and make relative to previous acc (forgetting)
        avg_forg_over_actions = AverageMeter()
        for action, action_current_acc in self.action_to_avgmeter.items():
            current_acc = action_current_acc.avg
            forg = current_acc - self.action_to_prev_acc[action]
            avg_forg_over_actions.update(forg, weight=1)

        return {self.name: avg_forg_over_actions.avg}

    @torch.no_grad()
    def save_result_to_history(self):
        """Optional: after getting the result, we may also want to re-use this result later on."""
        # save results per action in the dict
        for action, action_current_acc in self.action_to_avgmeter.items():
            self.action_to_prev_acc[action] = action_current_acc

# class OnlineForgetting(Metric):
# class ReexposureForgetting(Metric):
# class OnlineForgetting(Metric):

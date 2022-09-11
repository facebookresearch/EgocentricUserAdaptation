import logging
from abc import ABC, abstractmethod
import torch
from typing import Dict, Set, Union, Tuple
from continual_ego4d.utils.meters import AverageMeter
from ego4d.evaluation import lta_metrics as metrics
from collections import defaultdict
from continual_ego4d.datasets.continual_action_recog_dataset import verbnoun_to_action
from continual_ego4d.metrics.metric import Metric, ACTION_MODES, get_metric_tag, TAG_PAST

import matplotlib.pyplot as plt

from ego4d.utils import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from continual_ego4d.tasks.continual_action_recog_task import StreamStateTracker
logger = logging.get_logger(__name__)


class ConditionalOnlineForgettingMetric(Metric):
    reset_before_batch = True  # Resets forgetting Avg metrics, but history is kept
    modes = ACTION_MODES
    base_metric_modes = ['loss', 'acc']

    plot_config = {
        "color": 'royalblue',
        "dpi": 600,
        "figsize": (6, 6),
    }

    def __init__(self, base_metric_mode: str, action_mode: str, main_metric_name: str,
                 is_current_batch_cond_set=False, in_cond_set=True,
                 k: int = None, loss_fun=None,
                 keep_action_results_over_time=False,
                 do_plot=True,
                 ):
        """
        Past data is iterated over AFTER update. This is the BEFORE update for the next batch.

        For AFTER update on current batch Bc, and next batch Bn:
        1) Bc:
        - evaluated on history stream and current batch: [0, c]
        We need to get performance for all actions in Bc, as they serve as reference for when they are encountered
        later in the stream.
        - We need revisit all history stream samples for the actions in Bc.

        2) Bn:
        - We need revisit all history stream samples for the actions in Bn.
          Evaluated on history stream [0, n[ (including current batch), BUT because Bc may
            i) include actions that are also in Bn: These have zero-delta anyway.
            ii) have no actions in Bn: Then for the pre-update results of actions in Bn, the data of Bc is irrelevant.
            Hence, we can evaluate on the same history stream: [0, c[
        - We need to get performance on all actions in Bn.
            This performance will be added as a tuple for an action a_i:
            (AFTER-update performance for a_i at iteration t, BEFORE-update performance for a_i at current iteration >t)
            The delta of this BEFORE-update performance and an earlier AFTER-update performance
            gives us an indication of forgetting in between two exposures.

        - Note: This result is the result reported at Bc (for Bn). Hence, there is an offset of 1 in the logging curves
        if the batch index would be used.

        For scalability: We can have per action/verb/noun a FIFO buffer of history indices for that action.
        This means that if we have the full map of <action,history_stream_idx-list>, we take the final [-M-1:-1] of the
        history_stream_idx as considered indices (this leaves out the final one that we used for updating).
        Hence on both models the same set is considered.

        In summary:
        - We measure for a given action on 2 models. Just after learning on the action, and the next time the action is
        encountered, before updating on it. The data we evaluate performance with is the latest M samples, including
        the sample that was just updated on. This gives a good indication of forgetting in between the 2 models.

        - We measure simultaneously before/after update performance, on history stream excluding current batch: [0, c[.
        - When sampling the history, we need to collect all indices in [0,c[ that contain actions in Bn and Bc.
        But only the latest M ones, to enable feasibility.


        Special cases:
        - For actions that are both in Bc and Bn, we report a delta of zero, as it concerns the same model for the same action.
        - For Replay: We only consider re-occurrence of the new stream batch, not replay samples.

        Notes:
        The metric stores the accuracies for previous actions on save_result_to_history.
        The reset only resets the running average meters, not the history.

        :param: current_batch_cond_set: Use current observed batch as conditional set for constraints?
        :param: in_cond_set: If current_batch_cond_set=True, should consider samples in or out this set?
        :param: keep_action_results_over_time: Store everytime a result is queried, for all actions separately.
        Including the iteration number of previous time result was queried and current iteration number.
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
        self.is_current_batch_cond_set: bool = is_current_batch_cond_set

        self.name = get_metric_tag(TAG_PAST, action_mode=self.action_mode, base_metric_name=basic_metric_name)

        # track results per action
        self.keep_action_results_over_time = keep_action_results_over_time
        self.action_results_over_time = {
            "prev_batch_idx_after_update": defaultdict(list),  # <action, list(<prev_iter,current_iter,delta_value>)
            "current_batch_idx_before_update": defaultdict(list),
            "delta": defaultdict(list),
        }

        # Only something to plot if keeping track of state
        self.do_plot = do_plot and self.keep_action_results_over_time

    @torch.no_grad()
    def update(self, past_preds, past_labels, stream_sample_idxs, stream_state: 'StreamStateTracker' = None, **kwargs):
        """
        ASSUMPTION: The preds,labels are all from the history stream only, forwarded on the CURRENT model.
        Update per-action accuracy metric from predictions and labels.
        """
        assert past_preds[0].shape[0] == past_labels.shape[0], f"Batch sizes not matching for past in stream!"
        assert stream_state is not None

        # Verb/noun errors
        cond_set = None
        if self.action_mode in ['verb', 'noun']:

            # Use current batch as conditional set
            if self.is_current_batch_cond_set:
                cond_set = set(stream_state.stream_batch_labels[:, self.label_idx].tolist())

            self._get_verbnoun_metric_result(past_preds, past_labels, cond_set)

        # Action errors
        elif self.action_mode in ['action']:

            # Use current batch as conditional set
            if self.is_current_batch_cond_set:
                cond_set = set()
                for (verb, noun) in stream_state.stream_batch_labels.tolist():
                    action = verbnoun_to_action(verb, noun)
                    cond_set.add(action)
            self._get_action_metric_result(past_preds, past_labels, cond_set)

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

            subset_preds = verbnoun_preds[label_mask]
            subset_labels = verbnoun_labels[label_mask]

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
    def result(self, current_batch_idx, **kwargs) -> Dict:
        """Get the metric(s) with name in dict format.
        Averages with equal weight over all actions with a meter.
        Stores per-action results and timestamps in action_results_over_time.
        """
        # avg acc results over all actions, and make relative to previous acc (forgetting)
        avg_forg_over_actions = AverageMeter()
        for action, action_current_acc in self.action_to_current_perf.items():
            if action not in self.action_to_prev_perf:  # First time the acc is measured on the model for this action
                logger.debug(f"{self.action_mode} {action} acc measured for first time in history stream.")
                continue
            current_acc = action_current_acc.avg
            prev_acc, prev_batch_idx = self.action_to_prev_perf[action]
            assert action_current_acc.count > 0, f"Acc {current_acc} has zero count."

            # Delta
            forg = self.delta_fun(current_acc, prev_acc)

            # Updates
            avg_forg_over_actions.update(forg, weight=1)

            if self.keep_action_results_over_time:
                self.action_results_over_time["prev_batch_idx"][action].append(prev_batch_idx)
                self.action_results_over_time["current_batch_idx"][action].append(current_batch_idx)
                self.action_results_over_time["delta"][action].append(forg)

        if avg_forg_over_actions.count == 0:
            return {}
        return {self.name: avg_forg_over_actions.avg}

    @torch.no_grad()
    def save_result_to_history(self, current_batch_idx, *args, **kwargs):
        """ Save acc on current model results per action in the dict."""
        for action, action_current_perf in self.action_to_current_perf.items():
            if action_current_perf.count == 0:
                continue
            self.action_to_prev_perf[action] = (action_current_perf.avg, current_batch_idx)

    @torch.no_grad()
    def dump(self) -> Dict:
        if not self.keep_action_results_over_time:
            return {}

        # Check equal lengths (same nb actions)
        single_entry_len = None
        for key, timelist in self.action_results_over_time.items():
            if single_entry_len is None:
                single_entry_len = len(timelist)
            else:
                assert len(timelist) == single_entry_len, \
                    f"{key} in action_results_over_time not populated with other keys"

        return {self.name: self.action_results_over_time}

    @torch.no_grad()
    def plot(self) -> Dict:
        if not self.do_plot or len(self.action_results_over_time["delta"]) == 0:  # Don't log if state is not kept
            return {}

        # Get deltas on x-axis
        deltas_x_per_action = defaultdict(list)
        for action, prev_res_over_time in self.action_results_over_time["prev_batch_idx"].items():
            cur_res_over_time = self.action_results_over_time["current_batch_idx"][action]
            for prev_t, new_t in zip(prev_res_over_time, cur_res_over_time):
                assert new_t > prev_t, f"New iteration {new_t} <= prev iteration {prev_t}"
                deltas_x_per_action[action].append(new_t - prev_t)

        # Get values on y-axis
        deltas_y_per_action = self.action_results_over_time["delta"]

        # Plot task-agnostic
        logger.debug(f"Plotting scatter: x={deltas_x_per_action}, y={deltas_y_per_action}")
        figure = plt.figure(figsize=self.plot_config['figsize'],
                            dpi=self.plot_config['dpi'])  # So all bars are visible!
        for action, deltas_x in deltas_x_per_action.items():
            deltas_y = deltas_y_per_action[action]
            plt.scatter(deltas_x, deltas_y, c=self.plot_config['color'])

        plt.ylim(None, None)
        plt.xlim(0, None)

        xlabel = 'nb_iters_between_exposures'
        ylabel = self.name
        title = f"{xlabel}_VS_{ylabel}"

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        return {self.name: figure}


# class FullOnlineForgettingAccMetric(ConditionalOnlineForgettingMetric):
#     """ Measure on history data over all actions that have already been observed."""
#
#     def __init__(self, k, action_mode, keep_action_results_over_time=False):
#         super().__init__(
#             base_metric_mode='acc', k=k, loss_fun=None,  # Acc
#             action_mode=action_mode, main_metric_name="FORG_full",
#             # current_batch_cond_set=False, in_cond_set=False  # Fixed
#             keep_action_results_over_time=keep_action_results_over_time,
#         )
#
#
# class FullOnlineForgettingLossMetric(ConditionalOnlineForgettingMetric):
#     """ Measure on history data over all actions that have already been observed."""
#
#     def __init__(self, loss_fun, action_mode, keep_action_results_over_time=False):
#         super().__init__(
#             base_metric_mode='loss', k=None, loss_fun=loss_fun,  # Acc
#             action_mode=action_mode, main_metric_name="FORG_full",
#             # current_batch_cond_set=False, in_cond_set=False  # Fixed
#             keep_action_results_over_time=keep_action_results_over_time,
#         )


class ReexposureForgettingAccMetric(ConditionalOnlineForgettingMetric):
    """ Measure on history data over actions that have already been observed AND are observed in current mini-batch."""

    def __init__(self, k, action_mode, keep_action_results_over_time=False):
        super().__init__(
            base_metric_mode='acc', k=k, loss_fun=None,  # Acc
            action_mode=action_mode, main_metric_name="FORG_EXPOSE",
            is_current_batch_cond_set=True, in_cond_set=True,
            keep_action_results_over_time=keep_action_results_over_time,
        )


class ReexposureForgettingLossMetric(ConditionalOnlineForgettingMetric):
    """ Measure on history data over actions that have already been observed AND are observed in current mini-batch."""

    def __init__(self, loss_fun, action_mode, keep_action_results_over_time=False):
        super().__init__(
            base_metric_mode='loss', k=None, loss_fun=loss_fun,  # Acc
            action_mode=action_mode, main_metric_name="FORG_EXPOSE",
            is_current_batch_cond_set=True, in_cond_set=True,
            keep_action_results_over_time=keep_action_results_over_time,
        )

# class CollateralForgettingAccMetric(ConditionalOnlineForgettingMetric):
#     """ Measure on history data over actions that have already been observed
#     AND are NOTobserved in current mini-batch."""
#
#     def __init__(self, k, action_mode, keep_action_results_over_time=False):
#         super().__init__(
#             base_metric_mode='acc', k=k, loss_fun=None,  # Acc
#             action_mode=action_mode, main_metric_name="FORG_COLLAT",
#             current_batch_cond_set=True, in_cond_set=False,
#             keep_action_results_over_time=keep_action_results_over_time,
#         )
#
#
# class CollateralForgettingLossMetric(ConditionalOnlineForgettingMetric):
#     """ Measure on history data over actions that have already been observed
#     AND are NOTobserved in current mini-batch."""
#
#     def __init__(self, loss_fun, action_mode, keep_action_results_over_time=False):
#         super().__init__(
#             base_metric_mode='loss', k=None, loss_fun=loss_fun,  # Acc
#             action_mode=action_mode, main_metric_name="FORG_COLLAT",
#             current_batch_cond_set=True, in_cond_set=False,
#             keep_action_results_over_time=keep_action_results_over_time,
#         )

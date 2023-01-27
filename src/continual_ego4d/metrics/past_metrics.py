import copy
import torch
from typing import Dict, Union
from continual_ego4d.metrics.meters import AverageMeter
from ego4d.evaluation import lta_metrics as metrics
from collections import defaultdict
from continual_ego4d.datasets.continual_action_recog_dataset import verbnoun_to_action
from continual_ego4d.metrics.metric import Metric, ACTION_MODES, get_metric_tag, TAG_PAST
from continual_ego4d.metrics.standard_metrics import OnlineLossMetric

import matplotlib.pyplot as plt

from ego4d.utils import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from continual_ego4d.tasks.continual_action_recog_task import StreamStateTracker
logger = logging.get_logger(__name__)


class ConditionalOnlineForgettingMetric(Metric):
    reset_before_batch = False
    modes = ACTION_MODES
    base_metric_modes = ['loss', 'acc']

    plot_config = {
        "color": 'royalblue',
        "dpi": 600,
        "figsize": (6, 6),
    }

    def __init__(self,
                 base_metric_mode: str, action_mode: str, main_metric_name: str,
                 k: int = None, loss_fun_unred=None,
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
        - For Replay: We only consider re-occurrence of the new stream batch, not replay_strategies samples.

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
        self.do_plot = do_plot

        # Base Metric-mode specific
        if self.base_metric_mode == 'loss':
            assert loss_fun_unred is not None, f"Need to pass loss_fun to metric if in loss mode."
            self.loss_fun = loss_fun_unred
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

        self.name = get_metric_tag(TAG_PAST, action_mode=self.action_mode, base_metric_name=basic_metric_name)

        # track results per action
        self.action_results_over_time = {
            "prev_after_update_iter": defaultdict(list),  # <action, list(<prev_iter,current_iter,delta_value>)
            "current_before_update_iter": defaultdict(list),
            "delta": defaultdict(list),
        }

        # Keep Forgetting states
        self.action_after_update_meter = defaultdict(AverageMeter)  # First reference value
        self.action_after_update_iter = {}  # Keep iteration number when last measured
        self.action_next_before_update_meter = defaultdict(AverageMeter)  # Next model to calc delta

        # Keep so we know when new batch is processed, triggers processing current batch predictions
        self.last_processed_batch_idx = -1

    @torch.no_grad()
    def update(self, past_preds_t, past_labels_t, past_sample_idxs, stream_state: 'StreamStateTracker' = None,
               is_current_stream_batch=False,
               **kwargs):
        """
        ASSUMPTION: The preds,labels are all from the history stream only, forwarded on the CURRENT model.
        Update per-action accuracy metric from predictions and labels.

        Current batch should ONLY be used for the AFTER-UPDATE, as the next BEFORE-UPDATE the current
        action will also be in the batch (but was not included for the previous after-update).
        Edge-case: when both are adjacent, then the current batch does concern both after/before update.
        To fix this, we just skip the metric calculation and add a zero entries for the adjacent actions.
        """
        assert past_preds_t[0].shape[0] == past_labels_t.shape[0], f"Batch sizes not matching for past in stream!"
        assert stream_state is not None

        # First process current batch as well (for after-update only)!

        if self.last_processed_batch_idx != stream_state.batch_idx:
            assert not is_current_stream_batch
            self.last_processed_batch_idx = stream_state.batch_idx

            # Reconstruct from stream-tracker
            verb_preds = torch.stack([
                stream_state.sample_idx_to_verb_pred[idx] for idx in stream_state.stream_batch_sample_idxs
            ]).to(stream_state.stream_batch_labels.device)
            noun_preds = torch.stack([
                stream_state.sample_idx_to_noun_pred[idx] for idx in stream_state.stream_batch_sample_idxs
            ]).to(stream_state.stream_batch_labels.device)

            logger.info("Updating CURRENT BATCH in PAST STREAM for Re-exposure performance")
            self.update(past_preds_t=[verb_preds, noun_preds],
                        past_labels_t=stream_state.stream_batch_labels,
                        past_sample_idxs=stream_state.stream_batch_sample_idxs,
                        stream_state=stream_state,
                        is_current_stream_batch=True
                        )

        # Action or verb/noun sets
        current_batch_labelset = set({
            verbnoun_to_action(verb, noun)
            for (verb, noun) in stream_state.stream_batch_labels.tolist()}
        )
        next_batch_labelset = stream_state.stream_next_batch_labelset

        # Subset for verb/noun
        if self.action_mode in ['verb', 'noun']:
            current_batch_labelset = set({x[self.label_idx] for x in current_batch_labelset})
            next_batch_labelset = set({x[self.label_idx] for x in next_batch_labelset})

        # For entries that are in both (comparing same data on same model): skip and add zero entry
        labels_in_both = current_batch_labelset.intersection(next_batch_labelset)
        if len(labels_in_both) > 0:
            for label in labels_in_both:
                assert self.action_after_update_meter[label].count == 0
                assert self.action_next_before_update_meter[label].count == 0
                self.action_after_update_iter[label] = stream_state.batch_idx

            current_batch_labelset = current_batch_labelset - labels_in_both
            next_batch_labelset = next_batch_labelset - labels_in_both

        # Update current batch:
        updated_labels = self._update_dict_action_selection(current_batch_labelset, past_preds_t, past_labels_t,
                                                            dict_to_update=self.action_after_update_meter)
        logger.debug(f"AFTER-UPDATE: [PAST_IDXS={past_sample_idxs}] "
                     f"current_batch_labelset={current_batch_labelset}\n"
                     f"past_labels_t={past_labels_t}\n"
                     f"UPDATED labels {updated_labels} in self.action_after_update_meter={self.action_after_update_meter}")

        # Update iteration references for later delta
        for updated_label in updated_labels:
            self.action_after_update_iter[updated_label] = stream_state.batch_idx

        # Next batch at t+1: Update
        # Skip to avoid including next-exposure action (which isn't included for the previous after-update perf
        if not is_current_stream_batch:
            updated_labels = self._update_dict_action_selection(next_batch_labelset, past_preds_t, past_labels_t,
                                                                dict_to_update=self.action_next_before_update_meter)
            logger.debug(f"BEFORE-UPDATE: [PAST_IDXS={past_sample_idxs}] "
                         f"next_batch_labelset={next_batch_labelset}\n"
                         f"past_labels_t={past_labels_t}\n"
                         f"UPDATED labels {updated_labels} in self.action_after_update_meter={self.action_next_before_update_meter}")

    def _update_dict_action_selection(self, selected_labels: set[tuple],
                                      past_preds, past_labels,
                                      dict_to_update: dict,
                                      ):
        """ Conditionally update dict_to_update based on past_labels being in current_batch_labelset."""

        updated_labels = []
        for label in selected_labels:
            # Get performance and add to after_update dict
            if self.action_mode in ['verb', 'noun']:
                result, subset_size = self._get_verbnoun_metric_result(label, past_preds, past_labels)

            elif self.action_mode in ['action']:
                result, subset_size = self._get_action_metric_result(label, past_preds, past_labels)
            else:
                raise ValueError()

            if result is not None:
                dict_to_update[label].update(result.item(), weight=subset_size)
                updated_labels.append(label)
        return updated_labels

    def _get_verbnoun_metric_result(self, select_verbnoun, preds, labels) -> (Union[torch.FloatTensor, None], int):
        # Iterate actions in batch, only return when found
        label_batch_axis = 0
        for idx, verbnoun_t in enumerate(torch.unbind(labels, dim=label_batch_axis)):
            verbnoun = verbnoun_t[self.label_idx].item()  # Verb or noun
            verbnoun_labels = labels[:, self.label_idx]
            verbnoun_preds = preds[self.label_idx]

            # Filter out based on selection
            if verbnoun != select_verbnoun:
                continue

            # Mask
            label_mask = (verbnoun_labels == verbnoun).bool()
            subset_batch_size = sum(label_mask).item()

            subset_preds = verbnoun_preds[label_mask]
            subset_labels = verbnoun_labels[label_mask]

            # Acc metric
            if self.base_metric_mode == 'acc':
                metric_result: torch.FloatTensor = metrics.distributed_topk_errors(
                    subset_preds, subset_labels, [self.k], return_mode='acc')[0]

            elif self.base_metric_mode == 'loss':
                metric_result: torch.FloatTensor = self.loss_fun(subset_preds, subset_labels).sum()

            return metric_result, subset_batch_size

        # None were found
        return None, 0

    def _get_action_metric_result(self, select_action, preds, labels) -> (Union[torch.FloatTensor, None], int):
        """ Filter preds and labels based on selected action. """
        label_batch_axis = 0

        # Process all samples in batch for this action at once
        label_mask = torch.cat([
            torch.BoolTensor([verbnoun_to_action(*verbnoun_t.tolist()) == select_action])
            for verbnoun_t in torch.unbind(labels, dim=label_batch_axis)
        ], dim=label_batch_axis)

        subset_batch_size = sum(label_mask).item()
        if subset_batch_size == 0:
            return None, 0

        # Subset
        preds1, preds2 = preds[0][label_mask], preds[1][label_mask]
        subset_labels = labels[label_mask]
        labels1, labels2 = subset_labels[:, 0], subset_labels[:, 1]

        if self.base_metric_mode == 'acc':
            metric_result: torch.FloatTensor = metrics.distributed_twodistr_top1_errors(
                preds1, preds2, labels1, labels2, return_mode='acc')

        elif self.base_metric_mode == 'loss':
            metric_result, *_ = OnlineLossMetric.get_losses_from_preds(
                [preds1, preds2], subset_labels, self.loss_fun, take_sum=True)

        return metric_result, subset_batch_size

    @torch.no_grad()
    def reset(self):
        """Reset the metric."""
        self.action_after_update_meter = defaultdict(AverageMeter)
        self.action_after_update_iter = {}
        self.action_next_before_update_meter = defaultdict(AverageMeter)

    @torch.no_grad()
    def result(self, current_batch_idx, **kwargs) -> Dict:
        """
        Report metrics in before_update mode (from next batch).
        Clear reported actions from states.
        """

        # avg acc results over all actions, and make relative to previous acc (forgetting)
        result_dict = copy.deepcopy(self.action_next_before_update_meter)  # Attr will be altered in-place
        for action, action_before_update_perf_meter in result_dict.items():

            # Check if first time the acc is measured on the model for this action
            # It will be ignored, and re-processed in the next after-update
            if action not in self.action_after_update_meter:
                logger.debug(f"{self.action_mode} {action} perf measured for first time in history stream.")
                del self.action_next_before_update_meter[action]
                continue

            # Current before-update perf
            action_before_update_perf = action_before_update_perf_meter.avg

            # Previous after-update perf and idx
            action_after_update_perf = self.action_after_update_meter[action].avg
            prev_batch_idx = self.action_after_update_iter[action]

            # Delta
            forg = self.delta_fun(action_before_update_perf, action_after_update_perf)

            if forg != 0:
                assert self.action_after_update_meter[action].count == action_before_update_perf_meter.count, \
                    f"Forgetting is {forg} for action {action}. " \
                    f"Count after update={self.action_after_update_meter[action].count}, " \
                    f"count before update={action_before_update_perf_meter.count}"

                assert prev_batch_idx < current_batch_idx, \
                    f"Forgetting should be zero but is {forg} for batch idxs " \
                    f"(prev={prev_batch_idx},cur={current_batch_idx}) on action {action}"

            # Add to collector
            self.action_results_over_time["prev_after_update_iter"][action].append(prev_batch_idx)
            self.action_results_over_time["current_before_update_iter"][action].append(current_batch_idx)
            self.action_results_over_time["delta"][action].append(forg)

            # Delete both delta values
            del self.action_after_update_meter[action]
            del self.action_after_update_iter[action]
            del self.action_next_before_update_meter[action]
        return {}

    @torch.no_grad()
    def dump(self) -> Dict:
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
        for action, prev_res_over_time in self.action_results_over_time["prev_after_update_iter"].items():
            cur_res_over_time = self.action_results_over_time["current_before_update_iter"][action]
            for prev_t, new_t in zip(prev_res_over_time, cur_res_over_time):
                assert new_t >= prev_t, f"New iteration {new_t} <= prev iteration {prev_t}"
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


class ReexposureForgettingAccMetric(ConditionalOnlineForgettingMetric):
    """ Measure on history data over actions that have already been observed AND are observed in current mini-batch."""

    def __init__(self, k, action_mode):
        super().__init__(
            base_metric_mode='acc', k=k, loss_fun_unred=None,  # Acc
            action_mode=action_mode, main_metric_name="FORG_EXPOSE",
        )


class ReexposureForgettingLossMetric(ConditionalOnlineForgettingMetric):
    """ Measure on history data over actions that have already been observed AND are observed in current mini-batch."""

    def __init__(self, loss_fun_unred, action_mode):
        super().__init__(
            base_metric_mode='loss', k=None, loss_fun_unred=loss_fun_unred,  # Acc
            action_mode=action_mode, main_metric_name="FORG_EXPOSE",
        )

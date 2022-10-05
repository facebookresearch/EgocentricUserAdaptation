import torch
from typing import Dict, Set, Union, Tuple
from continual_ego4d.utils.meters import AverageMeter
from continual_ego4d.metrics.metric import AvgMeterMetric, Metric, get_metric_tag
# from continual_ego4d.metrics.metric import AvgMeterDictMetric
from ego4d.evaluation import lta_metrics as metrics
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from continual_ego4d.tasks.continual_action_recog_task import StreamStateTracker


# class OnlineTopkAccBalancedMetric(AvgMeterDictMetric):
#     """Balances over all actions/verbs/nouns equally."""
#     reset_before_batch = True
#
#     def __init__(self, metric_tag: str, k: int = 1, mode="action"):
#         super().__init__(mode=mode)
#         self.k = k
#         self.name = get_metric_tag(main_parent_tag=metric_tag, action_mode=mode,
#                                    base_metric_name=f"top{self.k}_acc_balanced")
#
#         # Checks
#         if self.mode == 'action':
#             assert self.k == 1, f"Action mode only supports top1, not top-{self.k}"
#
#     @torch.no_grad()
#     def update(self, stream_current_batch_idx: int, preds, labels, *args, **kwargs):
#         """Update metric from predictions and labels."""
#         assert preds[0].shape[0] == labels.shape[0], f"Batch sizes not matching!"
#         batch_size = labels.shape[0]
#
#         # Verb/noun errors
#         if self.mode in ['verb', 'noun']:
#             topk_acc: torch.FloatTensor = metrics.distributed_topk_errors(
#                 preds[self.label_idx], labels[:, self.label_idx], [self.k], return_mode='acc'
#             )[0]  # Unpack self.k
#         elif self.mode in ['action']:
#             topk_acc: torch.FloatTensor = metrics.distributed_twodistr_top1_errors(
#                 preds[0], preds[1], labels[:, 0], labels[:, 1], return_mode='acc')
#
#         self.avg_meter.update(topk_acc.item(), weight=batch_size)


class OnlineTopkAccMetric(AvgMeterMetric):
    reset_before_batch = True

    def __init__(self, metric_tag: str, k: int = 1, mode="action"):
        super().__init__(mode=mode)
        self.k = k
        self.name = get_metric_tag(main_parent_tag=metric_tag, action_mode=mode, base_metric_name=f"top{self.k}_acc")
        self.avg_meter = AverageMeter()

        # Checks
        if self.mode == 'action':
            assert self.k == 1, f"Action mode only supports top1, not top-{self.k}"

    @torch.no_grad()
    def update(self, preds, labels, stream_sample_idxs, **kwargs):
        """Update metric from predictions and labels."""
        assert preds[0].shape[0] == labels.shape[0], f"Batch sizes not matching!"
        batch_size = labels.shape[0]

        # Verb/noun errors
        if self.mode in ['verb', 'noun']:
            topk_acc: torch.FloatTensor = metrics.distributed_topk_errors(
                preds[self.label_idx], labels[:, self.label_idx], [self.k], return_mode='acc'
            )[0]  # Unpack self.k
        elif self.mode in ['action']:
            topk_acc: torch.FloatTensor = metrics.distributed_twodistr_top1_errors(
                preds[0], preds[1], labels[:, 0], labels[:, 1], return_mode='acc')

        self.avg_meter.update(topk_acc.item(), weight=batch_size)


class RunningAvgOnlineTopkAccMetric(OnlineTopkAccMetric):
    reset_before_batch = False

    def __init__(self, metric_tag: str, k: int = 1, mode="action"):
        super().__init__(metric_tag, k, mode)
        self.name = f"{self.name}_running_avg"


class OnlineLossMetric(AvgMeterMetric):
    """
    The current batch loss is tracked anyway for training in the CL-Task.
    This Metric allows to track the loss on any other data rather than the current task (e.g. on data from the history)
    """
    reset_before_batch = True

    def __init__(self, metric_tag: str, loss_fun, mode="action"):
        super().__init__(mode=mode)
        self.loss_fun = loss_fun
        self.name = get_metric_tag(main_parent_tag=metric_tag, action_mode=mode, base_metric_name=f"loss")
        self.avg_meter = AverageMeter()

    @torch.no_grad()
    def update(self, preds, labels, stream_sample_idxs, **kwargs):
        """Update metric from predictions and labels."""
        assert preds[0].shape[0] == labels.shape[0], f"Batch sizes not matching!"
        batch_size = labels.shape[0]

        loss_action, loss_verb, loss_noun = self.get_losses_from_preds(preds, labels, self.loss_fun, mean=True)

        # Verb/noun errors
        if self.mode in ['verb']:
            loss = loss_verb
        elif self.mode in ['noun']:
            loss = loss_noun
        elif self.mode in ['action']:
            loss = loss_action
        else:
            raise ValueError()

        self.avg_meter.update(loss.item(), weight=batch_size)

    @staticmethod
    def get_losses_from_preds(preds: tuple[Tensor, Tensor], labels: Tensor, loss_fun, mean=False, take_sum=False):
        """Apply the loss function on the 2-headed classifier given model outputs and labels."""
        loss_verb = loss_fun(preds[0], labels[:, 0])  # Verbs
        loss_noun = loss_fun(preds[1], labels[:, 1])  # Nouns
        loss_action = loss_verb + loss_noun  # Avg losses

        assert not (take_sum and mean)
        if mean:
            return loss_action.mean(), loss_verb.mean(), loss_noun.mean()
        elif take_sum:
            return loss_action.sum(), loss_verb.sum(), loss_noun.sum()
        else:
            return loss_action, loss_verb, loss_noun


class RunningAvgOnlineLossMetric(OnlineLossMetric):
    reset_before_batch = False

    def __init__(self, metric_tag: str, loss_fun, mode="action"):
        super().__init__(metric_tag, loss_fun, mode)
        self.name = f"{self.name}_running_avg"

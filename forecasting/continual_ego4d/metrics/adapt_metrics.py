import logging
import torch
from typing import Dict, Set, Union, Tuple
from continual_ego4d.metrics.metric import AvgMeterMetric, get_metric_tag
from continual_ego4d.metrics.batch_metrics import TAG_BATCH

TAG_ADAPT = 'adapt'


class OnlineAdaptationGainMetric(AvgMeterMetric):
    """ Compare with pretrained model for verb, noun, or combined action loss what the delta is. """
    reset_before_batch = True

    def __init__(self, unreduced_loss_fun, sample_idx_to_pretrain_loss: dict, loss_mode="action",
                 main_metric_name="AG_online"):
        """
        :param unreduced_loss_fun:
        :param sample_idx_to_pretrain_loss:
        :param loss_mode: Which (sub)loss to consider for the gain calculation per instance.
        :param main_metric_name:
        """
        super().__init__(loss_mode)
        self.name = get_metric_tag(TAG_ADAPT, action_mode=self.mode, base_metric_name=main_metric_name)
        self.unreduced_loss_fun = unreduced_loss_fun
        self.sample_idx_to_pretrain_loss = sample_idx_to_pretrain_loss

    @torch.no_grad()
    def update(self, current_batch_idx: int, preds, labels, current_batch_sample_idxs: list):
        """Update metric from predictions and labels."""
        assert preds[0].shape[0] == labels.shape[0], f"Batch sizes not matching!"

        loss_verb = self.unreduced_loss_fun(preds[0], labels[:, 0])  # Verbs
        loss_noun = self.unreduced_loss_fun(preds[1], labels[:, 1])  # Nouns
        loss_action = loss_verb + loss_noun  # Avg losses

        # Verb/noun errors
        if self.mode in ['verb']:
            pretrain_key = get_metric_tag(TAG_BATCH, train_mode='pred', action_mode='verb', base_metric_name='loss')
            current_batch_loss = loss_verb

        elif self.mode in ['noun']:
            pretrain_key = get_metric_tag(TAG_BATCH, train_mode='pred', action_mode='noun', base_metric_name='loss')
            current_batch_loss = loss_noun

        elif self.mode in ['action']:
            pretrain_key = get_metric_tag(TAG_BATCH, train_mode='pred', action_mode='action', base_metric_name='loss')
            current_batch_loss = loss_action

        # Iterate samples
        for batch_idx, stream_sample_idx in enumerate(current_batch_sample_idxs):
            pretrain_sample_loss = self.sample_idx_to_pretrain_loss[stream_sample_idx][pretrain_key]
            current_sample_loss = current_batch_loss[batch_idx].item()
            adaptation_gain: float = pretrain_sample_loss - current_sample_loss
            self.avg_meter.update(adaptation_gain, weight=1)


class RunningAvgOnlineAdaptationGainMetric(OnlineAdaptationGainMetric):
    reset_before_batch = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, main_metric_name="AG_running_avg")


class CumulativeOnlineAdaptationGainMetric(OnlineAdaptationGainMetric):
    reset_before_batch = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, main_metric_name="AG_cumul")

    @torch.no_grad()
    def result(self, current_batch_idx: int, *args, **kwargs) -> Dict:
        """Get the metric(s) with name in dict format."""
        return {self.name: self.avg_meter.sum}

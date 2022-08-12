import logging
from abc import ABC, abstractmethod
import torch
from typing import Dict, Set, Union, Tuple
from continual_ego4d.utils.meters import AverageMeter
from ego4d.evaluation import lta_metrics as metrics
from continual_ego4d.metrics.batch_metrics import Metric


class OnlineAdaptationGainMetric(Metric):
    """ Compare with pretrained model for verb, noun, or combined action loss what the delta is. """
    reset_before_batch = True

    modes = ["verb", "noun", "action"]

    def __init__(self, unreduced_loss_fun, sample_idx_to_pretrain_loss: dict, mode="action", prefix="online"):
        self.unreduced_loss_fun = unreduced_loss_fun
        self.sample_idx_to_pretrain_loss = sample_idx_to_pretrain_loss
        self.name = f"{prefix}_{mode}_AG"
        self.avg_meter = AverageMeter()

        self.mode = mode
        assert self.mode in self.modes
        if self.mode == 'verb':
            self.label_idx = 0
        elif self.mode == 'noun':
            self.label_idx = 1
        elif self.mode == 'action':
            pass
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def update(self, preds, labels, current_batch_sample_idxs):
        """Update metric from predictions and labels."""
        assert preds[0].shape[0] == labels.shape[0], f"Batch sizes not matching!"
        batch_size = labels.shape[0]

        loss_verb = self.unreduced_loss_fun(preds[0], labels[:, 0])  # Verbs
        loss_noun = self.unreduced_loss_fun(preds[1], labels[:, 1])  # Nouns
        loss_action = loss_verb + loss_noun  # Avg losses

        # Verb/noun errors
        if self.mode in ['verb']:
            pretrain_key = "pred_verb_loss"
            current_loss = loss_verb
        elif self.mode in ['noun']:
            pretrain_key = "pred_noun_loss"
            current_loss = loss_noun
        elif self.mode in ['action']:
            pretrain_key = "pred_action_loss"
            current_loss = loss_action

        for batch_idx, stream_sample_idx in enumerate(current_batch_sample_idxs):
            pretrain_loss = self.sample_idx_to_pretrain_loss[stream_sample_idx][pretrain_key]
            adaptation_gain = pretrain_loss - current_loss
            self.avg_meter.update(adaptation_gain, weight=1)

    @torch.no_grad()
    def reset(self):
        """Reset the metric."""
        self.avg_meter.reset()

    @torch.no_grad()
    def result(self) -> Dict:
        """Get the metric(s) with name in dict format."""
        return {self.name: self.avg_meter.avg}


class RunningAvgOnlineAdaptationGainMetric(OnlineAdaptationGainMetric):
    reset_before_batch = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = f"running_avg_{self.name}"

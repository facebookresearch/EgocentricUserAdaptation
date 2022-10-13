import pandas as pd
import wandb
import pprint
import os
import torch
from collections import defaultdict
from continual_ego4d.utils.meters import AverageMeter

api = wandb.Api()


class ConditionalAverageMeterDict:
    def __init__(self, action_balanced=True):
        self.meter_dict = defaultdict(AverageMeter)
        self.action_balanced = action_balanced  # Or instance-balanced

    def reset(self):
        self.meter_dict = defaultdict(AverageMeter)

    def update(self, val_list: list, cond_list: list):
        assert len(val_list) == len(cond_list)

        for val, conditional in zip(val_list, cond_list):
            self.meter_dict[conditional].update(val, weight=1)

    def result(self):
        """ Return equally weighed average over conditionals, which are each averaged as well. """
        meter_total = AverageMeter()
        for meter_cond in self.meter_dict.values():
            weight = 1 if self.action_balanced else meter_cond.count  # Weigh equally or weigh by nb of samples
            meter_total.update(meter_cond.avg, weight=weight)
        return meter_total.avg


def loss_CE_to_likelihood(loss_t: torch.Tensor):
    """

    :param loss_t: tensor of shape (B,1) with B=batch size.
    :return:
    """
    return loss_t.mul(-1).exp()


def get_group_names_from_csv(selected_group_names_csv_path):
    """ Read from WandB downloaded CSV """
    selected_group_names_df = pd.read_csv(selected_group_names_csv_path)
    group_names = selected_group_names_df['Name'].to_list()
    return group_names


def get_group_run_iterator(project_name, group_name, finished_runs_only=True, run_filter=None):
    """ Only get user-runs that finished processing stream (finished_run=True)."""

    if run_filter is None:
        if finished_runs_only:
            run_filter = {
                "$and": [
                    {"group": group_name},
                    {"summary_metrics.finished_run": finished_runs_only}
                ]
            }
        else:
            run_filter = {"group": group_name}

    group_runs = api.runs(project_name, run_filter)
    return group_runs


def get_delta(delta_sign, user_val, pretrain_val):
    """ Calculate delta based on delta sign and user/pretrain values. """
    return delta_sign * (user_val - pretrain_val)


def get_delta_mappings():
    """ Get metric name to delta sign mapping. -1 for loss, +1 for acc."""
    loss_sign = -1
    acc_sign = 1

    loss_mapping = {
        name: loss_sign
        for name in [
            'train_action_batch/loss_running_avg',
            'train_verb_batch/loss_running_avg',
            'train_noun_batch/loss_running_avg',

            'test_action_batch/loss',
            'test_verb_batch/loss',
            'test_noun_batch/loss',

            'train_action_batch/balanced_loss',
            'train_verb_batch/balanced_loss',
            'train_noun_batch/balanced_loss',

            'train_action_batch/balanced_loss'
            'train_verb_batch/balanced_loss',
            'train_noun_batch/balanced_loss',
        ]
    }

    acc_mapping = {
        name: acc_sign
        for name in [
            'train_action_batch/top1_acc_running_avg',
            'train_verb_batch/top1_acc_running_avg',
            'train_noun_batch/top1_acc_running_avg',
            'train_verb_batch/top5_acc_running_avg',
            'train_noun_batch/top5_acc_running_avg',

            'test_action_batch/top1_acc',
            'test_verb_batch/top1_acc',
            'test_verb_batch/top5_acc',
            'test_noun_batch/top1_acc',
            'test_noun_batch/top5_acc',

            'train_action_batch/balanced_top1_acc',
            'train_verb_batch/balanced_top1_acc',
            'train_noun_batch/balanced_top1_acc',

            'train_action_batch/LL',
            'train_noun_batch/LL',
            'train_verb_batch/LL',

            'train_action_batch/balanced_LL',
            'train_verb_batch/balanced_LL',
            'train_noun_batch/balanced_LL',





        ]
    }

    delta_sign_map = {
        **loss_mapping, **acc_mapping
    }

    return delta_sign_map

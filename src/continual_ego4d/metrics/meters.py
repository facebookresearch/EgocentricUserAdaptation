# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
from typing import Union

import torch
import torch.distributed as dist


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Adapted from: https://github.com/pytorch/examples/blob/main/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, weight=1):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count if self.count > 0 else 0

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count])
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.__class__.__name__}: " + str(self.__dict__)


class ConditionalAverageMeterDict:
    def __init__(self, action_balanced=True):
        self.meter_dict = defaultdict(AverageMeter)
        self.action_balanced = action_balanced  # Or instance-balanced

    def reset(self):
        self.meter_dict = defaultdict(AverageMeter)

    def update(self, val_list: list, cond_list: list[Union[int, tuple, str]]):
        if not isinstance(cond_list, (list, tuple)):  # If single element
            cond_list = [cond_list]
        assert isinstance(cond_list[0], (int, tuple, str)), f"Type {type(cond_list[0])} is not hashable!"

        if isinstance(val_list, torch.Tensor):
            val_list = val_list.squeeze()
            if len(val_list.shape) == 0:
                val_list = val_list.unsqueeze(dim=0)
            assert len(val_list.shape) == 1, "Can only use 1-dim tensors"
        elif not isinstance(val_list, (list, tuple)):  # If single element
            val_list = [val_list]

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

    def __len__(self):
        return len(self.meter_dict)

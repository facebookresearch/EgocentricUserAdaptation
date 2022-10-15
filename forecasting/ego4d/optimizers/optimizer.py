#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Optimizer."""

import torch

from . import lr_policy
from ego4d.utils import logging

logger = logging.get_logger(__name__)


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """

    bn_params = []  # Batchnorm parameters.
    feat_params_non_bn = []  # Non-batchnorm parameters.
    classifier_params = []  # Separate classifier
    for name, p in model.named_parameters():
        if "bn" in name:
            bn_params.append(p)
        elif "head" in name:
            classifier_params.append(p)
        else:
            feat_params_non_bn.append(p)

    # Apply different weight decay to Batchnorm and non-batchnorm parameters.
    # In Caffe2 classification codebase the weight decay for batchnorm is 0.0.
    # Having a different weight decay on batchnorm might cause a performance
    # drop.

    classifier_lr = cfg.SOLVER.CLASSIFIER_LR if cfg.SOLVER.CLASSIFIER_LR is not None else cfg.SOLVER.BASE_LR
    classifier_momentum = cfg.SOLVER.MOMENTUM if cfg.SOLVER.MOMENTUM_HEAD is None else cfg.SOLVER.MOMENTUM_HEAD
    feat_momentum = cfg.SOLVER.MOMENTUM if cfg.SOLVER.MOMENTUM_FEAT is None else cfg.SOLVER.MOMENTUM_FEAT

    optim_params = [

        # FEATS:BN
        {"params": bn_params,
         "weight_decay": cfg.BN.WEIGHT_DECAY,
         "lr": cfg.SOLVER.BASE_LR,
         "momentum": feat_momentum
         },

        # FEATS: non-BN
        {"params": feat_params_non_bn,
         "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
         "lr": cfg.SOLVER.BASE_LR,
         "momentum": feat_momentum,
         },

        # CLASSIFIER
        {"params": classifier_params,
         "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
         "lr": classifier_lr,
         "momentum": classifier_momentum
         },
    ]
    logger.info(f"Classifier LR={classifier_lr}, base LR={cfg.SOLVER.BASE_LR}")

    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == len(feat_params_non_bn) + len(bn_params) + len(classifier_params), \
        "parameter size does not match: {} + {} + {} != {}".format(
            len(feat_params_non_bn), len(bn_params), len(classifier_params), len(list(model.parameters()))
        )

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        settings_dict = {
            # "momentum": cfg.SOLVER.MOMENTUM,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            "dampening": cfg.SOLVER.DAMPENING,
            "nesterov": cfg.SOLVER.NESTEROV if cfg.SOLVER.MOMENTUM > 0 else False,
        }
        logger.info(f"Init SGD with params: {settings_dict}")
        return torch.optim.SGD(
            optim_params,
            **settings_dict
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        settings_dict = {
            # "lr": cfg.SOLVER.BASE_LR,
            "betas": (0.9, 0.999),
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
        }
        logger.info(f"Init Adam with params: {settings_dict}")

        return torch.optim.Adam(
            optim_params,
            **settings_dict
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decays.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

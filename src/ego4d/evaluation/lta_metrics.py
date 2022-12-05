#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""

import numpy as np
import torch

import editdistance

from ..utils import distributed as du
from ..utils import logging

from sklearn.metrics import average_precision_score

logger = logging.get_logger(__name__)

topk_return_modes = ['acc', 'err', 'correct_cnt']
def distributed_twodistr_top1_errors(preds1, preds2, labels1, labels2, return_mode='err') -> torch.FloatTensor:
    """
    Prediction from both distributions (verb,noun) has to be correct for a correct (action) prediction.
    Only takes top1 as top-K with K>1 results in combinatorial solutions.
    (e.g. is top-2 the second best of first or second distr?)"""
    assert preds1.shape[0] == preds2.shape[0] == labels1.shape[0] == labels2.shape[0]
    batch_size = preds1.shape[0]
    k = 1

    # (1 x batch_size) indicator matrix
    top1_correct1 = _get_topk_correct_onehot_matrix(preds1, labels1, ks=[k])
    top1_correct2 = _get_topk_correct_onehot_matrix(preds2, labels2, ks=[k])

    # Take AND operation on indicator matrix
    top1_correct_both = torch.logical_and(top1_correct1, top1_correct2)

    # Count corrects over batch
    top1_correct_count = top1_correct_both.reshape(-1).float().sum()

    format_fn = get_return_format(return_mode, batch_size)
    return format_fn(top1_correct_count)


def distributed_topk_errors(preds, labels, ks, return_mode='err') -> list[torch.FloatTensor]:
    """
    Computes the top-k error for each k. Average reduces the result with all other
    distributed processes.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    assert return_mode in topk_return_modes

    preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
    labels = torch.cat(du.all_gather_unaligned(labels), dim=0)
    errors = topk_errors(preds, labels, ks, return_mode=return_mode)
    return errors


def topks_correct(preds, labels, ks) -> list[torch.FloatTensor]:
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list[torch.FloatTensor]): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    top_max_k_correct = _get_topk_correct_onehot_matrix(preds, labels, ks)
    # Compute the number of topk correct predictions for each k. (= sums over batch dim)
    topks_correct = [top_max_k_correct[:k, :].reshape(-1).float().sum() for k in ks]
    return topks_correct


def _get_topk_correct_onehot_matrix(preds, labels, ks):
    """
    Returns (max_k, batch_size) indicator matrix, from which each batch-dim (a single sample) indicates from which
    k the prediction is correct, this will be for only one k, or None.
    """
    assert preds.size(0) == labels.size(0), \
        "Batch dim of predictions and labels must match"

    # Find the top max_k predictions for each sample
    maxk = max(ks)
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, maxk, dim=1, largest=True, sorted=True
    )

    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)

    return top_max_k_correct


def get_return_format(return_mode, total_cnt=None):
    if return_mode == 'err':
        format_fn = lambda correct_cnt: (1.0 - correct_cnt / total_cnt) * 100.0
    elif return_mode == 'acc':
        format_fn = lambda correct_cnt: (correct_cnt / total_cnt) * 100.0
    elif return_mode == 'correct_cnt':
        format_fn = lambda correct_cnt: correct_cnt
    else:
        raise ValueError(f"return_mode='{return_mode}' unknown")
    return format_fn


def topk_errors(preds, labels, ks, return_mode='err') -> list[torch.FloatTensor]:
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct: list[torch.FloatTensor] = topks_correct(preds, labels, ks)
    format_fn = get_return_format(return_mode, preds.size(0))
    return [format_fn(x) for x in num_topks_correct]


def edit_distance(preds, labels):
    """
    Damerau–Levenshtein edit distance from: https://github.com/gfairchild/pyxDamerauLevenshtein
    Lowest among K predictions
    """
    N, Z, K = preds.shape
    dists = []
    for n in range(N):
        dist = min([editdistance.eval(preds[n, :, k], labels[n]) / Z for k in range(K)])
        dists.append(dist)
    return np.mean(dists)


def distributed_edit_distance(preds, labels):
    preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
    labels = torch.cat(du.all_gather_unaligned(labels), dim=0)
    return edit_distance(preds, labels)


def AUED(preds, labels):
    N, Z, K = preds.shape
    preds = preds.numpy()  # (N, Z, K)
    labels = labels.squeeze(-1).numpy()  # (N, Z)
    ED = np.vstack(
        [edit_distance(preds[:, :z], labels[:, :z]) for z in range(1, Z + 1)]
    )
    AUED = np.trapz(y=ED, axis=0) / (Z - 1)

    output = {"AUED": AUED}
    output.update({f"ED_{z}": ED[z] for z in range(Z)})
    return output


def distributed_AUED(preds, labels):
    preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
    labels = torch.cat(du.all_gather_unaligned(labels), dim=0)
    return AUED(preds, labels)

from typing import Union

import torch

from continual_ego4d.datasets.continual_action_recog_dataset import label_tensor_to_list
from continual_ego4d.metrics.meters import ConditionalAverageMeterDict
from ego4d.evaluation.lta_metrics import _get_topk_correct_onehot_matrix
from continual_ego4d.metrics.metric import ACTION_MODES


def get_micro_macro_avg_acc(
        action_mode,
        predictions: tuple[torch.Tensor, torch.Tensor],
        action_labels_t: torch.Tensor, k=1,
        macro_avg=True,
        return_per_sample_result=False
) -> Union[float, torch.Tensor]:
    assert action_mode in ACTION_MODES
    if action_mode == 'action':
        assert k == 1, "Action only defined for k=1"

    verb_preds_t, noun_preds_t = predictions
    verb_labels_t = action_labels_t[:, 0]
    noun_labels_t = action_labels_t[:, 1]

    # (k x batch_size) indicator matrix
    verb_corrects_t = _get_topk_correct_onehot_matrix(verb_preds_t, verb_labels_t, ks=[k])
    noun_corrects_t = _get_topk_correct_onehot_matrix(noun_preds_t, noun_labels_t, ks=[k])

    # Select which corrects-tensor to use
    action_labels = label_tensor_to_list(action_labels_t)
    if action_mode == 'action':
        selected_labels = action_labels

        # Take AND operation on indicator matrix
        action_corrects_t = torch.logical_and(verb_corrects_t, noun_corrects_t)
        corrects_t = action_corrects_t

    elif action_mode == 'verb':
        selected_labels = [x[0] for x in action_labels]
        corrects_t = verb_corrects_t

    elif action_mode == 'noun':
        selected_labels = [x[1] for x in action_labels]
        corrects_t = noun_corrects_t

    else:
        raise ValueError()

    # Squeeze
    corrects_t = corrects_t.squeeze()

    if return_per_sample_result:
        return corrects_t

    # Use corrects-tensor, and label list
    if macro_avg:
        result = per_sample_metric_to_macro_avg(corrects_t.tolist(), selected_labels)  # Get total weighed avg LL
    else:
        result = corrects_t.mean().item()

    result *= 100
    return result


def per_sample_metric_to_macro_avg(sample_values, sample_cond_list: list[Union[int, tuple, str]]):
    """Balanced = Macro avg, based on sample_cond_list. """
    cond_meter = ConditionalAverageMeterDict(action_balanced=True)  # Return micro or macro avg result
    cond_meter.update(sample_values, cond_list=sample_cond_list)
    return cond_meter.result()

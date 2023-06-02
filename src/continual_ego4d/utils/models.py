# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn

from ego4d.utils import logging

# Avoid circular dependencies (not imported at runtime, but typechecking does enable defining the type)
from typing import TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from continual_ego4d.tasks.continual_action_recog_task import StreamStateTracker

logger = logging.get_logger(__name__)


def get_name_to_grad_dict(model: torch.nn.Module):
    """
    Return full model dict with mapping to grads that are not None.
    """
    ret = {}
    for param_idx, (param_name, param) in enumerate(model.named_parameters()):
        if param.grad is not None:
            ret[param_name] = param.grad.clone().view(-1)

    return ret


def grad_dict_to_vector(grad_dict: dict[torch.Tensor], name_filters: list[str] = None, include_filter=True, verbose=True):
    """
    [x[0] for x in list(model.named_parameters())]
    :param name_filters: list of strings. If not None, only select grads with names containing any of the strings.
    :return:
    """
    # Consistent ordering
    grad_names = []

    def includes_any_in_filter(grad_name, name_filters):
        for name_filter in name_filters:
            if name_filter in grad_name:
                # logger.debug(f"Gradient for include filters {name_filters} has match: {grad_name}")
                return True
        return False

    def excludes_all_in_filter(grad_name, name_filters):
        for name_filter in name_filters:
            if name_filter in grad_name:
                return False
        # logger.debug(f"Gradient for exclude filters {name_filters} has match: {grad_name}")
        return True

    # Check if satisfies filter
    for grad_name in grad_dict.keys():

        if name_filters is not None:
            if include_filter and not includes_any_in_filter(grad_name, name_filters):
                continue
            elif not include_filter and not excludes_all_in_filter(grad_name, name_filters):
                continue

        grad_names.append(grad_name)

    # Make deterministic order
    grad_names = sorted(grad_names)
    grad_dims = [grad_dict[grad_name].numel() for grad_name in grad_names]  # Make vector

    flat_grad = None  # Init with model params device
    offset_idx = 0
    for grad_name in grad_names:
        param_grad = grad_dict[grad_name]

        if flat_grad is None:
            flat_grad = torch.zeros(sum(grad_dims), device=param_grad.device)

        if param_grad is not None:
            param_size = param_grad.numel()
            flat_grad[offset_idx: offset_idx + param_size].copy_(param_grad.clone().view(-1))
            offset_idx += param_size

            if verbose:
                logger.debug(f"GRADIENT SUMMARY: include_filters={include_filter}, name_filters={name_filters}. "
                             f"grad_name = '{grad_name}' contains {param_size} dim grad.")

    return flat_grad


def get_flat_gradient_vector(model: torch.nn.Module, name_filters: list[str] = None):
    """
    [x[0] for x in list(model.named_parameters())]
    :param model:
    :param name_filters: list of strings. If not None, only select grads with names containing any of the strings.
    :return:
    """
    grad_dims = [p.numel() for p in model.parameters()]

    flat_grad = None  # Init with model params device
    offset_idx = 0
    for param_idx, (param_name, param) in enumerate(model.named_parameters()):

        # Check if satisfies filter
        if name_filters is not None:

            name_matches_any_filter = False
            for name_filter in name_filters:
                if name_filter in param_name:
                    logger.debug(f"Gradient for filters {name_filters} has match: {param_name}")
                    name_matches_any_filter = True
                    continue

            if not name_matches_any_filter:
                continue

        if flat_grad is None:
            flat_grad = torch.zeros(sum(grad_dims), device=param.device)

        if param.grad is not None:
            param_size = param.numel()
            assert grad_dims[param_idx] == param_size

            flat_grad[offset_idx: offset_idx + param_size].copy_(param.grad.clone().view(-1))

            offset_idx += param_size

    return flat_grad


def reset_optimizer_stats_(optimizer: torch.optim.Optimizer):
    """ Reset stats of optimizer, such as momentum.

    State keeps per parameter a dict of state-pairs, such as 'momentum_buffer' for SGD.
    The state is independent of the number of parameter groups, hence resetting state resets for all groups.

    e.g. for SGD: https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
    State: dict(<param, {'momentum_buffer': <value> })>)
    """
    optimizer.__setstate__({'state': defaultdict(dict)})
    logger.info("Optimizer state is reset.")


def freeze_full_model(model):
    """ Freeze all parameters in model. """
    for param in model.parameters():
        param.requires_grad = False


def freeze_backbone_not_head(model):
    """ Freeze all params except the head. """
    for param in model.parameters():
        param.requires_grad = False

    # Never freeze head.
    for param in model.head.parameters():
        param.requires_grad = True

def freeze_head(model):
    """ Freeze all params in head. """
    for param in model.head.parameters():
        param.requires_grad = False


def model_trainable_summary(model):
    """Summarize trainable params over full model and head only. """
    full_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    full_total_p = sum(p.numel() for p in model.parameters())
    full_perc = "{:.1f}".format(full_train_p / full_total_p * 100)

    head_train_p = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    head_total_p = sum(p.numel() for p in model.head.parameters())
    head_perc = "{:.1f}".format(full_train_p / full_total_p * 100)

    excl_head_train_p = full_train_p - head_train_p
    excl_head_total_p = full_total_p - head_total_p
    excl_head_perc = "{:.1f}".format(excl_head_train_p / excl_head_total_p * 100)

    logger.info(f"[FROZEN backbone] Trainable parameters: FULL={full_train_p}/{full_total_p} ({full_perc}), "
                f"EXCL-HEAD={excl_head_train_p}/{excl_head_total_p} ({excl_head_perc})"
                f"HEAD-ONLY={head_train_p}/{head_total_p} ({head_perc})")

    return (full_train_p, full_total_p), (head_train_p, head_total_p)


class UnseenVerbNounMaskerHead(nn.Module):
    def __init__(self, stream_state: 'StreamStateTracker'):
        super().__init__()
        self.stream_state = stream_state
        self.mask_val = -1e12  # Extremely negative value

    def forward(self, verbnoun_logits):
        """ Make sure that current_batch before forwarding the verbs and nouns are in the seen set. """

        # Verb
        verb_x = verbnoun_logits[0]
        seen_sets_verb = {  # Include both past seen and current seen
            x for subset in [
                self.stream_state.pretrain_verb_freq_dict,
                self.stream_state.stream_seen_verb_freq_dict,
                self.stream_state.batch_verb_freq_dict]
            for x in subset
        }
        masked_verb_x, nb_masked = self._mask_unseen(verb_x, seen_sets_verb)
        logger.info(f"Masked out {nb_masked} verb logits")

        # Noun
        noun_x = verbnoun_logits[1]
        seen_sets_noun = {  # Include both past seen and current seen
            x for subset in [
                self.stream_state.pretrain_noun_freq_dict,
                self.stream_state.stream_seen_noun_freq_dict,
                self.stream_state.batch_noun_freq_dict]
            for x in subset
        }
        masked_noun_x, nb_masked = self._mask_unseen(noun_x, seen_sets_noun)
        logger.info(f"Masked out {nb_masked} noun logits")

        return [masked_verb_x, masked_noun_x]

    def _mask_unseen(self, out, passthrough_set: set):
        """
        :param out: Tensor of logits in shape <batch_size, pred_head_size>
        :param passthrough_set:
        :return:
        """
        pred_size = out.shape[1]
        maskout_set = [idx for idx in range(pred_size) if idx not in passthrough_set]
        out[:, maskout_set] = self.mask_val
        return out, len(maskout_set)

    # def __repr__(self):
    #     """Keeps reference to task, hence __repr__ of CL task results in recursion overflow. """
    #     return f"{self.__class__.__name__}-Wrapper, mask_val={self.mask_val}"

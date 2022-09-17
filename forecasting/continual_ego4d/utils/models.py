import torch
import torch.nn as nn

from ego4d.utils import logging

# Avoid circular dependencies (not imported at runtime, but typechecking does enable defining the type)
from typing import TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from continual_ego4d.tasks.continual_action_recog_task import StreamStateTracker

logger = logging.get_logger(__name__)


def reset_optimizer_stats_(optimizer):
    """ Reset stats of optimizer, such as momentum. """
    optimizer.__setstate__({'state': defaultdict(dict)})
    logger.info("Optimizer state is reset.")


def freeze_backbone_not_head(model):
    """ Freeze all params except the head. """
    for param in model.parameters():
        param.requires_grad = False

    # Never freeze head.
    for param in model.head.parameters():
        param.requires_grad = True


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

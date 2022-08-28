import torch
import torch.nn as nn

from ego4d.utils import logging

# Avoid circular dependencies (not imported at runtime, but typechecking does enable defining the type)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from continual_ego4d.tasks.continual_action_recog_task import StreamStateTracker

logger = logging.get_logger(__name__)


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
            x for subset in [self.stream_state.stream_seen_verb_set, self.stream_state.batch_verb_set] for x in subset
        }
        masked_verb_x, nb_masked = self._mask_unseen(verb_x, seen_sets_verb)
        logger.info(f"Masked out {nb_masked} verb logits")

        # Noun
        noun_x = verbnoun_logits[1]
        seen_sets_noun = {  # Include both past seen and current seen
            x for subset in [self.stream_state.stream_seen_noun_set, self.stream_state.batch_noun_set] for x in subset
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

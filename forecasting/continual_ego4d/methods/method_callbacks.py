from abc import ABC, abstractmethod
from continual_ego4d.methods.build import build_method, METHOD_REGISTRY
from typing import Dict, Tuple, List, Any
from torch import Tensor
from pytorch_lightning.core import LightningModule
import torch
from collections import defaultdict


class Method:
    def __init__(self, cfg, lightning_module: LightningModule):
        self.cfg = cfg  # For method-specific params
        self.lightning_module = lightning_module
        self.device = self.lightning_module.device
        self.trainer = self.lightning_module.trainer

        self.train_result_prefix = 'train'
        self.pred_result_prefix = 'pred'

        self.loss_fun_train = self.lightning_module.loss_fun
        self.loss_fun_pred = self.lightning_module.loss_fun_pred

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch

    def training_step(self, inputs, labels, *args, **kwargs) -> \
            Tuple[Tensor, List[Tensor], Dict]:
        """ Training step for the method when observing a new batch.
        Return Loss,  prediction outputs,a nd dictionary of result metrics to log."""

        preds: list = self.lightning_module.forward(inputs)
        loss1 = self.loss_fun_train(preds[0], labels[:, 0])  # Verbs
        loss2 = self.loss_fun_train(preds[1], labels[:, 1])  # Nouns
        loss = loss1 + loss2  # Avg losses

        log_results = {
            f"{self.train_result_prefix}_action_loss": loss.item(),
            f"{self.train_result_prefix}_verb_loss": loss1.item(),
            f"{self.train_result_prefix}_noun_loss": loss2.item(),
        }
        return loss, preds, log_results

    def prediction_step(self, inputs, labels, stream_sample_idxs: list, *args, **kwargs) \
            -> Tuple[Tensor, List[Tensor], Dict]:
        """ Default: Get all info we also get during training."""
        preds: list = self.lightning_module.forward(inputs)
        loss_verb = self.loss_fun_pred(preds[0], labels[:, 0])  # Verbs
        loss_noun = self.loss_fun_pred(preds[1], labels[:, 1])  # Nouns
        loss_action = loss_verb + loss_noun  # Avg losses

        sample_to_results = {}
        for batch_idx, stream_sample_idx in enumerate(stream_sample_idxs):
            sample_to_results[stream_sample_idx] = {
                f"{self.pred_result_prefix}_action_loss": loss_action[batch_idx].item(),
                f"{self.pred_result_prefix}_verb_loss": loss_verb[batch_idx].item(),
                f"{self.pred_result_prefix}_noun_loss": loss_noun[batch_idx].item(),
            }

        return loss_action, preds, sample_to_results


@METHOD_REGISTRY.register()
class Finetuning(Method):
    def __init__(self, cfg, lightning_module):
        super().__init__(cfg, lightning_module)

    def training_step(self, inputs, labels, *args, **kwargs):
        return super().training_step(inputs, labels, *args, **kwargs)


@METHOD_REGISTRY.register()
class Replay(Method):
    """
    Pytorch Subset of original stream, with expanding indices.
    """

    def __init__(self, cfg, lightning_module):
        super().__init__(cfg, lightning_module)
        self.mem_size = cfg.METHOD.REPLAY.MEMORY_SIZE_SAMPLES
        self.is_action_balanced = cfg.METHOD.REPLAY.IS_ACTION_BALANCED

        self.train_stream_dataset = lightning_module.train_dataloader().dataset


        self.conditional_memory = defaultdict(set)
        self.memory_dataloader_idxs = []  # Use this to update the memory indices of the stream

        # Retrieval state vars
        self.new_batch_size = None  # How many from stream

    def on_before_batch_transfer(self, new_batch: Any, dataloader_idx: int) -> Any:
        self.new_batch_size = new_batch.shape[0]

        # Retrieve from memory
        mem_batch = self.retrieve_batch_from_mem()

        # join mem and new
        joined_batch = torch.cat([new_batch, mem_batch], dim=0)  # Add in batch dim

        return joined_batch

    def retrieve_batch_from_mem(self) -> Tensor:
        """ """
        # TODO add replay samples to batch
        # Create new Subset and dataloader to fetch batch

        # Sample idcs without resampling over total buffersize, with each class weighted by its len

        raise NotImplementedError()

    def training_step(self, inputs, labels, current_batch_sample_idxs=None, *args, **kwargs) \
            -> Tuple[Tensor, List[Tensor], Dict]:
        """ Training step for the method when observing a new batch.
        Return Loss,  prediction outputs,a nd dictionary of result metrics to log."""
        total_batch_size = inputs.shape[0]

        # Forward at once
        preds: list = self.lightning_module.forward(inputs)

        # Unreduced losses
        loss_verbs = self.loss_fun_pred(preds[0], labels[:, 0])  # Verbs
        loss_nouns = self.loss_fun_pred(preds[1], labels[:, 1])  # Nouns

        # Disentangle new, buffer, and total losses
        loss_new_verbs = torch.mean(loss_verbs[:self.new_batch_size])
        loss_new_nouns = torch.mean(loss_nouns[:self.new_batch_size])
        loss_new_actions = loss_new_verbs + loss_new_nouns

        if total_batch_size > self.new_batch_size:  # Memory samples added
            loss_mem_verbs = torch.mean(loss_verbs[self.new_batch_size:])
            loss_mem_nouns = torch.mean(loss_nouns[self.new_batch_size:])
            loss_mem_actions = loss_mem_verbs + loss_mem_nouns

        else:
            loss_mem_verbs = loss_mem_nouns = loss_mem_actions = torch.FloatTensor([0]).to(self.device)

        loss_total_verbs = (loss_new_verbs + loss_mem_verbs) / 2
        loss_total_nouns = (loss_new_nouns + loss_mem_nouns) / 2
        loss_total_actions = (loss_new_actions + loss_mem_actions) / 2

        log_results = {
            # Total
            f"{self.train_result_prefix}_total_verb_loss": loss_total_verbs.item(),
            f"{self.train_result_prefix}_total_noun_loss": loss_total_nouns.item(),
            f"{self.train_result_prefix}_total_action_loss": loss_total_actions.item(),

            # Mem
            f"{self.train_result_prefix}_mem_verb_loss": loss_mem_verbs.item(),
            f"{self.train_result_prefix}_mem_noun_loss": loss_mem_nouns.item(),
            f"{self.train_result_prefix}_mem_action_loss": loss_mem_actions.item(),

            # New
            f"{self.train_result_prefix}_new_verb_loss": loss_new_verbs.item(),
            f"{self.train_result_prefix}_new_noun_loss": loss_new_nouns.item(),
            f"{self.train_result_prefix}_new_action_loss": loss_new_actions.item(),
        }

        # TODO: Store samples
        self._store_samples_in_replay_memory()

        return loss_total_actions, preds, log_results

    def _store_samples_in_replay_memory(self):
        """"""
        # Based on action labels create new bins and cutoff others, e.g. reservoir sampling

        # Do 2 versions: 1 conditional (set per class) , 2 unconditional( Only 1 set)
        self.is_action_balanced
        raise NotImplementedError()

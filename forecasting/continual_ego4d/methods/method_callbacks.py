from abc import ABC, abstractmethod
from continual_ego4d.methods.build import build_method, METHOD_REGISTRY
from typing import Dict, Tuple, List, Any
from torch import Tensor
from pytorch_lightning.core import LightningModule
import torch
from collections import defaultdict
from continual_ego4d.metrics.metric import get_metric_tag
from continual_ego4d.metrics.batch_metrics import TAG_BATCH
from collections import OrderedDict
import random
import itertools

from ego4d.utils import logging

logger = logging.get_logger(__name__)


class Method:
    def __init__(self, cfg, lightning_module: LightningModule):
        self.cfg = cfg  # For method-specific params
        self.lightning_module = lightning_module
        self.device = self.lightning_module.device
        self.trainer = self.lightning_module.trainer

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
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='action', base_metric_name='loss'): loss.item(),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='verb', base_metric_name='loss'): loss1.item(),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='noun', base_metric_name='loss'): loss2.item(),
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
                get_metric_tag(TAG_BATCH, train_mode='pred', action_mode='action', base_metric_name='loss'):
                    loss_action[batch_idx].item(),
                get_metric_tag(TAG_BATCH, train_mode='pred', action_mode='verb', base_metric_name='loss'):
                    loss_verb[batch_idx].item(),
                get_metric_tag(TAG_BATCH, train_mode='pred', action_mode='noun', base_metric_name='loss'):
                    loss_noun[batch_idx].item(),
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

    storage_policies = ['reservoir_stream', 'reservoir_action', 'reservoir_verbnoun']
    """
    reservoir_stream: Reservoir sampling agnostic of any conditionals.
    
    reservoir_action: Reservoir sampling per action-bin, with bins defined by separate actions (verb,noun) pairs.
    All bins have same capacity (and may not be entirely filled).
    
    reservoir_verbnoun: Reservoir sampling per verbnoun-bin, with bins defined by separate verbs or nouns. This allows 
    sharing between actions with an identical verb or noun. All bins have same capacity (and may not be entirely filled).
    """

    # TODO VERBNOUN_BALANCED: A BIN PER SEPARATE VERB N NOUN (be careful not to store samples twice!)

    def __init__(self, cfg, lightning_module):
        super().__init__(cfg, lightning_module)
        self.mem_size = int(cfg.METHOD.REPLAY.MEMORY_SIZE_SAMPLES)
        self.storage_policy = cfg.METHOD.REPLAY.STORAGE_POLICY
        assert self.storage_policy in self.storage_policies

        self.train_stream_dataset = lightning_module.train_dataloader().dataset
        self.num_workers_replay = cfg.DATA_LOADER.NUM_WORKERS  # Doubles the number of workers

        self.conditional_memory = OrderedDict({})  # Map <Conditional, datastream_idx_list>
        self.memory_dataloader_idxs = []  # Use this to update the memory indices of the stream

        # Retrieval state vars
        self.new_batch_size = None  # How many from stream

    def on_before_batch_transfer(self, new_batch: Any, dataloader_idx: int) -> Any:
        self.new_batch_size = new_batch.shape[0]

        # Retrieve from memory
        mem_batch = self.retrieve_rnd_batch_from_mem(mem_batch_size=self.new_batch_size)

        # unpack and join mem and new
        joined_batch = self.concat_batches(new_batch, mem_batch)

        return joined_batch

    def retrieve_rnd_batch_from_mem(self, mem_batch_size) -> Tensor:
        """ Sample stream idxs from replay memory. Create loader and load all selected samples at once. """
        # Sample idxs of our stream_idxs (without resampling)
        # Random sampling, weighed by len per conditional bin
        flat_mem = list(itertools.chain(self.conditional_memory.values()))
        stream_idxs = random.sample(flat_mem, mem_batch_size)

        # Load the samples from history of stream
        stream_subset = torch.utils.data.Subset(self.train_stream_dataset, stream_idxs)

        loader = torch.utils.data.DataLoader(
            stream_subset,
            batch_size=mem_batch_size,
            num_workers=self.num_workers_replay,
            shuffle=False, pin_memory=True, drop_last=False,
        )
        logger.debug(f"Created Replay dataloader. batch_size={loader.batch_size}, "
                     f"samples={mem_batch_size}, num_batches={len(loader)}")

        assert len(loader) == 1, f"Dataloader should return all in 1 batch."
        mem_batch = next(loader)
        return mem_batch

    @staticmethod
    def concat_batches(batch1, batch2):
        # inputs, labels, video_names, stream_sample_idxs = batch
        joined_batch = [None] * 4

        # Tensors concat in batch dim
        tensor_idxs = [0, 1, 3]
        for tensor_idx in tensor_idxs:
            joined_batch[tensor_idx] = torch.cat([batch1[tensor_idx], batch2[tensor_idx]], dim=0)  # Add in batch dim

        # List
        joined_batch.append(batch1[2] + batch2[2])

        return joined_batch

    def training_step(self, inputs, labels, *args, current_batch_sample_idxs=None, **kwargs) \
            -> Tuple[Tensor, List[Tensor], Dict]:
        """ Training step for the method when observing a new batch.
        Return Loss,  prediction outputs,a nd dictionary of result metrics to log."""
        assert current_batch_sample_idxs is not None, "Specify current_batch_sample_idxs for Replay"
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
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='verb',
                           base_metric_name=f"loss_total"): loss_total_verbs.item(),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='noun',
                           base_metric_name=f"loss_total"): loss_total_nouns.item(),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='action',
                           base_metric_name=f"loss_total"): loss_total_actions.item(),

            # Mem
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='verb',
                           base_metric_name=f"loss_mem"): loss_mem_verbs.item(),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='noun',
                           base_metric_name=f"loss_mem"): loss_mem_nouns.item(),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='action',
                           base_metric_name=f"loss_mem"): loss_mem_actions.item(),

            # New
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='verb',
                           base_metric_name=f"loss_new"): loss_new_verbs.item(),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='noun',
                           base_metric_name=f"loss_new"): loss_new_nouns.item(),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='action',
                           base_metric_name=f"loss_new"): loss_new_actions.item(),
        }

        # TODO: Store samples
        self._store_samples_in_replay_memory(labels, current_batch_sample_idxs)

        return loss_total_actions, preds, log_results

    def _store_samples_in_replay_memory(self, labels, current_batch_sample_idxs):
        """"""
        # Based on action labels create new bins and cutoff others, e.g. reservoir sampling

        # Do 2 versions: 1 conditional (set per class) , 2 unconditional( Only 1 set)
        self.is_action_balanced
        raise NotImplementedError()

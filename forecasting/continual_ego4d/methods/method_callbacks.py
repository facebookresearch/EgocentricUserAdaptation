from abc import ABC, abstractmethod
from continual_ego4d.methods.build import build_method, METHOD_REGISTRY
from typing import Dict, Tuple, List, Any
from torch import Tensor
from pytorch_lightning.core import LightningModule
import torch
from collections import defaultdict
from continual_ego4d.metrics.metric import get_metric_tag
from continual_ego4d.metrics.count_metrics import TAG_BATCH
from collections import OrderedDict
import random
import itertools
from continual_ego4d.datasets.continual_action_recog_dataset import verbnoun_to_action
from continual_ego4d.metrics.standard_metrics import OnlineLossMetric
from ego4d.utils import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from continual_ego4d.tasks.continual_action_recog_task import ContinualMultiTaskClassificationTask

logger = logging.get_logger(__name__)


class Method:
    def __init__(self, cfg, lightning_module: 'ContinualMultiTaskClassificationTask'):
        self.cfg = cfg  # For method-specific params
        self.lightning_module = lightning_module
        self.trainer = self.lightning_module.trainer

        self.loss_fun_train = self.lightning_module.loss_fun_unred
        self.loss_fun_pred = self.lightning_module.loss_fun_unred

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch

    def update_stream_tracking(
            self,
            stream_sample_idxs: list[int],
            new_batch_feats,
            new_batch_verb_pred,  # Preds
            new_batch_noun_pred,
            new_batch_action_loss,  # Losses
            new_batch_verb_loss,
            new_batch_noun_loss,
    ):
        """ Wrapper that forwards samples but also """
        ss = self.lightning_module.stream_state

        streamtrack_to_batchval = (
            (ss.sample_idx_to_feat, new_batch_feats),
            (ss.sample_idx_to_verb_pred, new_batch_verb_pred),
            (ss.sample_idx_to_noun_pred, new_batch_noun_pred),
            (ss.sample_idx_to_action_loss, new_batch_action_loss),
            (ss.sample_idx_to_verb_loss, new_batch_verb_loss),
            (ss.sample_idx_to_noun_loss, new_batch_noun_loss),
        )

        # Make copies on cpu and detach from computational graph
        streamtrack_to_batchval_cp = []
        for streamtrack_list, batch_vals in streamtrack_to_batchval:
            if isinstance(batch_vals, torch.Tensor):
                batch_vals = batch_vals.cpu().detach()

            # Check all same length
            assert len(batch_vals) == len(stream_sample_idxs), \
                f"batch_shape {len(batch_vals)} not matching len stream_idxs {len(stream_sample_idxs)}"

            streamtrack_to_batchval_cp.append(
                (streamtrack_list, batch_vals)
            )
        del streamtrack_to_batchval

        # Add tracking values
        for batch_idx, stream_sample_idx in enumerate(stream_sample_idxs):
            for streamtrack_list, batch_vals in streamtrack_to_batchval_cp:
                batch_val = batch_vals[batch_idx]
                try:
                    batch_val = batch_val.item()
                except:
                    pass
                streamtrack_list[stream_sample_idx] = batch_val

    def training_step(self, inputs, labels, current_batch_stream_idxs: list, *args, **kwargs) -> \
            Tuple[Tensor, List[Tensor], Dict]:
        """ Training step for the method when observing a new batch.
        Return Loss,  prediction outputs,a nd dictionary of result metrics to log."""

        preds, feats = self.lightning_module.forward(inputs, return_feats=True)
        loss_action, loss_verb, loss_noun = OnlineLossMetric.get_losses_from_preds(
            preds, labels, self.loss_fun_train, mean=False
        )

        self.update_stream_tracking(
            stream_sample_idxs=current_batch_stream_idxs,
            new_batch_feats=feats,
            new_batch_verb_pred=preds[0],
            new_batch_noun_pred=preds[1],
            new_batch_action_loss=loss_action,
            new_batch_verb_loss=loss_verb,
            new_batch_noun_loss=loss_noun,
        )

        # Reduce
        loss_action_m = loss_action.mean()
        loss_verb_m = loss_verb.mean()
        loss_noun_m = loss_noun.mean()

        log_results = {
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='action', base_metric_name='loss'):
                loss_action_m.item(),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='verb', base_metric_name='loss'):
                loss_verb_m.item(),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='noun', base_metric_name='loss'):
                loss_noun_m.item(),
        }
        return loss_action_m, preds, log_results

    def prediction_step(self, inputs, labels, stream_sample_idxs: list, *args, **kwargs) \
            -> Tuple[Tensor, List[Tensor], Dict]:
        """ Default: Get all info we also get during training."""
        preds: list = self.lightning_module.forward(inputs)
        loss_action, loss_verb, loss_noun = OnlineLossMetric.get_losses_from_preds(
            preds, labels, self.loss_fun_pred, mean=False
        )

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

    def training_step(self, inputs, labels, current_batch_stream_idxs: list, *args, **kwargs):
        return super().training_step(inputs, labels, current_batch_stream_idxs, *args, **kwargs)


@METHOD_REGISTRY.register()
class Replay(Method):
    """
    Pytorch Subset of original stream, with expanding indices.
    """

    storage_policies = ['window', 'reservoir_stream', 'reservoir_action', 'reservoir_verbnoun']
    """
    window: A moving window with the current bath iteration. e.g. for Mem-size M, [t-M,t] indices are considered.
    
    reservoir_stream: Reservoir sampling agnostic of any conditionals.
    {None:[Memory-list]}
    
    reservoir_action: Reservoir sampling per action-bin, with bins defined by separate actions (verb,noun) pairs.
    All bins have same capacity (and may not be entirely filled).
    {action-tuple:[Memory-list]}
    
    reservoir_verbnoun: Reservoir sampling per verbnoun-bin, with bins defined by separate verbs or nouns. This allows 
    sharing between actions with an identical verb or noun. All bins have same capacity (and may not be entirely filled).
    {str: "{verb,noun}_label-int":[Memory-list]}
    """

    # TODO VERBNOUN_BALANCED: A BIN PER SEPARATE VERB N NOUN (be careful not to store samples twice!)

    def __init__(self, cfg, lightning_module):
        super().__init__(cfg, lightning_module)
        self.total_mem_size = int(cfg.METHOD.REPLAY.MEMORY_SIZE_SAMPLES)
        self.storage_policy = cfg.METHOD.REPLAY.STORAGE_POLICY
        assert self.storage_policy in self.storage_policies

        self.train_stream_dataset = lightning_module.train_dataloader().dataset
        self.num_workers_replay = cfg.DATA_LOADER.NUM_WORKERS  # Doubles the number of workers

        self.conditional_memory = OrderedDict({})  # Map <Conditional, datastream_idx_list>
        self.memory_dataloader_idxs = []  # Use this to update the memory indices of the stream

        # Retrieval state vars
        self.new_batch_size = None  # How many from stream

        # storage state vars
        self.mem_size_per_conditional = self.total_mem_size  # Will be updated
        self.num_observed_samples = 0
        self.num_samples_memory = 0

    def on_before_batch_transfer(self, new_batch: Any, dataloader_idx: int) -> Any:
        _, new_batch_labels, *_ = new_batch
        self.new_batch_size = new_batch_labels.shape[0]

        # Retrieve from memory
        mem_batch = None
        num_samples_retrieve = min(self.new_batch_size, self.num_samples_memory)
        if num_samples_retrieve > 0:
            mem_batch = self.retrieve_rnd_batch_from_mem(mem_batch_size=num_samples_retrieve)

        # unpack and join mem and new
        joined_batch = self.concat_batches(new_batch, mem_batch) if mem_batch is not None else new_batch

        return joined_batch

    def retrieve_rnd_batch_from_mem(self, mem_batch_size) -> Tensor:
        """ Sample stream idxs from replay memory. Create loader and load all selected samples at once. """
        # Sample idxs of our stream_idxs (without resampling)
        # Random sampling, weighed by len per conditional bin
        flat_mem = list(itertools.chain(*self.conditional_memory.values()))
        nb_samples_mem = min(len(flat_mem), mem_batch_size)
        stream_idxs = random.sample(flat_mem, k=nb_samples_mem)
        logger.debug(f"Retrieving {nb_samples_mem} samples from memory: {stream_idxs}")

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
        mem_batch = next(iter(loader))
        return mem_batch

    @staticmethod
    def concat_batches(batch1, batch2):
        # inputs, labels, video_names, stream_sample_idxs = batch
        joined_batch = [None] * 4

        # Input is 2dim-list (verb,noun) of input-tensors
        joined_batch[0] = [
            torch.cat([batch1[0][idx], batch2[0][idx]], dim=0) for idx in range(2)
        ]

        # Tensors concat directly in batch dim
        tensor_idxs = [1, 3]
        for tensor_idx in tensor_idxs:
            joined_batch[tensor_idx] = torch.cat([batch1[tensor_idx], batch2[tensor_idx]], dim=0)  # Add in batch dim

        # List
        joined_batch[2] = batch1[2] + batch2[2]

        return joined_batch

    def training_step(self, inputs, labels, current_batch_stream_idxs: list, *args, **kwargs) \
            -> Tuple[Tensor, List[Tensor], Dict]:
        """ Training step for the method when observing a new batch.
        Return Loss,  prediction outputs,a nd dictionary of result metrics to log."""
        assert len(current_batch_stream_idxs) == self.new_batch_size, \
            "current_batch_stream_idxs should only include new-batch idxs"
        total_batch_size = labels.shape[0]
        mem_batch_size = total_batch_size - self.new_batch_size
        self.num_observed_samples += self.new_batch_size

        # Forward at once
        preds, feats = self.lightning_module.forward(inputs, return_feats=True)

        # Unreduced losses
        loss_verbs = self.loss_fun_pred(preds[0], labels[:, 0])  # Verbs
        loss_nouns = self.loss_fun_pred(preds[1], labels[:, 1])  # Nouns

        # Disentangle new, buffer, and total losses
        loss_new_verbs = loss_verbs[:self.new_batch_size]
        loss_new_nouns = loss_nouns[:self.new_batch_size]
        loss_new_actions = loss_new_verbs + loss_new_nouns

        # Reduce
        loss_new_verbs_m = loss_new_verbs.mean()
        loss_new_nouns_m = loss_new_nouns.mean()
        loss_new_actions_m = loss_new_actions.mean()

        if mem_batch_size > 0:  # Memory samples added
            loss_mem_verbs_m = torch.mean(loss_verbs[self.new_batch_size:])
            loss_mem_nouns_m = torch.mean(loss_nouns[self.new_batch_size:])
            loss_mem_actions_m = loss_mem_verbs_m + loss_mem_nouns_m

        else:
            loss_mem_verbs_m = loss_mem_nouns_m = loss_mem_actions_m = torch.FloatTensor([0]).to(loss_verbs.device)

        loss_total_verbs_m = (loss_new_verbs_m + loss_mem_verbs_m) / 2
        loss_total_nouns_m = (loss_new_nouns_m + loss_mem_nouns_m) / 2
        loss_total_actions_m = (loss_new_actions_m + loss_mem_actions_m) / 2

        self.update_stream_tracking(
            stream_sample_idxs=current_batch_stream_idxs,
            new_batch_feats=feats[:self.new_batch_size],
            new_batch_verb_pred=preds[0][:self.new_batch_size],
            new_batch_noun_pred=preds[1][:self.new_batch_size],
            new_batch_action_loss=loss_new_actions,
            new_batch_verb_loss=loss_new_verbs,
            new_batch_noun_loss=loss_new_nouns,
        )

        log_results = {
            # Total
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='verb',
                           base_metric_name=f"loss_total"): loss_total_verbs_m.mean().item(),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='noun',
                           base_metric_name=f"loss_total"): loss_total_nouns_m.item(),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='action',
                           base_metric_name=f"loss_total"): loss_total_actions_m.item(),

            # Mem
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='verb',
                           base_metric_name=f"loss_mem"): loss_mem_verbs_m.item(),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='noun',
                           base_metric_name=f"loss_mem"): loss_mem_nouns_m.item(),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='action',
                           base_metric_name=f"loss_mem"): loss_mem_actions_m.item(),

            # New
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='verb',
                           base_metric_name=f"loss_new"): loss_new_verbs.item(),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='noun',
                           base_metric_name=f"loss_new"): loss_new_nouns.item(),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='action',
                           base_metric_name=f"loss_new"): loss_new_actions.item(),
        }

        # Store samples
        self._store_samples_in_replay_memory(labels, current_batch_stream_idxs)

        # Update size
        self.num_samples_memory = sum(len(cond_mem) for cond_mem in self.conditional_memory.values())

        return loss_total_actions_m, preds, log_results

    def _store_samples_in_replay_memory(self, labels: torch.LongTensor, current_batch_stream_idxs: list):
        """"""

        if self.storage_policy == 'reservoir_stream':
            self.conditional_memory[None] = self.reservoir_sampling(
                self.conditional_memory.get(None, []), current_batch_stream_idxs, self.total_mem_size)

        elif self.storage_policy == 'reservoir_action':
            self.reservoir_action_storage_policy(labels, current_batch_stream_idxs)

        elif self.storage_policy == 'reservoir_verbnoun':
            raise NotImplementedError()

        elif self.storage_policy == 'window':
            min_current_batch_idx = min(current_batch_stream_idxs)
            self.conditional_memory[None] = list(range(
                max(0, min_current_batch_idx - self.total_mem_size),
                min_current_batch_idx)
            )

        else:
            raise ValueError()

    def reservoir_action_storage_policy(self, labels: torch.LongTensor, current_batch_stream_idxs: list):
        """ Memory is divided in equal memory bins per observed action.
        Each bin is populated using reservoir sampling when new samples for that bin are encountered.
        """
        label_batch_axis = 0

        # Collect actions (label pairs) and count new ones
        batch_actions = []
        new_actions_observed = 0
        for idx, verbnoun_t in enumerate(torch.unbind(labels, dim=label_batch_axis)):
            action = verbnoun_to_action(*verbnoun_t.tolist())
            batch_actions.append(action)

            if action not in self.conditional_memory:
                new_actions_observed += 1
                self.conditional_memory[action] = []

        # Update max mem size and cutoff those exceeding
        if new_actions_observed >= 0:
            self.mem_size_per_conditional = self.total_mem_size // len(self.conditional_memory)

            for action, action_mem in self.conditional_memory.items():
                self.conditional_memory[action] = action_mem[:self.mem_size_per_conditional]

        # Add new batch samples
        obs_actions_batch = set()
        for action in batch_actions:
            if action in obs_actions_batch:  # Already processed, so skip
                continue
            obs_actions_batch.add(action)

            # Process all samples in batch for this action at once
            label_mask = torch.cat([
                torch.BoolTensor([verbnoun_to_action(*verbnoun_t.tolist()) == action])
                for verbnoun_t in torch.unbind(labels, dim=label_batch_axis)
            ], dim=label_batch_axis)

            selected_inbatch_idxs = torch.nonzero(label_mask)
            current_batch_stream_idxs_subset = [
                stream_idx for inbatch_idx, stream_idx in enumerate(current_batch_stream_idxs)
                if inbatch_idx in selected_inbatch_idxs
            ]

            self.conditional_memory[action] = self.reservoir_sampling(
                self.conditional_memory[action], current_batch_stream_idxs_subset, self.mem_size_per_conditional)

    def reservoir_sampling(self, memory: list, new_stream_idxs: list, mem_size_limit: int) -> list:
        """ Fill buffer if not full yet, otherwise replace with probability mem_size/num_observed_samples. """

        for new_stream_idx in new_stream_idxs:
            if len(memory) < mem_size_limit:  # Buffer not filled yet
                memory.append(new_stream_idx)
            else:  # Replace with probability mem_size/num_observed_samples
                rnd_idx = random.randint(0, self.num_observed_samples)  # [a,b]
                if rnd_idx < mem_size_limit:  # Replace if sampled in memory
                    memory[rnd_idx] = new_stream_idx

        return memory

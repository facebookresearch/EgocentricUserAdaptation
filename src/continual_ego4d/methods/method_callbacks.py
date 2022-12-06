import copy
import itertools
import pprint
import random
from collections import OrderedDict
from collections import deque
from typing import Dict, Tuple, List, Any
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from continual_ego4d.datasets.continual_action_recog_dataset import verbnoun_to_action
from continual_ego4d.methods.build import METHOD_REGISTRY
from continual_ego4d.metrics.count_metrics import TAG_BATCH
from continual_ego4d.metrics.metric import get_metric_tag
from continual_ego4d.metrics.standard_metrics import OnlineLossMetric
from continual_ego4d.utils.models import get_name_to_grad_dict, grad_dict_to_vector
from ego4d.utils import logging

if TYPE_CHECKING:
    from continual_ego4d.tasks.continual_action_recog_task import ContinualMultiTaskClassificationTask

logger = logging.get_logger(__name__)


class Method:
    run_predict_before_train = True

    def __init__(self, cfg, lightning_module: 'ContinualMultiTaskClassificationTask', iid: bool = False):
        """ At Task init, the lightning_module.trainer=None. """
        if not iid:
            assert not cfg.DATA.SHUFFLE_DS_ORDER
        else:
            assert cfg.DATA.SHUFFLE_DS_ORDER, f"Need shuffled dataset list for iid method."

        self.cfg = cfg  # For method-specific params
        self.lightning_module = lightning_module

        self.loss_fun_train = self.lightning_module.loss_fun_unred
        self.loss_fun_pred = self.lightning_module.loss_fun_unred

        # Compare current gradient with previous steps
        self.compare_gradients = cfg.METHOD.ANALYZE_GRADS_WINDOW
        self.max_gradient_window_size = cfg.METHOD.MAX_ANALYZE_GRADS_WINDOW_SIZE
        self._gradient_window = deque()  # FIFO: From Old to new

    def train_before_update_batch_adapt(self, batch: Any, batch_idx: int) -> Any:
        return batch

    def _update_stream_tracking(
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
        for entry_idx, (streamtrack_list, batch_vals) in enumerate(streamtrack_to_batchval):
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

    def training_first_forward(self, inputs, labels, current_batch_stream_idxs: list, *args, **kwargs) -> \
            Tuple[Tensor, List[Tensor], Dict]:
        """ Training step for the method when observing a new batch.
        Return Loss,  prediction outputs,a nd dictionary of result metrics to log.
        For multiple (N) training iterations,
        - we do updates in this method on all N-1 iterations,
        - For final iteration: we return the loss for normal flow in PL.
        """
        log_results = {}

        fwd_inputs = copy.deepcopy(inputs)  # SlowFast in-place alters the inputs
        preds, feats = self.lightning_module.forward(fwd_inputs, return_feats=True)
        loss_action, loss_verb, loss_noun = OnlineLossMetric.get_losses_from_preds(
            preds, labels, self.loss_fun_train, mean=False
        )

        # Only return latest loss
        self._update_stream_tracking(
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

        # import pdb;pdb.set_trace()
        if self.compare_gradients:
            self.log_gradient_analysis_(log_results, loss_action_m)

        log_results = {
            **log_results,
            **{
                get_metric_tag(TAG_BATCH, train_mode='train', action_mode='action', base_metric_name='loss'):
                    loss_action_m.item(),
                get_metric_tag(TAG_BATCH, train_mode='train', action_mode='verb', base_metric_name='loss'):
                    loss_verb_m.item(),
                get_metric_tag(TAG_BATCH, train_mode='train', action_mode='noun', base_metric_name='loss'):
                    loss_noun_m.item(),
            }
        }

        return loss_action_m, preds, log_results

    def training_update_loop(self, loss_first_fwd, inputs, labels, current_batch_stream_idxs: list, *args, **kwargs):
        """ Given the loss from the first forward, update for inner_loop_iters. """
        opt = self.lightning_module.optimizers()

        assert self.lightning_module.inner_loop_iters >= 1
        for inner_iter in range(1, self.lightning_module.inner_loop_iters + 1):

            if inner_iter == 1:
                loss_action = loss_first_fwd

            else:
                fwd_inputs = copy.deepcopy(inputs)  # SlowFast in-place alters the inputs
                preds = self.lightning_module.forward(fwd_inputs, return_feats=False)
                loss_action, loss_verb, loss_noun = OnlineLossMetric.get_losses_from_preds(
                    preds, labels, self.loss_fun_train, mean=False
                )

            # Reduce
            loss_action_m = loss_action.mean()

            opt.zero_grad()  # Also clean grads for final
            self.lightning_module.manual_backward(loss_action_m)  # Calculate grad
            opt.step()

            logger.info(f"[INNER-LOOP UPDATE] iter {inner_iter}/{self.lightning_module.inner_loop_iters}: "
                        f"fwd, bwd, step. Action_loss={loss_action_m.item()}")

    def prediction_step(self, inputs, labels, stream_sample_idxs: list, *args, **kwargs) \
            -> Tuple[Dict[int, Tensor], Dict[int, tuple], Dict[int, dict]]:
        """ Default: Get all info we also get during training."""
        preds: list[Tensor, Tensor] = self.lightning_module.forward(inputs, return_feats=False)
        loss_action, loss_verb, loss_noun = OnlineLossMetric.get_losses_from_preds(
            preds, labels, self.loss_fun_pred, mean=False
        )

        sample_to_loss = {}
        sample_to_pred = {}
        sample_to_label = {}
        for batch_idx, stream_sample_idx in enumerate(stream_sample_idxs):
            sample_to_loss[stream_sample_idx] = {
                get_metric_tag(TAG_BATCH, train_mode='pred', action_mode='action', base_metric_name='loss'):
                    loss_action[batch_idx].item(),
                get_metric_tag(TAG_BATCH, train_mode='pred', action_mode='verb', base_metric_name='loss'):
                    loss_verb[batch_idx].item(),
                get_metric_tag(TAG_BATCH, train_mode='pred', action_mode='noun', base_metric_name='loss'):
                    loss_noun[batch_idx].item(),
            }
            # Keep batch dim consistently (unsqueeze)
            sample_to_pred[stream_sample_idx] = tuple([preds[0][batch_idx].cpu().unsqueeze(0),
                                                       preds[1][batch_idx].cpu().unsqueeze(0)])
            sample_to_label[stream_sample_idx] = tuple(labels[batch_idx].cpu().tolist())

        return sample_to_pred, sample_to_label, sample_to_loss

    def log_gradient_analysis_(self, log_results, loss_stream_m):
        logger.debug(f"Comparing current batch gradient with window of previous")

        nb_grads_to_lookback = min(self.max_gradient_window_size, len(self._gradient_window))
        logger.debug(f"#grads to go back in history: {nb_grads_to_lookback}")

        opt = self.lightning_module.optimizers()

        # NEW batch
        opt.zero_grad()  # Make sure no grads
        self.lightning_module.manual_backward(loss_stream_m, retain_graph=True)  # keep intermediate vals
        current_grad_dict = get_name_to_grad_dict(self.lightning_module.model)
        opt.zero_grad()  # Clear gradients again for later full backprop

        # Now get for all subsets of grads the norm and cos-sim
        for nb_steps_lookback in range(1, len(self._gradient_window) + 1):
            for log_name, grad_name_filters, filter_incl in [
                ("full", None, True),
                ("slow", ['pathway0'], True),
                ("fast", ['pathway1'], True),
                ("head", ['head'], True),
                ("feat", ['head'], False),  # all but head
            ]:
                grad_mem_idx = len(self._gradient_window) - nb_steps_lookback
                log_prefix = f"LOOKBACK_STEP_{nb_steps_lookback}/{log_name}"

                # CURRENT results
                current_grad = grad_dict_to_vector(current_grad_dict, grad_name_filters,
                                                   include_filter=filter_incl, verbose=nb_grads_to_lookback == 1)
                current_grad_norm = current_grad.pow(2).sum().sqrt().item()  # Get L2 norm

                # Add results
                log_results[
                    get_metric_tag(TAG_BATCH, train_mode='analyze', action_mode='action',
                                   base_metric_name=f"{log_prefix}_current_grad_norm")
                ] = current_grad_norm

                # grad MEM result
                mem_grad_dict = self._gradient_window[grad_mem_idx]
                mem_grad = grad_dict_to_vector(mem_grad_dict, grad_name_filters,
                                               include_filter=filter_incl, verbose=nb_grads_to_lookback == 1)
                mem_grad_norm = mem_grad.pow(2).sum().sqrt().item()  # Get L2 norm

                log_results[
                    get_metric_tag(TAG_BATCH, train_mode='analyze', action_mode='action',
                                   base_metric_name=f"{log_prefix}_mem_grad_norm")
                ] = mem_grad_norm

                grad_cos_sim = (F.normalize(current_grad, p=2, dim=0) * F.normalize(mem_grad, p=2, dim=0)).sum().item()

                log_results[
                    get_metric_tag(TAG_BATCH, train_mode='analyze', action_mode='action',
                                   base_metric_name=f"{log_prefix}_grad_cos_sim")
                ] = grad_cos_sim

        # Update window grad memory
        self._gradient_window.append(current_grad_dict)  # Add current as newest
        if len(self._gradient_window) > self.max_gradient_window_size:
            self._gradient_window.popleft()

        logger.info(f"GRADIENT MEMORY SIZE: {len(self._gradient_window)}/{self.max_gradient_window_size}")


@METHOD_REGISTRY.register()
class IIDFinetuning(Method):
    """
    Overwrite lightning_module dataloaders to do IID finetuning.
    This still enables learning from the stream, but an IID shuffled stream, while tracking online metrics such as OAG.
    For multi-epoch training, define a IID-task instead, where we only use the same dataloader, but don't care to
    track metrics.
    """
    run_predict_before_train = True

    def __init__(self, cfg, lightning_module):
        super().__init__(cfg, lightning_module, iid=True)

        # Adhoc setting shuffle = True
        assert isinstance(lightning_module.train_loader.sampler, SequentialSampler), \
            f"Regular training should have sequential sampler (no shuffle)"

        # ASSERT ds is shuffled a priori, as dataloader idx in __get_item__ is used as id of the samples in the list
        # These are assumed to be consecutive in the stream (need Sequential sampler).
        # As dataset is initialized before method, we need to define in cfg at creation, as changing ds params may
        # result in wrong references in the StreamState.
        assert cfg.DATA.SHUFFLE_DS_ORDER, f"Need shuffled dataset list."


@METHOD_REGISTRY.register()
class Finetuning(Method):
    run_predict_before_train = True

    def __init__(self, cfg, lightning_module):
        super().__init__(cfg, lightning_module)

    def training_first_forward(self, inputs, labels, current_batch_stream_idxs: list, *args, **kwargs):
        return super().training_first_forward(inputs, labels, current_batch_stream_idxs, *args, **kwargs)


@METHOD_REGISTRY.register()
class FixedNetwork(Method):
    """ Network that is not updated. Can be used to obtain running performance over stream for fixed model."""
    run_predict_before_train = False

    def __init__(self, cfg, lightning_module):
        super().__init__(cfg, lightning_module)

    def training_first_forward(self, inputs, labels, current_batch_stream_idxs: list, *args, **kwargs):
        return super().training_first_forward(inputs, labels, current_batch_stream_idxs, *args, **kwargs)

    def training_update_loop(self, loss_first_fwd, inputs, labels, current_batch_stream_idxs: list, *args, **kwargs):
        logger.info("Skipping update as no params to train")


@METHOD_REGISTRY.register()
class LabelWindowPredictor(Method):
    """
    Stores window of labels to use frequency for prediction distribution.
    """
    run_predict_before_train = False

    def __init__(self, cfg, lightning_module):
        super().__init__(cfg, lightning_module)

        # Empty task metrics
        lightning_module.past_metrics = []
        lightning_module.future_metrics = []
        # lightning_module.current_batch_metrics = [] # Get ACC etc for current batch
        lightning_module.continual_eval_freq = -1
        lightning_module.plotting_log_freq = -1
        logger.debug(f"Reset all metrics to empty.")

        # Overwrite dataloaders
        assert cfg.DATA.RETURN_VIDEO is False, \
            f"Must set cfg.DATA.RETURN_VIDEO=False for efficient loading of stream labels only"
        assert lightning_module.train_loader.dataset.return_video is False  # Only return labels
        lightning_module.predict_loader = None

        # States
        self.window_size = cfg.ANALYZE_STREAM.WINDOW_SIZE_SAMPLES  # Determines length of window in samples
        assert self.window_size > 0

        # Memory history
        self.label_window: deque[tuple] = deque()  # Each window is a list of action-tuples

        # For pred creation
        self.verbs_count, self.nouns_count = cfg.MODEL.NUM_CLASSES  # Verbs, nouns

    def training_first_forward(self, inputs_t, labels_t, current_batch_stream_idxs: list, *args, **kwargs):
        """ For each sample, use the WINDOW-SIZE preceding samples for prediction distr.
        As action is predicted as 2 independent classifiers (verbs,nouns), we should normalize predictions for each
        distribution. This also means we should count the verbs/nouns separately to form these distributions. """
        loss = None
        log_results = {}

        verb_prediction_list: list[torch.Tensor] = []
        noun_prediction_list: list[torch.Tensor] = []
        for current_batch_stream_idx in current_batch_stream_idxs:

            # Get prediction tensors based on existing window
            pred_verb_t = torch.zeros(self.verbs_count).to(labels_t.device)
            pred_noun_t = torch.zeros(self.nouns_count).to(labels_t.device)
            if len(self.label_window) > 0:
                verbs_in_window = [a[0] for a in self.label_window]
                nouns_in_window = [a[1] for a in self.label_window]

                for verb_idx, noun_idx in zip(verbs_in_window, nouns_in_window):
                    pred_verb_t[verb_idx] += 1
                    pred_noun_t[noun_idx] += 1

                # Normalize
                pred_verb_tn = pred_verb_t / sum(pred_verb_t)
                pred_noun_tn = pred_noun_t / sum(pred_noun_t)

            else:  # If empty window, predict uniform
                pred_verb_tn = pred_verb_t.fill_(1 / self.verbs_count)
                pred_noun_tn = pred_noun_t.fill_(1 / self.verbs_count)

            verb_prediction_list.append(pred_verb_tn.unsqueeze(0))
            noun_prediction_list.append(pred_noun_tn.unsqueeze(0))

            # Get formatted (non-tensor) label
            action_label: tuple[int, int] = self.lightning_module.stream_state.sample_idx_to_action_list[
                current_batch_stream_idx
            ]

            # Add to window
            self.label_window.append(action_label)

            # Pop if oversize
            if len(self.label_window) > self.window_size:
                self.label_window.popleft()

        # Concat preds
        verb_preds = torch.cat(verb_prediction_list)
        noun_preds = torch.cat(noun_prediction_list)
        preds = (verb_preds, noun_preds)

        return loss, preds, log_results

    def training_update_loop(self, loss_first_fwd, inputs, labels, current_batch_stream_idxs: list, *args, **kwargs):
        """ No updates made. """
        pass

    def prediction_step(self, inputs, labels, stream_sample_idxs: list, *args, **kwargs) \
            -> Tuple[Tensor, List[Tensor], Dict]:
        raise NotImplementedError("Should not call prediction for this method.")


@METHOD_REGISTRY.register()
class BatchedLabelWindowPredictor(LabelWindowPredictor):
    """
    Uses the previous batch iteration to predict current batch samples.
    This is different from the previous <batch-size> preceding samples of each sample in the LabelWindowPredictor!
    BatchedLabelWindowPredictor gives a fair comparison with methods to see how much they fit to the current batch.
    """
    run_predict_before_train = False

    def __init__(self, cfg, lightning_module):
        super().__init__(cfg, lightning_module)

        # Empty task metrics
        lightning_module.past_metrics = []
        lightning_module.future_metrics = []
        # lightning_module.current_batch_metrics = [] # Get ACC etc for current batch
        lightning_module.continual_eval_freq = -1
        lightning_module.plotting_log_freq = -1
        logger.debug(f"Reset all metrics to empty.")

        # Overwrite dataloaders
        assert cfg.DATA.RETURN_VIDEO is False, \
            f"Must set cfg.DATA.RETURN_VIDEO=False for efficient loading of stream labels only"
        assert lightning_module.train_loader.dataset.return_video is False  # Only return labels
        lightning_module.predict_loader = None

        # States
        self.window_size = cfg.ANALYZE_STREAM.WINDOW_SIZE_SAMPLES  # Determines length of window in samples
        assert self.window_size > 0

        # Memory history
        self.label_window: deque[tuple] = deque()  # Each window is a list of action-tuples

        # For pred creation
        self.verbs_count, self.nouns_count = cfg.MODEL.NUM_CLASSES  # Verbs, nouns

    def training_first_forward(self, inputs_t, labels_t, current_batch_stream_idxs: list, *args, **kwargs):
        """ For each sample, use the WINDOW-SIZE preceding samples for prediction distr.
        As action is predicted as 2 independent classifiers (verbs,nouns), we should normalize predictions for each
        distribution. This also means we should count the verbs/nouns separately to form these distributions. """
        loss = None
        log_results = {}

        batch_idx = self.lightning_module.stream_state.batch_idx
        batch_size = self.lightning_module.train_loader.batch_size  # Retrieve like this for last batch remain cst
        assert self.window_size == batch_size, f"Only supports to use the exact batch size."

        # Get prediction tensors based on previous batch
        pred_verb_t = torch.zeros(self.verbs_count).to(labels_t.device)
        pred_noun_t = torch.zeros(self.nouns_count).to(labels_t.device)
        if batch_idx > 0:
            assert len(self.label_window) > 0

            verbs_in_window = [a[0] for a in self.label_window]
            nouns_in_window = [a[1] for a in self.label_window]

            for verb_idx, noun_idx in zip(verbs_in_window, nouns_in_window):
                pred_verb_t[verb_idx] += 1
                pred_noun_t[noun_idx] += 1

            # Normalize
            pred_verb_tn = pred_verb_t / sum(pred_verb_t)
            pred_noun_tn = pred_noun_t / sum(pred_noun_t)

        elif batch_idx == 0:  # If empty window, predict uniform
            pred_verb_tn = pred_verb_t.fill_(1 / self.verbs_count)
            pred_noun_tn = pred_noun_t.fill_(1 / self.verbs_count)

        else:
            raise ValueError("Negative batch idx")

        # ADD samples only after predicting with previous batch: Current bath not influenced
        # Get formatted (non-tensor) label
        for current_batch_stream_idx in current_batch_stream_idxs:
            action_label: tuple[int, int] = self.lightning_module.stream_state.sample_idx_to_action_list[
                current_batch_stream_idx
            ]
            self.label_window.append(action_label)  # Add to window

        # Pop if oversize
        while len(self.label_window) > self.window_size:
            self.label_window.popleft()

        # Concat preds
        verb_preds = pred_verb_tn.unsqueeze(0).expand(len(current_batch_stream_idxs), -1)
        noun_preds = pred_noun_tn.unsqueeze(0).expand(len(current_batch_stream_idxs), -1)
        preds = (verb_preds, noun_preds)

        # Update
        empty = [None] * len(current_batch_stream_idxs)
        self._update_stream_tracking(
            stream_sample_idxs=current_batch_stream_idxs,
            new_batch_feats=empty,
            new_batch_verb_pred=preds[0],
            new_batch_noun_pred=preds[1],
            new_batch_action_loss=empty,
            new_batch_verb_loss=empty,
            new_batch_noun_loss=empty,
        )

        return loss, preds, log_results

    def training_update_loop(self, loss_first_fwd, inputs, labels, current_batch_stream_idxs: list, *args, **kwargs):
        """ No updates made. """
        pass

    def prediction_step(self, inputs, labels, stream_sample_idxs: list, *args, **kwargs) \
            -> Tuple[Tensor, List[Tensor], Dict]:
        raise NotImplementedError("Should not call prediction for this method.")


@METHOD_REGISTRY.register()
class HindsightLabelWindowPredictor(Method):
    """
    Use last batch-size samples as prediction distribution.
    """
    run_predict_before_train = False

    def __init__(self, cfg, lightning_module):
        super().__init__(cfg, lightning_module)

        # Empty task metrics
        lightning_module.past_metrics = []
        lightning_module.future_metrics = []
        # lightning_module.current_batch_metrics = [] # Get ACC etc for current batch
        lightning_module.continual_eval_freq = -1
        lightning_module.plotting_log_freq = -1
        logger.debug(f"Reset all metrics to empty.")

        # Overwrite dataloaders
        assert cfg.DATA.RETURN_VIDEO is False, \
            f"Must set cfg.DATA.RETURN_VIDEO=False for efficient loading of stream labels only"
        assert lightning_module.train_loader.dataset.return_video is False  # Only return labels
        lightning_module.predict_loader = None

        # States
        self.window_size = cfg.ANALYZE_STREAM.WINDOW_SIZE_SAMPLES  # Determines length of window in samples
        assert self.window_size > 0

        # For pred creation
        self.verbs_count, self.nouns_count = cfg.MODEL.NUM_CLASSES  # Verbs, nouns

        self._initialized_pred_distr = False
        self._final_window_verb_pred = None
        self._final_window_noun_pred = None

    def training_first_forward(self, inputs_t, labels_t, current_batch_stream_idxs: list, *args, **kwargs):
        """ For each sample, use the WINDOW-SIZE preceding samples for prediction distr.
        As action is predicted as 2 independent classifiers (verbs,nouns), we should normalize predictions for each
        distribution. This also means we should count the verbs/nouns separately to form these distributions. """
        loss = None
        log_results = {}
        batch_size = labels_t.shape[0]

        if not self._initialized_pred_distr:
            self._initialized_pred_distr = True

            # Get prediction tensors based on final sample window
            pred_verb_t = torch.zeros(self.verbs_count).to(labels_t.device)
            pred_noun_t = torch.zeros(self.nouns_count).to(labels_t.device)

            final_action_window = self.lightning_module.stream_state.sample_idx_to_action_list[-self.window_size:]

            verbs_in_window = [a[0] for a in final_action_window]
            nouns_in_window = [a[1] for a in final_action_window]

            for verb_idx, noun_idx in zip(verbs_in_window, nouns_in_window):
                pred_verb_t[verb_idx] += 1
                pred_noun_t[noun_idx] += 1

            # Normalize
            self._final_window_verb_pred = (pred_verb_t / sum(pred_verb_t)).unsqueeze(0)
            self._final_window_noun_pred = (pred_noun_t / sum(pred_noun_t)).unsqueeze(0)

        # Concat preds
        verb_preds = self._final_window_verb_pred.expand(batch_size, -1)
        noun_preds = self._final_window_noun_pred.expand(batch_size, -1)
        preds = (verb_preds, noun_preds)

        return loss, preds, log_results

    def training_update_loop(self, loss_first_fwd, inputs, labels, current_batch_stream_idxs: list, *args, **kwargs):
        """ No updates made. """
        pass

    def prediction_step(self, inputs, labels, stream_sample_idxs: list, *args, **kwargs) \
            -> Tuple[Tensor, List[Tensor], Dict]:
        raise NotImplementedError("Should not call prediction for this method.")


@METHOD_REGISTRY.register()
class Replay(Method):
    run_predict_before_train = True
    storage_policies = ['FIFO', 'reservoir', 'hybrid_CBRS']
    """
    FIFO: A moving window with the current bath iteration. e.g. for Mem-size M, [t-M,t] indices are considered.
    
    reservoir: Reservoir sampling agnostic of any conditionals.
    {None:[Memory-list]}
    
    hybrid_CBRS: Reservoir sampling per action-bin, with bins defined by separate actions (verb,noun) pairs.
    All bins have same capacity (and may not be entirely filled).
    {action-tuple:[Memory-list]}
    """

    def __init__(self, cfg, lightning_module):
        super().__init__(cfg, lightning_module)
        self.total_mem_size = int(cfg.METHOD.REPLAY.MEMORY_SIZE_SAMPLES)
        self.storage_policy = cfg.METHOD.REPLAY.STORAGE_POLICY
        self.resample_multi_iter = cfg.METHOD.REPLAY.RESAMPLE_MULTI_ITER
        assert self.storage_policy in self.storage_policies

        self.train_stream_dataset = lightning_module.train_dataloader().dataset
        self.num_workers_replay = cfg.METHOD.REPLAY.NUM_WORKERS  # Retrieve all batches in parallel

        self.conditional_memory: dict[Any, list] = OrderedDict({})  # Map <Conditional, datastream_idx_list>
        self.memory_dataloader_idxs = []  # Use this to update the memory indices of the stream

        # For class-balanced reservoir sampling (CBRS)
        self.conditional_num_observed_samples: dict[Any, int] = OrderedDict({})
        self.conditional_full: dict[Any, bool] = OrderedDict({})

        # Retrieval state vars
        self.new_batch_size = None  # How many from stream

        # storage state vars
        self.num_observed_samples = 0
        self.num_samples_memory = 0

        # Compare gradient norms and dot-product
        self.compare_gradients = cfg.METHOD.REPLAY.ANALYZE_GRADS

        # Tmp state vars
        self._current_new_batch = None

    def train_before_update_batch_adapt(self, new_batch: Any, batch_idx: int) -> Any:
        """ Before update, alter the new batch in the stream by adding a batch from Replay buffer of same size. """
        self._current_new_batch = new_batch  # Keep ref
        return self.new_batch_to_joined_batches(new_batch, n_sample_rounds=1)[0]  # Single batch

    def new_batch_to_joined_batches(self, new_batch, n_sample_rounds=1) -> list[Any]:
        """ Returns a list of joined batches, with length equal to n_sample_rounds.
        Each sample round samples randomly from the replay_strategies buffer and adds this to the new batch.
        A joined batch consists of the new batch concatenated with the replay_strategies batch, both of same size.
        """
        _, new_batch_labels, *_ = new_batch
        self.new_batch_size = new_batch_labels.shape[0]
        device = new_batch_labels.device
        num_samples_retrieve = min(self.new_batch_size, self.num_samples_memory)  # Single batch retrieve

        # Retrieve from memory
        if num_samples_retrieve == 0:
            joined_batches = [new_batch] * n_sample_rounds  # Return new batch only

        else:
            joined_batches = []
            all_rounds_stream_idxs: list[list[int]] = []

            # Sample from memory for each round
            for sample_round in range(n_sample_rounds):
                round_stream_idxs: list[int] = self.retrieve_rnd_sample_idxs_from_mem(
                    mem_batch_size=num_samples_retrieve
                )
                all_rounds_stream_idxs.append(round_stream_idxs)

            # Make set of all samples to retrieve (flatten, unique, sort)
            mem_load_idx_to_stream_idx = sorted(
                list(set([idx for idxlist in all_rounds_stream_idxs for idx in idxlist]))
            )
            stream_idx_to_mem_load_idx = {
                stream_idx: mem_load_ix for mem_load_ix, stream_idx in enumerate(mem_load_idx_to_stream_idx)
            }

            # Load in memory all at once
            all_rounds_single_mem_batch = self.load_stream_idxs(mem_load_idx_to_stream_idx)

            for round_stream_idxs in all_rounds_stream_idxs:
                # Stream idxs to mem_load idxs
                mem_load_idxs = [stream_idx_to_mem_load_idx[stream_idx] for stream_idx in round_stream_idxs]

                # Subset loaded memory:
                mem_batch_round = self.slice_batch(all_rounds_single_mem_batch, mem_load_idxs)

                # Add to return list
                joined_batch = self.concat_batches(new_batch, mem_batch_round, device)
                joined_batches.append(joined_batch)

        return joined_batches

    def retrieve_rnd_sample_idxs_from_mem(self, mem_batch_size) -> list[int]:
        """ Sample stream idxs from replay_strategies memory. """
        # Sample idxs of our stream_idxs (without resampling)
        # Random sampling, weighed by len per conditional bin
        flat_mem = list(itertools.chain(*self.conditional_memory.values()))
        nb_samples_mem = min(len(flat_mem), mem_batch_size)
        stream_idxs = random.sample(flat_mem, k=nb_samples_mem)
        logger.debug(f"Retrieved {len(stream_idxs)} sample idxs from memory: {stream_idxs}")

        return stream_idxs

    def load_stream_idxs(self, stream_idxs: list[int]) -> Tensor:
        """ Create loader and load all selected samples in stream at once.  """
        logger.debug(f"LOADING {len(stream_idxs)} memory stream idxs: {stream_idxs}")

        # Load the samples from history of stream
        stream_subset = torch.utils.data.Subset(self.train_stream_dataset, stream_idxs)

        loader = torch.utils.data.DataLoader(
            stream_subset,
            batch_size=max(1, len(stream_idxs) // self.num_workers_replay),
            num_workers=self.num_workers_replay,
            shuffle=False, pin_memory=True, drop_last=False,
        )
        logger.debug(f"Created Replay dataloader. batch_size={loader.batch_size}, "
                     f"samples={len(stream_idxs)}, num_batches={len(loader)}")

        full_batch = None
        for new_batch in loader:
            if full_batch is None:
                full_batch = new_batch
            else:
                full_batch = self.concat_batches(full_batch, new_batch)

        return full_batch

    @staticmethod
    def concat_batches(batch1, batch2, device=None):
        if device is None:  # Pick a default device
            device = batch1[1].device

        # inputs, labels, video_names, stream_sample_idxs = batch
        joined_batch = [None] * 4

        # Input is 2dim-list (verb,noun) of input-tensors
        joined_batch[0] = [
            torch.cat([batch1[0][idx].detach().to(device), batch2[0][idx].detach().to(device)], dim=0)
            for idx in range(2)
        ]

        # Tensors concat directly in batch dim
        tensor_idxs = [1, 3]
        for tensor_idx in tensor_idxs:  # Add in batch dim
            joined_batch[tensor_idx] = torch.cat([
                batch1[tensor_idx].detach().to(device),
                batch2[tensor_idx].detach().to(device)
            ], dim=0)

        # List
        joined_batch[2] = batch1[2] + batch2[2]

        return joined_batch

    @staticmethod
    def slice_batch(batch, selected_idxs):
        inputs, labels, video_names, stream_sample_idxs_t = batch

        inputs_ret: tuple[torch.Tensor, torch.Tensor] = (
            inputs[0][selected_idxs], inputs[1][selected_idxs]
        )
        labels_ret: torch.Tensor = labels[selected_idxs]
        video_names_ret: list[str] = [video_name for idx, video_name in enumerate(video_names) if idx in selected_idxs]
        stream_sample_idxs_t_ret: torch.Tensor = stream_sample_idxs_t[selected_idxs]

        return (inputs_ret, labels_ret, video_names_ret, stream_sample_idxs_t_ret)

    def training_first_forward(self, inputs, labels, current_batch_stream_idxs: list, *args, **kwargs) \
            -> Tuple[Tensor, List[Tensor], Dict]:
        """ Training step for the method when observing a new batch.
        Return Loss,  prediction outputs,a nd dictionary of result metrics to log."""
        assert len(current_batch_stream_idxs) == self.new_batch_size, \
            "current_batch_stream_idxs should only include new-batch idxs"

        total_batch_size = labels.shape[0]
        mem_batch_size = total_batch_size - self.new_batch_size
        self.num_observed_samples += self.new_batch_size

        log_results = {}

        # Forward at once
        fwd_inputs = copy.deepcopy(inputs)  # SlowFast in-place alters the inputs
        preds, feats = self.lightning_module.forward(fwd_inputs, return_feats=True)

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
            has_mem_batch = True
        else:
            loss_mem_verbs_m = loss_mem_nouns_m = loss_mem_actions_m = torch.FloatTensor([0]).to(loss_verbs.device)
            has_mem_batch = False

        loss_total_verbs_m = (loss_new_verbs_m + loss_mem_verbs_m) / 2
        loss_total_nouns_m = (loss_new_nouns_m + loss_mem_nouns_m) / 2
        loss_total_actions_m = (loss_new_actions_m + loss_mem_actions_m) / 2

        if self.compare_gradients:
            self.log_gradient_analysis_(log_results, loss_new_actions_m, loss_mem_actions_m, new_only=not has_mem_batch)

        self._update_stream_tracking(
            stream_sample_idxs=current_batch_stream_idxs,
            new_batch_feats=feats[:self.new_batch_size],
            new_batch_verb_pred=preds[0][:self.new_batch_size],
            new_batch_noun_pred=preds[1][:self.new_batch_size],
            new_batch_action_loss=loss_new_actions,
            new_batch_verb_loss=loss_new_verbs,
            new_batch_noun_loss=loss_new_nouns,
        )

        log_results = {
            **log_results, **{
                # Total
                get_metric_tag(TAG_BATCH, train_mode='train', action_mode='verb',
                               base_metric_name=f"loss_total"): loss_total_verbs_m.item(),
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
                               base_metric_name=f"loss_new"): loss_new_verbs_m.item(),
                get_metric_tag(TAG_BATCH, train_mode='train', action_mode='noun',
                               base_metric_name=f"loss_new"): loss_new_nouns_m.item(),
                get_metric_tag(TAG_BATCH, train_mode='train', action_mode='action',
                               base_metric_name=f"loss_new"): loss_new_actions_m.item(),
            }
        }

        return loss_total_actions_m, preds, log_results

    def log_gradient_analysis_(self, log_results, loss_new_actions_m, loss_mem_actions_m, new_only=False):
        logger.debug(f"Comparing gradients new vs memory")
        opt = self.lightning_module.optimizers()

        # NEW batch
        opt.zero_grad()  # Make sure no grads
        self.lightning_module.manual_backward(loss_new_actions_m, retain_graph=True)  # Calculate grad
        new_grad_dict = get_name_to_grad_dict(self.lightning_module.model)

        # MEM batch
        if not new_only:
            opt.zero_grad()  # Make sure no grads
            self.lightning_module.manual_backward(loss_mem_actions_m, retain_graph=True)  # Calculate grad
            mem_grad_dict = get_name_to_grad_dict(self.lightning_module.model)

        # Clear gradients again for later full backprop
        opt.zero_grad()

        # Now get for all subsets of grads the norm and cos-sim
        for log_name, grad_name_filters, filter_incl in [
            ("full", None, True),
            ("slow", ['pathway0'], True),
            ("fast", ['pathway1'], True),
            ("head", ['head'], True),
            ("feat", ['head'], False),  # all but head
        ]:
            # NEW results
            new_grad = grad_dict_to_vector(new_grad_dict, grad_name_filters, include_filter=filter_incl)
            new_grad_norm = new_grad.pow(2).sum().sqrt().item()  # Get L2 norm

            # Add results
            log_results[
                get_metric_tag(TAG_BATCH, train_mode='analyze', action_mode='action',
                               base_metric_name=f"{log_name}_new_grad_norm")
            ] = new_grad_norm

            # MEM results
            if not new_only:
                mem_grad = grad_dict_to_vector(mem_grad_dict, grad_name_filters, include_filter=filter_incl)
                mem_grad_norm = mem_grad.pow(2).sum().sqrt().item()  # Get L2 norm

                grad_cos_sim = (F.normalize(new_grad, p=2, dim=0) * F.normalize(mem_grad, p=2, dim=0)).sum().item()

                log_results[
                    get_metric_tag(TAG_BATCH, train_mode='analyze', action_mode='action',
                                   base_metric_name=f"{log_name}_grad_cos_sim")
                ] = grad_cos_sim

                log_results[
                    get_metric_tag(TAG_BATCH, train_mode='analyze', action_mode='action',
                                   base_metric_name=f"{log_name}_mem_grad_norm")
                ] = mem_grad_norm

    def training_update_loop(self, loss_first_fwd, inputs, labels, current_batch_stream_idxs: list, *args, **kwargs):
        """ Given the loss from the first forward, update for inner_loop_iters. """
        opt = self.lightning_module.optimizers()

        # Efficient loading for multi-iter (dataloading = bottleneck)
        if self.lightning_module.inner_loop_iters > 1 and self.resample_multi_iter:  # Prefetch data all at once
            n_sample_rounds = self.lightning_module.inner_loop_iters - 1  # First one already sampled on entering
            joined_batches = self.new_batch_to_joined_batches(self._current_new_batch, n_sample_rounds=n_sample_rounds)
            assert len(joined_batches) == n_sample_rounds

        assert self.lightning_module.inner_loop_iters >= 1
        for inner_iter in range(1, self.lightning_module.inner_loop_iters + 1):

            if inner_iter == 1:
                loss_action = loss_first_fwd

            else:
                # Resample from replay_strategies buffer: Reset inputs/labels
                if self.resample_multi_iter:  # inner iter >=2
                    joined_batch_idx = inner_iter - 2  # Skip first and one offset in inner_iter loop
                    joined_batch = joined_batches[joined_batch_idx]
                    joined_batch_inputs = joined_batch[0]
                    joined_batch_labels = joined_batch[1]

                else:
                    joined_batch_inputs = inputs
                    joined_batch_labels = labels

                fwd_inputs = copy.deepcopy(joined_batch_inputs)  # SlowFast in-place alters the inputs
                preds = self.lightning_module.forward(fwd_inputs, return_feats=False)
                loss_action, loss_verb, loss_noun = OnlineLossMetric.get_losses_from_preds(
                    preds, joined_batch_labels, self.loss_fun_train, mean=True
                )

            opt.zero_grad()  # Also clean grads for final
            self.lightning_module.manual_backward(loss_action)  # Calculate grad
            opt.step()
            logger.info(f"[INNER-LOOP UPDATE] iter {inner_iter}/{self.lightning_module.inner_loop_iters}: "
                        f"fwd, bwd, step. Action_loss={loss_action.item()}")

        # Store samples after updates
        self._store_samples_in_replay_memory(labels[:self.new_batch_size], current_batch_stream_idxs)

        # Update size
        self.num_samples_memory = sum(len(cond_mem) for cond_mem in self.conditional_memory.values())
        logger.info(f"[REPLAY] nb samples in memory = {self.num_samples_memory}/{self.total_mem_size}")
        assert self.num_samples_memory <= self.total_mem_size

    def _store_samples_in_replay_memory(self, current_batch_labels: torch.LongTensor, current_batch_stream_idxs: list):

        if self.storage_policy == 'reservoir':
            self.conditional_memory[None] = self.reservoir_sampling(
                self.conditional_memory.get(None, []),
                current_batch_stream_idxs,
                self.total_mem_size,
                self.num_observed_samples
            )

        elif self.storage_policy == 'hybrid_CBRS':
            self.reservoir_action_storage_policy(current_batch_labels, current_batch_stream_idxs)

        elif self.storage_policy == 'FIFO':
            min_current_batch_idx = min(current_batch_stream_idxs)
            self.conditional_memory[None] = list(range(
                max(0, min_current_batch_idx - self.total_mem_size),
                min_current_batch_idx)
            )

        else:
            raise ValueError()
        logger.info(f"[REPLAY] buffer = \n{pprint.pformat(self.conditional_memory)}")

    def reservoir_action_storage_policy(self, labels: torch.LongTensor, current_batch_stream_idxs: list):
        """ Memory is divided in equal memory bins per observed action.
        Each bin is populated using reservoir sampling when new samples for that bin are encountered.
        """
        label_batch_axis = 0
        assert labels.shape[0] == len(current_batch_stream_idxs) == self.new_batch_size

        # Init new actions
        for batch_idx, verbnoun_t in enumerate(torch.unbind(labels, dim=label_batch_axis)):
            action = verbnoun_to_action(*verbnoun_t.tolist())

            # Init
            if action not in self.conditional_memory:
                self.conditional_memory[action] = []
                self.conditional_full[action] = False
                self.conditional_num_observed_samples[action] = 0

        # If nb classes is >= nb memory samples, there is no use in the conditional memory (max 1 sample/class)
        # -> Use reservoir instead
        if len(self.conditional_memory) >= self.total_mem_size or None in self.conditional_memory:

            # Convert conditional memory once to single memory
            if len(self.conditional_memory) != 1 and None not in self.conditional_memory:
                all_memories_list = []
                for mem_idxs in self.conditional_memory.values():
                    all_memories_list.extend(mem_idxs)
                self.conditional_memory = OrderedDict({None: all_memories_list})

                logger.info(
                    f"Converted to regular reservoir sampling as more actions than memory samples "
                    f"({self.total_mem_size}): {self.conditional_memory}")

            # Do reservoir
            logger.info(
                f"Using regular reservoir sampling as more actions than memory samples "
                f"({self.total_mem_size}): {self.conditional_memory}")
            self.conditional_memory[None] = self.reservoir_sampling(
                self.conditional_memory[None],
                current_batch_stream_idxs,
                self.total_mem_size, self.num_observed_samples
            )
            assert len(self.conditional_memory[None]) <= self.total_mem_size
            return

        # Checks
        assert self.num_samples_memory == sum(len(cond_mem) for cond_mem in self.conditional_memory.values())
        assert len(self.conditional_num_observed_samples) == len(self.conditional_memory) == len(self.conditional_full), \
            f"Both conditional memories should have exact same number of actions"

        # Collect actions (label pairs) and count new ones
        batch_actions = []
        for batch_idx, verbnoun_t in enumerate(torch.unbind(labels, dim=label_batch_axis)):
            action = verbnoun_to_action(*verbnoun_t.tolist())
            batch_actions.append(action)
            current_batch_stream_idx = current_batch_stream_idxs[batch_idx]

            # Add observed samples for specific class
            self.conditional_num_observed_samples[action] += 1

            # If memory not full, add sample
            if self.num_samples_memory < self.total_mem_size:
                logger.info(f"Memory not full, adding current sample for action {action}")
                self.conditional_memory[action].append(current_batch_stream_idx)
                self.num_samples_memory += 1

            else:
                # Find max count class (define as full: cannot grow anymore)
                max_conditional = max(self.conditional_memory, key=lambda cond: len(self.conditional_memory[cond]))
                self.conditional_full[max_conditional] = True

                assert len(self.conditional_memory[max_conditional]) > 0, \
                    f"Empty max conditional '{max_conditional}' in: {self.conditional_memory}"

                if not self.conditional_full[action]:  # Select random of max to replace
                    logger.info(f"action {action} NOT FULL: "
                                f"Random replacing from max {max_conditional} with new action {action}")

                    # Randomly remove from max
                    num_in_max_conditional = len(self.conditional_memory[max_conditional])
                    replace_idx = random.randrange(0, num_in_max_conditional)  # [a,b[
                    self.conditional_memory[max_conditional] = \
                        self.conditional_memory[max_conditional][:replace_idx] + \
                        self.conditional_memory[max_conditional][replace_idx + 1:]

                    # Add to new class
                    self.conditional_memory[action].append(current_batch_stream_idx)

                    # Checks
                    assert len(self.conditional_memory[max_conditional]) == num_in_max_conditional - 1, \
                        f"Didn't remove sample from max action: {max_conditional}"

                else:  # When full class (was max at least once), do reservoir
                    logger.info(f"action {action} FULL: Reservoir sampling for its memory {self.conditional_memory}")
                    self.reservoir_sampling(
                        self.conditional_memory[action],
                        [current_batch_stream_idx],
                        mem_size_limit=len(self.conditional_memory[action]),  # Assume full
                        num_observed_samples=self.conditional_num_observed_samples[action]
                    )
        logger.info(f"REPLAY SUMMARY:\n"
                    f"conditional_memory={self.conditional_memory}\n"
                    f"conditional_num_observed_samples={self.conditional_num_observed_samples}\n"
                    f"conditional_full={self.conditional_full}\n"
                    f"total_mem_size={self.total_mem_size}")

    def reservoir_sampling(self, memory: list, new_stream_idxs: list,
                           mem_size_limit: int, num_observed_samples: int) -> list:
        """ Fill buffer if not full yet, otherwise replace with probability mem_size/num_observed_samples. """

        for new_stream_idx in new_stream_idxs:
            if len(memory) < mem_size_limit:  # Buffer not filled yet
                memory.append(new_stream_idx)
            else:  # Replace with probability mem_size/num_observed_samples
                rnd_idx = random.randrange(0, num_observed_samples)  # [a,b[
                if rnd_idx < mem_size_limit:  # Replace if sampled in memory, Prob = M / (b-a)
                    memory[rnd_idx] = new_stream_idx

        return memory

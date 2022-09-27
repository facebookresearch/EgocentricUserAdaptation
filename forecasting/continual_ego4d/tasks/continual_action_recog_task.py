import copy
import pprint

import torch
import wandb
from fvcore.nn.precise_bn import get_bn_modules
from collections import Counter

from collections import defaultdict
import numpy as np
from itertools import product
from tqdm import tqdm

from ego4d.evaluation import lta_metrics as metrics
from ego4d.utils import misc
from ego4d.models import losses
from ego4d.optimizers import lr_scheduler
from ego4d.utils import distributed as du
from ego4d.models import build_model
from continual_ego4d.datasets.continual_dataloader import construct_trainstream_loader, construct_predictstream_loader
import os.path as osp
import random
from continual_ego4d.metrics.standard_metrics import OnlineLossMetric
# import wandb
from pytorch_lightning.loggers import WandbLogger

from continual_ego4d.utils.meters import AverageMeter
from continual_ego4d.methods.build import build_method
from continual_ego4d.methods.method_callbacks import Method
from continual_ego4d.metrics.metric import get_metric_tag
from continual_ego4d.metrics.count_metrics import Metric, \
    SetCountMetric, TAG_BATCH, WindowedUniqueCountMetric, HistoryCountMetric
from continual_ego4d.metrics.metric import TAG_BATCH, TAG_PAST
from continual_ego4d.metrics.standard_metrics import OnlineTopkAccMetric, RunningAvgOnlineTopkAccMetric, \
    OnlineLossMetric, RunningAvgOnlineLossMetric
from continual_ego4d.metrics.adapt_metrics import OnlineAdaptationGainMetric, RunningAvgOnlineAdaptationGainMetric, \
    CumulativeOnlineAdaptationGainMetric
from continual_ego4d.metrics.future_metrics import GeneralizationTopkAccMetric, FWTTopkAccMetric, \
    GeneralizationLossMetric, FWTLossMetric
from continual_ego4d.metrics.past_metrics import ReexposureForgettingLossMetric, ReexposureForgettingAccMetric
from continual_ego4d.datasets.continual_action_recog_dataset import verbnoun_to_action, verbnoun_format
from continual_ego4d.utils.models import UnseenVerbNounMaskerHead
from pytorch_lightning.loggers import TensorBoardLogger
from continual_ego4d.utils.models import model_trainable_summary
import matplotlib.pyplot as plt

from pytorch_lightning.core import LightningModule
from typing import List, Tuple, Union, Any, Optional, Dict, Type
from ego4d.utils import logging

logger = logging.get_logger(__name__)
logging.get_logger('matplotlib.font_manager').disabled = True
logging.get_logger('PIL.PngImagePlugin').disabled = True


class PretrainState:
    """
    The pretrain state enables using the state for both dataset creation and StreamStateTracker creation.
    The both Ego4dContinualRecognition and StreamStateTracker depend on pretrain sets.
    """

    def __init__(self, pretrain_action_sets):
        # Prediction phase
        self.sample_idx_to_pretrain_loss = {}
        """ A dict containing a mapping of the stream sample index to the loss on the initial pretrain model.
        The dictionary is filled in the preprocessing predict phase before training."""

        # Pretraining stats
        # From JSON: {'ACTION_LABEL': {'name': "ACTION_NAME", 'count': "ACTION_COUNT"}}
        self.pretrain_verb_freq_dict = {
            verbnoun_format(a): a_dict['count'] for a, a_dict in pretrain_action_sets['verb_to_name_dict'].items()
        }
        self.pretrain_noun_freq_dict = {
            verbnoun_format(a): a_dict['count'] for a, a_dict in pretrain_action_sets['noun_to_name_dict'].items()
        }
        # Json format to tuple for actions
        self.pretrain_action_freq_dict = {
            verbnoun_to_action(*str(a).split('-')): a_dict['count']
            for a, a_dict in pretrain_action_sets['action_to_name_dict'].items()
        }
        """ Sets containing all the action/nouns/verbs from the pretraining phase. """


class StreamStateTracker:
    """
    Disentangles tracking stats from the stream, form the Lightning Module.
    This object can safely be shared as reference, as opposed to the LightningModule which is prone to
    recursion errors due to holding the model and optimizer.
    """

    def __init__(self, stream_loader, pretrain_state: PretrainState):

        # All attributes that will also be stored
        self.init_attrs_to_save(stream_loader, pretrain_state)

        # Gather attr names defined before
        self.attrs_to_dump = [attr_name for attr_name in vars(self).keys()]
        logger.info(f"{self.__class__.__name__} attributes included in dump: {self.attrs_to_dump}")

        # Attributes that are not saved
        self.init_transient_attrs()

    def init_attrs_to_save(self, stream_loader, pretrain_state: PretrainState):
        self.total_stream_sample_count = len(stream_loader.dataset)

        # Count-sets: Current stream
        self.user_verb_freq_dict = stream_loader.dataset.verb_freq_dict
        self.user_noun_freq_dict = stream_loader.dataset.noun_freq_dict
        self.user_action_freq_dict = stream_loader.dataset.action_freq_dict
        """ Counter dictionaries from the user stream for actions/verbs/nouns. """

        self.clip_5min_transition_idx_set = set(stream_loader.dataset.clip_5min_transition_idxs)
        self.parent_video_transition_idx_set = set(stream_loader.dataset.parent_video_transition_idxs)
        """ Current stream video and clip transitions. """

        self.sample_idx_to_action_list = stream_loader.dataset.sample_idx_to_action_list
        """ A list containing all actions in the full stream, the array index corresponds to the stream sample idx. """

        # Additional dump references
        self.dataset_all_entries_ordered = stream_loader.dataset.seq_input_list
        """ Keep entire dataset list, per sample we have the action, video_path, and additional meta data. """

        self.sample_to_batch_idx = [-1] * self.total_stream_sample_count
        """ Mapping index in the stream to which batch it belongs. """

        self.sample_idx_to_feat = [None] * self.total_stream_sample_count
        self.sample_idx_to_verb_pred = [None] * self.total_stream_sample_count
        self.sample_idx_to_noun_pred = [None] * self.total_stream_sample_count
        self.sample_idx_to_action_loss = [None] * self.total_stream_sample_count
        self.sample_idx_to_verb_loss = [None] * self.total_stream_sample_count
        self.sample_idx_to_noun_loss = [None] * self.total_stream_sample_count
        """ Per sample feat/prediction/loss. """

        # PRETRAIN STATES
        for name, val in vars(pretrain_state).items():
            setattr(self, name, val)
        """Add all pretraining states as attributes of the stream to save for dump."""

        # TODO for replay also track losses for new/mem?

    def init_transient_attrs(self):
        # Transient (not included in dump)
        # Current iteration State vars (single batch)
        self.batch_idx: int = -1  # Current batch idx
        self.is_parent_video_transition: bool = False
        self.is_clip_5min_transition: bool = False
        self.eval_this_step: bool = False
        self.plot_this_step: bool = False
        self.stream_batch_sample_idxs: list = []
        self.stream_batch_size: int = 0  # Size of the new data batch sampled from the stream (exclusive replay samples)
        self.stream_batch_labels: torch.Tensor = None  # Ref for Re-exposure based forgetting
        self.batch_action_freq_dict: dict = {}
        self.batch_verb_freq_dict: dict = {}
        self.batch_noun_freq_dict: dict = {}
        """ Variables set per iteration to share between methods. """

        self.stream_next_batch_sample_idxs: list[int] = []
        self.stream_next_batch_labelset: set = set()
        """ Next batch variables for look-ahead in Forgetting exps. """

        # Store vars of observed part of stream (Don't reassign, use ref)
        self.seen_samples_idxs = []
        self.stream_seen_action_freq_dict: dict = {}
        self.stream_seen_verb_freq_dict: dict = {}
        self.stream_seen_noun_freq_dict: dict = {}
        self.seen_action_to_stream_idxs = defaultdict(list)  # On-the-fly: For each action keep observed stream ids
        """ Summarize observed part of the stream. """

    def get_state_dump(self):
        dump_dict = {attr_name: getattr(self, attr_name) for attr_name in self.attrs_to_dump}
        return dump_dict

    def set_current_batch_states(
            self,
            batch: Any,
            batch_idx: int,
            plotting_log_freq: int,
            continual_eval_freq: int,
    ):
        self.batch_idx = batch_idx

        # Eval or plot at this iteration
        if plotting_log_freq <= 0:
            self.plot_this_step = False
        else:
            self.plot_this_step = batch_idx != 0 and (
                    batch_idx % plotting_log_freq == 0 or batch_idx == self.total_stream_sample_count)

        if continual_eval_freq <= 0:
            self.eval_this_step = False
        else:
            self.eval_this_step = batch_idx % continual_eval_freq == 0 or batch_idx == self.total_stream_sample_count
            logger.debug(f"Continual eval on batch {batch_idx}/{self.total_stream_sample_count}={self.eval_this_step}")

        # Observed idxs update before batch is altered
        _, labels, _, stream_sample_idxs = batch
        self.stream_batch_labels = labels
        self.stream_batch_sample_idxs = stream_sample_idxs.tolist()
        self.stream_batch_size = len(self.stream_batch_sample_idxs)

        # Next batch idxs (for look-ahead)
        next_batch_start_idx = max(self.stream_batch_sample_idxs) + 1
        self.stream_next_batch_sample_idxs = [
            idx for idx in range(
                min(self.total_stream_sample_count - 1, next_batch_start_idx),
                min(self.total_stream_sample_count, next_batch_start_idx + self.stream_batch_size)
            )
        ]
        self.stream_next_batch_labelset = set({
            self.sample_idx_to_action_list[idx] for idx in self.stream_next_batch_sample_idxs
        })

        self.is_parent_video_transition = sum(
            entry_idx in self.parent_video_transition_idx_set
            for entry_idx in self.stream_batch_sample_idxs
        ) > 0

        self.is_clip_5min_transition = sum(
            entry_idx in self.clip_5min_transition_idx_set
            for entry_idx in self.stream_batch_sample_idxs
        ) > 0

        # Get new actions/verbs current batch
        self.batch_action_freq_dict = defaultdict(int)
        self.batch_verb_freq_dict = defaultdict(int)
        self.batch_noun_freq_dict = defaultdict(int)
        for ((verb, noun), sample_idx) in zip(labels.tolist(), self.stream_batch_sample_idxs):
            action = verbnoun_to_action(verb, noun)
            self.batch_action_freq_dict[action] += 1
            self.batch_verb_freq_dict[verbnoun_format(verb)] += 1
            self.batch_noun_freq_dict[verbnoun_format(noun)] += 1

        logger.debug(f"current_batch_sample_idxs={self.stream_batch_sample_idxs}")

    def update_stream_seen_state(self, labels, batch_idx):
        self.add_counter_dicts_(self.stream_seen_action_freq_dict, self.batch_action_freq_dict)
        self.add_counter_dicts_(self.stream_seen_verb_freq_dict, self.batch_verb_freq_dict)
        self.add_counter_dicts_(self.stream_seen_noun_freq_dict, self.batch_noun_freq_dict)

        # Only iterate stream batch (not replay samples)
        for ((verb, noun), sample_idx) in zip(labels.tolist(), self.stream_batch_sample_idxs):
            action = verbnoun_to_action(verb, noun)
            self.sample_to_batch_idx[sample_idx] = batch_idx  # Possibly add multiple time batch_idx
            self.seen_action_to_stream_idxs[action].append(sample_idx)

        # Update Task states
        self.seen_samples_idxs.extend(self.stream_batch_sample_idxs)
        assert len(self.seen_samples_idxs) == len(np.unique(self.seen_samples_idxs)), \
            f"Duplicate visited samples in {self.seen_samples_idxs}"

    @staticmethod
    def add_counter_dicts_(source_dict: dict, new_dict: dict):
        for k, cnt in new_dict.items():
            if k not in source_dict:
                source_dict[k] = 0
            source_dict[k] += cnt


class ContinualMultiTaskClassificationTask(LightningModule):
    """
    Training mode: Visit samples in stream sequentially, update, and evaluate per update.
    Validation mode: Disabled.
    Test mode: Disabled (No held-out test sets available for online learning)
    Predict mode: Visit the stream and collect per-sample stats such as the loss. No learning is performed.

    For all lightning hooks, see: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
    """

    def __init__(self, cfg):
        """

        !Warning: In __init__ the self.device='cpu' always! After init it is set to the right device, see:
        https://github.com/Lightning-AI/lightning/issues/2638
        Als in setup() the device is still cpu, this is a PL bug.
        https://github.com/Lightning-AI/lightning/issues/13108

        :param cfg:
        :param future_metrics:
        :param past_metrics:
        :param shuffle_stream:
        """
        logger.debug('Starting init ContinualVideoTask')
        super().__init__()

        # Disable automatic updates after training_step(), instead do manually
        # We need to do this for alternative update schedules (e.g. multiple iters per batch)
        # See: https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html#automatic-optimization
        self.automatic_optimization = False

        # Backwards compatibility.
        if isinstance(cfg.MODEL.NUM_CLASSES, int):
            cfg.MODEL.NUM_CLASSES = [cfg.MODEL.NUM_CLASSES]

        if not hasattr(cfg.TEST, "NO_ACT"):
            logger.info("Default NO_ACT")
            cfg.TEST.NO_ACT = False

        if not hasattr(cfg.MODEL, "TRANSFORMER_FROM_PRETRAIN"):
            cfg.MODEL.TRANSFORMER_FROM_PRETRAIN = False

        if not hasattr(cfg.MODEL, "STRIDE_TYPE"):
            cfg.EPIC_KITCHEN.STRIDE_TYPE = "constant"

        # CFG checks
        self.cfg = cfg
        self.save_hyperparameters(logger=False)  # Save cfg to hparams file, don't send to

        # Multi-task (verb/noun) has classification head, mask out unseen classifier prototype outputs
        self.model = build_model(cfg)

        # Always use unreduced loss function
        self.loss_fun_unred = losses.get_loss_func(self.cfg.MODEL.LOSS_FUNC)(reduction="none")  # Training/ prediction

        self.continual_eval_freq = cfg.CONTINUAL_EVAL.FREQ
        self.plotting_log_freq = cfg.CONTINUAL_EVAL.PLOTTING_FREQ
        self.inner_loop_iters = cfg.TRAIN.INNER_LOOP_ITERS
        assert isinstance(self.inner_loop_iters, int) and self.inner_loop_iters >= 1

        # Sanity modes/debugging
        self.enable_prepost_comparing = cfg.CHECK_POST_VS_PRE_LOSS_DELTA  # Compare loss before/after update

        # Pretrain state
        self.pretrain_state = PretrainState(cfg.COMPUTED_PRETRAIN_ACTION_SETS)
        self.cfg.COMPUTED_PRETRAIN_STATE = self.pretrain_state  # Set for dataset creation

        # Dataloader
        self.train_loader = construct_trainstream_loader(self.cfg, shuffle=False)

        # State tracker of stream
        self.stream_state = StreamStateTracker(self.train_loader, self.pretrain_state)

        # Pretrain loader: Only samples that both verb and noun have been seen in pretrain
        self.predict_phase_load_idxs = []

        for stream_sample_idx, action in enumerate(self.stream_state.sample_idx_to_action_list):
            verb, noun = action
            if verb in self.pretrain_state.pretrain_verb_freq_dict \
                    and noun in self.pretrain_state.pretrain_noun_freq_dict:
                self.predict_phase_load_idxs.append(stream_sample_idx)

        if not self.cfg.ENABLE_FEW_SHOT:
            assert len(self.predict_phase_load_idxs) == len(self.stream_state.sample_idx_to_action_list), \
                f"Should load all idxs for prediction phase in non-few-shot mode, as all actions are seen in pretrain."

        self.predict_loader = construct_predictstream_loader(
            self.train_loader, self.cfg, subset_idxes=self.predict_phase_load_idxs)

        # Predict phase:
        # If we first run predict phase, we can fill this dict with the results, this can then be used in trainphase
        self.run_predict_before_train = True
        """ Triggers running prediction phase before starting training on the stream. 
        This allows preprocessing on the entire stream (e.g. collect pretraining losses and stream stats)."""

        # For data stream info dump. Is used as token for finishing the job
        self.dumpfile = self.cfg.COMPUTED_USER_DUMP_FILE

        # Stream samplers
        self.future_stream_sampler = FutureSampler(mode='FIFO_split_seen_unseen',
                                                   stream_idx_to_action_list=self.stream_state.sample_idx_to_action_list,
                                                   seen_action_set=self.stream_state.stream_seen_action_freq_dict,
                                                   total_capacity=self.cfg.CONTINUAL_EVAL.FUTURE_SAMPLE_CAPACITY)
        self.past_stream_sampler = PastSampler(mode=cfg.CONTINUAL_EVAL.PAST_SAMPLER_MODE,
                                               seen_action_to_stream_idxs=self.stream_state.seen_action_to_stream_idxs,
                                               total_capacity=self.cfg.CONTINUAL_EVAL.PAST_SAMPLE_CAPACITY)
        """ Samplers to process the future and past part of the stream. """

        # Method
        self.method: Method = build_method(cfg, self)

        # Metrics
        self.batch_metric_results = {}  # Stateful share metrics between different phases so can be reused

        self.current_batch_metrics = self._get_current_batch_metrics()
        self.future_metrics = []  # Empty metric-list skips future eval
        self.past_metrics = self._get_past_metrics()
        self.all_metrics = [*self.current_batch_metrics, *self.future_metrics, *self.past_metrics]

        logger.debug(f'Initialized {self.__class__.__name__}')

    # ---------------------
    # METRICS
    # ---------------------
    def _get_current_batch_metrics(self):
        batch_metrics = []

        for mode in ['verb', 'noun', 'action']:
            batch_metrics.extend([

                # LOSS/ACC
                OnlineTopkAccMetric(TAG_BATCH, k=1, mode=mode),
                RunningAvgOnlineTopkAccMetric(TAG_BATCH, k=1, mode=mode),
                # OnlineLossMetric -> Standard included for training
                RunningAvgOnlineLossMetric(TAG_BATCH, loss_fun=self.loss_fun_unred, mode=mode),

                # ADAPT METRICS
                OnlineAdaptationGainMetric(
                    TAG_BATCH, self.loss_fun_unred, self.pretrain_state.sample_idx_to_pretrain_loss, loss_mode=mode),
                RunningAvgOnlineAdaptationGainMetric(
                    TAG_BATCH, self.loss_fun_unred, self.pretrain_state.sample_idx_to_pretrain_loss, loss_mode=mode),
                CumulativeOnlineAdaptationGainMetric(
                    TAG_BATCH, self.loss_fun_unred, self.pretrain_state.sample_idx_to_pretrain_loss, loss_mode=mode),
            ])

            # TOP5-ACC
            if mode in ['verb', 'noun']:
                batch_metrics.extend([
                    OnlineTopkAccMetric(TAG_BATCH, k=5, mode=mode),
                    RunningAvgOnlineTopkAccMetric(TAG_BATCH, k=5, mode=mode)
                ])

            # Window-counts
            for window_size in [10, 100]:
                batch_metrics.append(
                    WindowedUniqueCountMetric(
                        preceding_window_size=window_size,
                        sample_idx_to_action_list=self.stream_state.sample_idx_to_action_list,
                        action_mode=mode)
                )

        # ADD INTERSECTION COUNT METRICS
        for mode, seen_set, user_ref_set, pretrain_ref_set in [
            ('action', self.stream_state.stream_seen_action_freq_dict, self.stream_state.user_action_freq_dict,
             self.pretrain_state.pretrain_action_freq_dict),
            ('verb', self.stream_state.stream_seen_verb_freq_dict, self.stream_state.user_verb_freq_dict,
             self.pretrain_state.pretrain_verb_freq_dict),
            ('noun', self.stream_state.stream_seen_noun_freq_dict, self.stream_state.user_noun_freq_dict,
             self.pretrain_state.pretrain_noun_freq_dict),
        ]:
            batch_metrics.extend([  # Seen actions (history part of stream) vs full user stream actions
                SetCountMetric(observed_set_name="seen", observed_set=seen_set,
                               ref_set_name="stream", ref_set=user_ref_set,
                               mode=mode
                               ),
                # Seen actions (history part of stream) vs all actions seen during pretraining phase
                SetCountMetric(observed_set_name="seen", observed_set=seen_set,
                               ref_set_name="pretrain", ref_set=pretrain_ref_set,
                               mode=mode
                               ),
            ])

        # ADD HISTORY VS CURRENT BATCH COUNTS:
        # how many times seen in pretrain vs how many times during stream
        for mode, history_action_instance_count, pretrain_action_instance_count in [
            ('action', self.stream_state.stream_seen_action_freq_dict, self.pretrain_state.pretrain_action_freq_dict),
            ('verb', self.stream_state.stream_seen_verb_freq_dict, self.pretrain_state.pretrain_verb_freq_dict),
            ('noun', self.stream_state.stream_seen_noun_freq_dict, self.pretrain_state.pretrain_noun_freq_dict),
        ]:
            batch_metrics.extend([
                HistoryCountMetric(history_action_instance_count=history_action_instance_count,
                                   action_mode=mode,
                                   pretrain_action_instance_count=pretrain_action_instance_count,
                                   ),
                # Don't include pretrain counts
                HistoryCountMetric(history_action_instance_count=history_action_instance_count,
                                   action_mode=mode,
                                   pretrain_action_instance_count=None,
                                   ),
                # Don't include action counts
                HistoryCountMetric(history_action_instance_count=None,
                                   action_mode=mode,
                                   pretrain_action_instance_count=pretrain_action_instance_count,
                                   ),
            ])

        return batch_metrics

    def _get_past_metrics(self):
        past_metrics = []

        # PAST ADAPTATION METRICS
        for action_mode in ['action', 'verb', 'noun']:
            past_metrics.extend([
                # ADAPTATION METRICS
                OnlineAdaptationGainMetric(
                    TAG_PAST, self.loss_fun_unred, self.pretrain_state.sample_idx_to_pretrain_loss,
                    loss_mode=action_mode),
                RunningAvgOnlineAdaptationGainMetric(
                    TAG_PAST, self.loss_fun_unred, self.pretrain_state.sample_idx_to_pretrain_loss,
                    loss_mode=action_mode),
                CumulativeOnlineAdaptationGainMetric(
                    TAG_PAST, self.loss_fun_unred, self.pretrain_state.sample_idx_to_pretrain_loss,
                    loss_mode=action_mode),

                # ACTION METRICS
                OnlineTopkAccMetric(TAG_PAST, k=1, mode=action_mode),
                OnlineLossMetric(TAG_PAST, loss_fun=self.loss_fun_unred, mode=action_mode),

            ])
            if self.continual_eval_freq == 1:
                past_metrics.extend([
                    ReexposureForgettingLossMetric(loss_fun_unred=self.loss_fun_unred, action_mode=action_mode),
                    ReexposureForgettingAccMetric(k=1, action_mode=action_mode),
                ])
                if action_mode in ['verb', 'noun']:  # Only k=1 for action mode
                    past_metrics.extend([
                        ReexposureForgettingAccMetric(k=5, action_mode=action_mode),
                        ReexposureForgettingAccMetric(k=20, action_mode=action_mode),
                    ])

        return past_metrics

    # ---------------------
    # DECORATORS
    # ---------------------
    def _eval_in_train_decorator(fn):
        """ Decorator for evaluation. """

        def parent_fn(self, *args, **kwargs):
            # SET TO EVAL MODE
            self.model.train(False)
            torch.set_grad_enabled(False)

            fn(self, *args, **kwargs)

            # SET BACK TO TRAIN MODE
            self.model.train()
            torch.set_grad_enabled(True)

        return parent_fn

    # ---------------------
    # TRAINING FLOW CALLBACKS
    # ---------------------
    def _init_training_step(self, batch: Any, batch_idx: int) -> Any:
        # Reset metrics
        self.batch_metric_results = {}
        for metric in self.all_metrics:
            if metric.reset_before_batch:
                metric.reset()

        # Set state and log
        self.stream_state.set_current_batch_states(
            batch,
            batch_idx=batch_idx,
            plotting_log_freq=self.plotting_log_freq,
            continual_eval_freq=self.continual_eval_freq,
        )
        self._log_current_batch_states(batch_idx=batch_idx)

        # add model trainable params
        (train_p, total_p), (head_train_p, head_total_p) = model_trainable_summary(self.model)
        self.add_to_dict_(self.batch_metric_results, {
            get_metric_tag(TAG_BATCH, base_metric_name=f"model_trainable_params"): train_p,
            get_metric_tag(TAG_BATCH, base_metric_name=f"model_all_params"): total_p,
            get_metric_tag(TAG_BATCH, base_metric_name=f"head_trainable_params"): head_train_p,
            get_metric_tag(TAG_BATCH, base_metric_name=f"head_all_params"): head_total_p,
        })

        # Update batch
        altered_batch = self.method.train_before_update_batch_adapt(batch, batch_idx)
        return altered_batch

    def training_step(self, batch, batch_idx):
        """ Before update: Forward and define loss. """
        batch = self._init_training_step(batch, batch_idx)

        # PREDICTIONS + LOSS
        inputs, labels, video_names, _ = batch

        # Do method callback (Get losses etc)
        # This is the loss used for updates and the outputs of the entire batch.
        # Note that for metrics: e.g. replay we don't want to use all outputs, but only the new ones from the stream
        fwd_inputs = inputs
        if self.enable_prepost_comparing:  # Make copy as inputs are adapted in-place in SlowFast
            fwd_inputs = [inputs[i].clone() for i in range(len(inputs))]

        loss, verbnoun_outputs, step_results = self.method.training_step(
            fwd_inputs, labels, self.stream_state.stream_batch_sample_idxs
        )
        self.add_to_dict_(self.batch_metric_results, step_results)

        logger.debug(f"Starting PRE-UPDATE evaluation: batch_idx={batch_idx}/{len(self.train_loader)}")
        self.eval_current_stream_batch_preupdate_(
            self.batch_metric_results,
            [verbnoun_outputs[i][:self.stream_state.stream_batch_size] for i in range(2)],
            labels[:self.stream_state.stream_batch_size]
        )
        # Perform additional eval
        if self.stream_state.eval_this_step:
            self.eval_future_data_(self.batch_metric_results, batch_idx)

        # Only loss should be used and stored for entire epoch (stream)
        logger.debug(f"Finished training_step batch_idx={batch_idx}/{len(self.train_loader)}")

        # On the end of the step, do update
        self._manual_update(loss)
        # return loss

    def _manual_update(self, loss):
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        logger.info(f"[INNER-LOOP UPDATE] FINAL iter {self.inner_loop_iters}/{self.inner_loop_iters}: "
                    f"action_loss={loss.item()}")

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        """
        Past samples should always be evaluated AFTER update step. The model is otherwise just
        updated on the latest batch in history it was just updated on (=pre-update model of the current batch).
        """
        inputs, labels, video_names, _ = batch

        # Measure difference of pre-update results of current batch (e.g. forward second time)
        if self.enable_prepost_comparing:
            self.eval_current_stream_batch_postupdate_(self.batch_metric_results, batch)

        # Do post-update evaluation of the past
        if self.stream_state.eval_this_step:
            logger.debug(f"Starting POST-UPDATE evaluation on batch_idx={batch_idx}/{len(self.train_loader)}")
            self.eval_past_data_(self.batch_metric_results, batch_idx)

            # (optionally) Save metrics after batch
            for metric in self.all_metrics:
                metric.save_result_to_history(current_batch_idx=batch_idx)

        # Update counts etc
        self.stream_state.update_stream_seen_state(labels, batch_idx)

        # Plot metrics if possible
        if self.stream_state.plot_this_step:
            self._log_plotting_metrics()

        # LOG results
        self.log_step_metric_results(self.batch_metric_results)
        logger.debug(f"Results for batch_idx={batch_idx}/{len(self.train_loader)}: "
                     f"{pprint.pformat(self.batch_metric_results)}")

    def on_train_end(self) -> None:
        """Dump any additional stats about the training."""
        wandb_logger: WandbLogger = self.get_logger_instance(WandbLogger)
        assert wandb_logger is not None, "Must have wandb logger to finish run!"

        dump_dict = self.stream_state.get_state_dump()

        # Gather states from metrics
        for metric in self.all_metrics:
            metric_dict = metric.dump()
            if metric_dict is not None and len(metric_dict) > 0:
                self.add_to_dict_(dump_dict, metric_dict)

        torch.save(dump_dict, self.dumpfile)
        # wandb_logger.experiment.log({'dump': dump_dict}) # Error, keys must be str, int, float, bool or None, not tuple
        logger.debug(f"Logged stream info to dumpfile {self.dumpfile}")

        # Let WandB logger know that run is fully executed
        wandb_logger.experiment.log({"finished_run": True})

    # ---------------------
    # PER-STEP EVALUATION
    # ---------------------
    def log_step_metric_results(self, log_dict):
        for logname, logval in log_dict.items():
            self.log(logname, float(logval), on_step=True, on_epoch=False)

    def _log_current_batch_states(self, batch_idx: int):
        """ Log stats for current iteration stream. """
        # Before-update counts
        metric_results = {
            get_metric_tag(TAG_BATCH, base_metric_name=f"history_sample_count"):
                len(self.stream_state.seen_samples_idxs),
            get_metric_tag(TAG_BATCH, base_metric_name=f"future_sample_count"):
                self.stream_state.total_stream_sample_count - len(self.stream_state.seen_samples_idxs),
            get_metric_tag(TAG_BATCH, base_metric_name=f"is_parent_video_transition"):
                self.stream_state.is_parent_video_transition,
            get_metric_tag(TAG_BATCH, base_metric_name=f"is_clip_5min_transition"):
                self.stream_state.is_clip_5min_transition,
        }

        self.add_to_dict_(self.batch_metric_results, metric_results)
        logger.debug(f"batch_metric_results: Added state of batch_idx={batch_idx}/{len(self.train_loader)}")

    @torch.no_grad()
    @_eval_in_train_decorator
    def eval_current_stream_batch_preupdate_(self, step_result, verbnoun_outputs, labels):
        """Add additional metrics for current batch in-place to the step_result dict."""
        logger.debug(f"Gathering online results")
        assert verbnoun_outputs[0].shape[0] == verbnoun_outputs[1].shape[0], \
            "Verbs and nouns output dims should be equal"
        assert verbnoun_outputs[0].shape[0] == labels.shape[0], \
            "Batch dim for input and label should be equal"
        assert verbnoun_outputs[0].shape[0] == self.stream_state.stream_batch_size, \
            "Eval on current batch should only contain new stream samples. Not samples from altered batch"

        # Update metrics
        for metric in self.current_batch_metrics:
            metric.update(verbnoun_outputs, labels, self.stream_state.stream_batch_sample_idxs,
                          stream_state=self.stream_state)

        # Gather results from metrics
        results = {}
        for metric in self.current_batch_metrics:
            results = {**results, **metric.result(self.stream_state.batch_idx)}

        self.add_to_dict_(step_result, results)

    @torch.no_grad()
    @_eval_in_train_decorator
    def eval_current_stream_batch_postupdate_(self, step_result, current_batch):
        """
        Measure difference of pre-update results of current batch (e.g. forward second time)
        Again we only measure on stream data, not potential replay data.
        """
        full_slowfast_inputs, full_labels, _, _ = current_batch
        assert isinstance(full_slowfast_inputs, list) and len(full_slowfast_inputs) == 2, \
            "Only implemented for slowfast model"

        # Make sure no replay data is considered
        stream_inputs = [full_slowfast_inputs[i][:self.stream_state.stream_batch_size]
                         for i in range(len(full_slowfast_inputs))]
        stream_labels = full_labels[:self.stream_state.stream_batch_size]

        post_update_preds = self.forward(stream_inputs)

        post_loss_action, post_loss_verb, post_loss_noun = OnlineLossMetric.get_losses_from_preds(
            post_update_preds, stream_labels, self.loss_fun_unred, mean=True
        )

        # Deltas
        pre_loss_action = self.batch_metric_results[
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='action', base_metric_name='loss')
        ]
        pre_loss_verb = self.batch_metric_results[
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='verb', base_metric_name='loss')
        ]
        pre_loss_noun = self.batch_metric_results[
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='noun', base_metric_name='loss')
        ]

        results = {
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='action', base_metric_name='post-pre_loss'):
                float(post_loss_action - pre_loss_action),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='verb', base_metric_name='post-pre_loss'):
                float(post_loss_verb - pre_loss_verb),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='noun', base_metric_name='post-pre_loss'):
                float(post_loss_noun - pre_loss_noun),
        }

        self.add_to_dict_(step_result, results)

    @torch.no_grad()
    @_eval_in_train_decorator
    def eval_future_data_(self, step_result, batch_idx):
        """Add additional metrics for future data (including current pre-update batch)
        in-place to the step_result dict."""
        if len(self.future_metrics) == 0:
            return
        if batch_idx == len(self.train_loader):  # last batch
            logger.debug(f"Skipping results on future data for last batch")
            return
        logger.debug(f"Gathering results on future data")

        # Include current batch
        all_future_idxs = list(range(min(self.stream_state.stream_batch_sample_idxs), len(self.train_loader.dataset)))
        sampled_future_idxs = self.future_stream_sampler(all_future_idxs)
        logger.debug(f"SAMPLED {len(sampled_future_idxs)} from all_future_idxs interval = "
                     f"[{all_future_idxs[0]},..., {all_future_idxs[-1]}]")

        # Create new dataloader
        future_dataloader = self._get_train_dataloader_subset(
            self.train_loader,
            subset_indices=sampled_future_idxs,  # Future data, including current
            batch_size=self.cfg.CONTINUAL_EVAL.BATCH_SIZE,
            num_workers=self.cfg.CONTINUAL_EVAL.NUM_WORKERS,
            pin_memory=self.cfg.DATA_LOADER.PIN_MEMORY,
        )

        result_dict = self._get_metric_results_over_dataloader(future_dataloader, metrics=self.future_metrics)
        self.add_to_dict_(step_result, result_dict)

    @torch.no_grad()
    @_eval_in_train_decorator
    def eval_past_data_(self, step_result, batch_idx):
        if len(self.past_metrics) == 0:
            return
        if batch_idx == 0:  # first batch
            logger.debug(f"Skipping first batch past data results (no observed data yet)")
            return
        logger.debug(f"Gathering results on past data")

        all_past_idxs = np.unique(self.stream_state.seen_samples_idxs).tolist()
        sampled_past_idxs = self.past_stream_sampler(all_past_idxs)
        logger.debug(f"SAMPLED {len(sampled_past_idxs)} from all_past_idxs interval = "
                     f"[{all_past_idxs[0]},..., {all_past_idxs[-1]}]")

        past_dataloader = self._get_train_dataloader_subset(
            self.train_loader,
            subset_indices=sampled_past_idxs,  # Previous data, not including current
            batch_size=self.cfg.CONTINUAL_EVAL.BATCH_SIZE,
            num_workers=self.cfg.CONTINUAL_EVAL.NUM_WORKERS,
            pin_memory=self.cfg.DATA_LOADER.PIN_MEMORY,
        )
        result_dict = self._get_metric_results_over_dataloader(past_dataloader, metrics=self.past_metrics)
        self.add_to_dict_(step_result, result_dict)

    def _log_plotting_metrics(self):
        """ Iterate over metrics to get Image plots. """

        logger.info("Collecting figures for metric plots")
        plot_dict = {}
        for metric in self.all_metrics:
            metric_plot_dict = metric.plot()
            if metric_plot_dict is not None and len(metric_plot_dict) > 0:
                self.add_to_dict_(plot_dict, metric_plot_dict)

        # Collect loggers
        tb_logger = self.get_logger_instance(TensorBoardLogger)
        wandb_logger = self.get_logger_instance(WandbLogger)

        # Log them
        logger.info("Plotting tensorboard figures")
        for name, mpl_figure in plot_dict.items():
            tb_logger.experiment.add_figure(
                tag=name, figure=mpl_figure
            )
            wandb_logger.experiment.log({name: wandb.Image(mpl_figure)})
        plt.close('all')

    # ---------------------
    # HELPER METHODS
    # ---------------------
    def get_logger_instance(
            self,
            logger_type: Union[Type[WandbLogger], Type[TensorBoardLogger]]) -> \
            Union[TensorBoardLogger, WandbLogger, None]:
        """ Get specific result logger from trainer. """
        result_loggers = [result_logger for result_logger in self.logger if isinstance(result_logger, logger_type)]
        if len(result_loggers) == 0:
            logger.info(f"No {logger_type.__class__.__name__} logger found, skipping image plotting.")
            return None
        elif len(result_loggers) > 1:
            raise Exception(
                f"Multiple {logger_type.__class__.__name__} loggers found, should only define one: {result_loggers}")
        return result_loggers[0]

    @staticmethod
    def add_to_dict_(source_dict: dict, dict_to_add: dict, key_exist_ok=False):
        """In-place add to dict"""
        for k, v in dict_to_add.items():
            if not key_exist_ok and k in source_dict:
                raise ValueError(f'dict_to_add is overwriting source_dict, existing key={k}')
            source_dict[k] = v

    def _get_train_dataloader_subset(self, train_dataloader: torch.utils.data.DataLoader,
                                     subset_indices: Union[List, Tuple],
                                     batch_size: int,
                                     num_workers: int,
                                     pin_memory: bool,
                                     ):
        """ Get a subset of the training dataloader's dataset.

        !Warning!: DONT COPY SAMPLER from train_dataloader to new dataloader as __len__ is re-used
        from the parent train_dataloader in the new dataloader (which may not match the Dataset).
        """
        dataset = train_dataloader.dataset

        if subset_indices is not None:
            dataset = torch.utils.data.Subset(dataset, indices=subset_indices)

        batch_size = min(len(dataset), batch_size)  # Effective batch size

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=None,
        )
        return loader

    @torch.no_grad()
    def _get_metric_results_over_dataloader(self, dataloader, metrics: List[Metric]):

        # Update metrics over dataloader data
        logger.debug(f"Iterating dataloader and transferring to device: {self.device}")
        for batch_idx, (inputs, labels, _, stream_sample_idxs) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Slowfast inputs (list):
            # inputs[0].shape = torch.Size([32, 3, 8, 224, 224]) -> Slow net
            # inputs[1].shape =  torch.Size([32, 3, 32, 224, 224]) -> Fast net

            labels = labels.to(self.device)
            inputs = [x.to(self.device) for x in inputs] if isinstance(inputs, list) \
                else inputs.to(self.device)
            preds = self.forward(inputs)

            for metric in metrics:
                metric.update(
                    preds, labels, stream_sample_idxs.tolist(),
                    stream_state=self.stream_state
                )

        # Gather results
        avg_metric_result_dict = {}
        for metric in metrics:
            avg_metric_result_dict = {**avg_metric_result_dict, **metric.result(self.stream_state.batch_idx)}

        return avg_metric_result_dict

    # ---------------------
    # PREDICTION FLOW CALLBACKS
    # ---------------------
    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> dict:
        """ Collect per-sample stats such as the loss.
        The returned values allow acces to the values through
        list_of_predictions = trainer.predict()
        """
        inputs, labels, video_names, stream_sample_idxs = batch

        # Loss per sample
        _, _, sample_to_results = self.method.prediction_step(inputs, labels, stream_sample_idxs.tolist())
        for k, v in sample_to_results.items():
            self.pretrain_state.sample_idx_to_pretrain_loss[k] = v

        return sample_to_results

    @torch.no_grad()
    def on_predict_end(self) -> None:
        """ Get uniform (max entropy) classifier predictions over seen verbs/nouns in pretraining.
        As these prototypes are unseen during pretrain, we need a proxy to calculate the delta in Adaptation Gain.
        The proxy baseline performance we choose is this from classifier always predicting a uniform distribution,
        or a maximum entropy classifier.
        We only consider the pretraining prototypes of the pretraining phase for the uniform predictor.
        As later adaptation with incremental classes is also seen as an improvement over the original pretrain model.
        """
        logger.info(f"Predict collected over stream: {pprint.pformat(self.pretrain_state.sample_idx_to_pretrain_loss)}")

        nb_verbs_total, nb_nouns_total = self.cfg.MODEL.NUM_CLASSES  # [ 115, 478 ]

        # retrieve how many seen during pretrain
        nb_verbs_seen_pretrain = len(self.pretrain_state.pretrain_verb_freq_dict)
        nb_nouns_seen_pretrain = len(self.pretrain_state.pretrain_noun_freq_dict)
        logger.info(f"Pretraining seen verbs = {nb_verbs_seen_pretrain}/{nb_verbs_total}, "
                    f"nouns= {nb_nouns_seen_pretrain}/{nb_nouns_total}")

        if self.cfg.ENABLE_FEW_SHOT:  # Reference uniform distr for unseen classes in pretrain for AG metric.
            # Random label in uniform prediction distribution
            gt = torch.tensor([0])  # Random label
            verb_uniform_preds = torch.zeros(1, nb_verbs_seen_pretrain).fill_(1 / nb_verbs_seen_pretrain)
            noun_uniform_preds = torch.zeros(1, nb_nouns_seen_pretrain).fill_(1 / nb_nouns_seen_pretrain)

            loss_verb_uniform = self.method.loss_fun_train(verb_uniform_preds, gt).item()
            loss_noun_uniform = self.method.loss_fun_train(noun_uniform_preds, gt).item()
            loss_action_uniform = loss_verb_uniform + loss_noun_uniform

            # Iterate over samples that have not been seen before
            unseen_in_pretrain_idxs = [idx for idx in range(self.stream_state.total_stream_sample_count)
                                       if idx not in self.predict_phase_load_idxs]
            for unseen_in_pretrain_idx in unseen_in_pretrain_idxs:
                self.pretrain_state.sample_idx_to_pretrain_loss[unseen_in_pretrain_idx] = {
                    get_metric_tag(TAG_BATCH, train_mode='pred', action_mode='action', base_metric_name='loss'):
                        loss_action_uniform,
                    get_metric_tag(TAG_BATCH, train_mode='pred', action_mode='verb', base_metric_name='loss'):
                        loss_verb_uniform,
                    get_metric_tag(TAG_BATCH, train_mode='pred', action_mode='noun', base_metric_name='loss'):
                        loss_noun_uniform,
                }

            logger.info(
                f"ALL Predict including uniform classifier loss: {pprint.pformat(self.pretrain_state.sample_idx_to_pretrain_loss)}")

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def forward(self, inputs, return_feats=False):
        return self.model(inputs, return_feats=return_feats)

    def setup(self, stage):
        """For distributed processes, init anything shared outside nn.Modules here. """
        # Setup is called immediately after the distributed processes have been
        # registered. We can now setup the distributed process groups for each machine
        # and create the distributed data loaders.
        self.configure_head()

    def configure_head(self):
        """ Load output masker at training start, to make checkpoint loading independent of the module. """
        if not self.head_masking_head_is_configured():
            self.model.head = torch.nn.Sequential(
                self.model.head,
                UnseenVerbNounMaskerHead(self.stream_state)
            )
            logger.info(f"Wrapped incremental head for model: {self.model.head}")

    def head_masking_head_is_configured(self):
        """Is the masking already configured. """
        for m in self.model.head.children():
            if isinstance(m, UnseenVerbNounMaskerHead):
                return True
        return False

    def configure_optimizers(self):
        steps_in_epoch = len(self.train_loader)
        return lr_scheduler.lr_factory(
            self.model, self.cfg, steps_in_epoch, self.cfg.SOLVER.LR_POLICY
        )

    def train_dataloader(self):
        return self.train_loader

    def predict_dataloader(self):
        """Gather predictions for train stream."""
        return self.predict_loader

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None

    def validation_step(self, *args, **kwargs):
        raise NotImplementedError()

    def test_step(self, batch, batch_idx):
        raise NotImplementedError()


class FutureSampler:
    """How to sample idxs from the future part of the stream (including current batch before update)."""
    modes = ['full', 'FIFO_split_seen_unseen']

    def __init__(self, mode,
                 stream_idx_to_action_list: list,
                 seen_action_set: Union[set, dict],
                 total_capacity=None):
        assert mode in self.modes
        self.mode = mode
        self.total_capacity = total_capacity
        self.stream_idx_to_action_list = stream_idx_to_action_list
        self.seen_action_set = seen_action_set

    def __call__(self, all_future_idxs: list, *args, **kwargs) -> list:
        raise NotImplementedError("Should not be calling before implemented windowed future sample, "
                                  "FWT is now deprecated so 2-bin balanced sampling is not necessary")
        if self.action_mode == 'full' or len(all_future_idxs) <= self.total_capacity:
            logger.debug(f"Returning all remaining future samples: {len(all_future_idxs)}")
            return all_future_idxs

        elif self.action_mode == 'FIFO_split_seen_unseen':
            return self.get_FIFO_split_seen_unseen(all_future_idxs)

    def get_FIFO_split_seen_unseen(self, all_future_idxs) -> list:
        """
        Divide total capacity equally over seen and unseen action bins.
        Bins are populated sequentially from stream.
        If one bin is not full, allocate left-over capacity to other bin.
        """
        # Sanity check
        if len(all_future_idxs) <= self.total_capacity:
            logger.debug(f"Returning all remaining future samples: {len(all_future_idxs)}")
            return all_future_idxs

        initial_capacity = self.total_capacity // 2

        # Iterate and allocate in bins until both have at least the initial_capacity or end of stream
        seen_bin = []
        unseen_bin = []
        for future_idx in all_future_idxs:
            action_for_idx = self.stream_idx_to_action_list[future_idx]

            if action_for_idx in self.seen_action_set:
                seen_bin.append(future_idx)
            else:
                unseen_bin.append(future_idx)

            # Stop criterion
            if len(seen_bin) >= initial_capacity and len(unseen_bin) >= initial_capacity:
                break

        # If one has < initial capacity and other >, we can reallocate capacity between the two
        min_bin, max_bin = (seen_bin, unseen_bin) if len(seen_bin) < len(unseen_bin) else (unseen_bin, seen_bin)

        # Return if both have less/equal the init capacity (no leftover)
        if len(max_bin) <= initial_capacity:
            logger.debug(f"Sampled for future: seen-action-bin={len(seen_bin)}, unseen-action-bin={len(unseen_bin)}")
            return min_bin + max_bin

        # Reallocate from max_bin
        if len(min_bin) < initial_capacity:  # Means that full future stream is in bins (stop criterion not met)
            extra_capacity = initial_capacity - len(min_bin)
            max_bin = max_bin[:initial_capacity + extra_capacity]
        else:  # If both have enough capacity just return earliest (FIFO) samples for both
            min_bin = min_bin[:initial_capacity]
            max_bin = max_bin[:initial_capacity]

        logger.debug(f"Sampled for future: min-bin={len(min_bin)}, max-bin={len(max_bin)}")
        return min_bin + max_bin


class PastSampler:
    """How to sample idxs from the history part of the stream."""

    modes = ['full', 'windowed', 'uniform_action_uniform_instance']

    def __init__(self, mode,
                 seen_action_to_stream_idxs: dict[tuple, list],
                 total_capacity=None):
        assert mode in self.modes
        self.mode = mode
        self.total_capacity = total_capacity
        self.seen_action_to_stream_idxs = seen_action_to_stream_idxs  # Mapping of all past actions to past stream idxs
        logger.info(f"PastSampler in mode={self.mode}, capacity={self.total_capacity}")

    def __call__(self, all_past_idxs: list, *args, **kwargs) -> list:
        if self.mode == 'full' or len(all_past_idxs) <= self.total_capacity:
            logger.debug(f"Returning all remaining past samples: {len(all_past_idxs)}")
            return all_past_idxs

        elif self.mode == 'windowed':
            window_idxs = all_past_idxs[-self.total_capacity:]
            logger.debug(f"Returning {self.total_capacity}-windowed past samples: {len(window_idxs)}: {window_idxs}")
            return window_idxs

        elif self.mode == 'uniform_action_uniform_instance':
            return self.get_uniform_action_uniform_instance(all_past_idxs)

    def get_uniform_action_uniform_instance(self, all_past_idxs: list) -> list:
        """
        Sample uniform over action-bins, then per bin sample uniform as well.
        This balances sampling for imbalanced action-to-stream-idx bins.
        """

        # Sample actions uniformly
        nb_actions_to_sample = self.total_capacity
        source_action_counts = {
            action: len(stream_idx_list)
            for action, stream_idx_list in self.seen_action_to_stream_idxs.items()
        }  # To sample from
        sampled_action_counts = defaultdict(int)  # The ones we have sampled
        sampled_count = 0

        # Sanity checks
        assert len(all_past_idxs) == sum(cnt for cnt in source_action_counts.values())
        if len(all_past_idxs) <= self.total_capacity:
            logger.debug(f"Returning all remaining past samples: {len(all_past_idxs)}")
            return all_past_idxs

        while sampled_count < nb_actions_to_sample:
            action = random.choice(list(source_action_counts.keys()))

            # Update target
            sampled_count += 1
            sampled_action_counts[action] += 1

            # Update source
            source_action_counts[action] -= 1
            if source_action_counts[action] <= 0:
                del source_action_counts[action]

        # Sample instances uniformly
        sampled_past_idxs = []
        for action, sample_size in sampled_action_counts.items():
            action_idxs = random.sample(self.seen_action_to_stream_idxs[action], k=sample_size)
            sampled_past_idxs.extend(action_idxs)

            logger.debug(f"Action {action}: Sampled {len(action_idxs)} samples = {action_idxs}")

        return sampled_past_idxs

import copy
import pprint

import torch
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

from continual_ego4d.utils.meters import AverageMeter
from continual_ego4d.methods.build import build_method
from continual_ego4d.methods.method_callbacks import Method
from continual_ego4d.metrics.metric import get_metric_tag
from continual_ego4d.metrics.count_metrics import Metric, \
    CountMetric, TAG_BATCH, WindowedUniqueCountMetric
from continual_ego4d.metrics.metric import TAG_BATCH, TAG_PAST
from continual_ego4d.metrics.standard_metrics import OnlineTopkAccMetric, RunningAvgOnlineTopkAccMetric, \
    OnlineLossMetric, RunningAvgOnlineLossMetric
from continual_ego4d.metrics.adapt_metrics import OnlineAdaptationGainMetric, RunningAvgOnlineAdaptationGainMetric, \
    CumulativeOnlineAdaptationGainMetric
from continual_ego4d.metrics.future_metrics import GeneralizationTopkAccMetric, FWTTopkAccMetric, \
    GeneralizationLossMetric, FWTLossMetric
from continual_ego4d.metrics.past_metrics import FullOnlineForgettingAccMetric, ReexposureForgettingAccMetric, \
    CollateralForgettingAccMetric, FullOnlineForgettingLossMetric, ReexposureForgettingLossMetric, \
    CollateralForgettingLossMetric
from continual_ego4d.datasets.continual_action_recog_dataset import verbnoun_to_action, verbnoun_format
from continual_ego4d.utils.models import UnseenVerbNounMaskerHead
from pytorch_lightning.loggers import TensorBoardLogger

import matplotlib.pyplot as plt

from pytorch_lightning.core import LightningModule
from typing import List, Tuple, Union, Any, Optional, Dict
from ego4d.utils import logging

logger = logging.get_logger(__name__)


class StreamStateTracker:
    """
    Disentangles tracking stats from the stream, form the Lightning Module.
    This object can safely be shared as reference, as opposed to the LightningModule which is prone to
    recursion errors due to holding the model and optimizer.
    """

    def __init__(self, stream_loader, pretrain_action_sets):
        self.total_stream_sample_count = len(stream_loader.dataset)

        # Store vars of observed part of stream (Don't reassign, use ref)
        self.seen_samples_idxs = []
        self.stream_seen_action_set = set()
        self.stream_seen_verb_set = set()
        self.stream_seen_noun_set = set()
        self.seen_action_to_stream_idxs = defaultdict(list)  # On-the-fly: For each action keep observed stream ids
        """ Summarize observed part of the stream. """

        # Current iteration State vars (single batch)
        self.batch_idx: int = -1  # Current batch idx
        self.eval_this_step: bool = False
        self.plot_this_step: bool = False
        self.stream_batch_idxs: list = []
        self.stream_batch_size: int = 0  # Size of the new data batch sampled from the stream (exclusive replay samples)
        self.stream_batch_labels: torch.Tensor = None  # Ref for Re-exposure based forgetting
        self.batch_action_set: set = set()
        self.batch_verb_set: set = set()
        self.batch_noun_set: set = set()
        """ Variables set per iteration to share between methods. """

        # For dump
        self.action_to_batches = defaultdict(list)  # On-the-fly: For each action keep all ids when it was observed
        self.batch_to_actions = defaultdict(list)
        """ Track info about stream to include in final dumpfile.
         The dumpfile is used as reference to check if user processing has finished. """

        # Prediction phase
        self.sample_idx_to_pretrain_loss = {}
        """ A dict containing a mapping of the stream sample index to the loss on the initial pretrain model.
        The dictionary is filled in the preprocessing predict phase before training."""

        # Count-sets: Current stream
        self.user_verb_freq_dict = stream_loader.dataset.verb_freq_dict
        self.user_noun_freq_dict = stream_loader.dataset.noun_freq_dict
        self.user_action_freq_dict = stream_loader.dataset.action_freq_dict
        """ Counter dictionaries from the user stream for actions/verbs/nouns. """

        self.sample_idx_to_action_list = stream_loader.dataset.sample_idx_to_action_list
        """ A list containing all actions in the full stream, the array index corresponds to the stream sample idx. """

        # Pretraining stats
        # From JSON: {'ACTION_LABEL': {'name': "ACTION_NAME", 'count': "ACTION_COUNT"}}
        self.pretrain_verb_set = {verbnoun_format(x) for x in pretrain_action_sets['verb_to_name_dict'].keys()}
        self.pretrain_noun_set = {verbnoun_format(x) for x in pretrain_action_sets['noun_to_name_dict'].keys()}
        self.pretrain_action_set = {verbnoun_to_action(*str(action).split('-'))  # Json format to tuple for actions
                                    for action in pretrain_action_sets['action_to_name_dict'].keys()}
        """ Sets containing all the action/nouns/verbs from the pretraining phase. """


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
        self.save_hyperparameters()  # Save cfg to '

        # Multi-task (verb/noun) has classification head, mask out unseen classifier prototype outputs
        self.model = build_model(cfg)

        self.loss_fun = losses.get_loss_func(self.cfg.MODEL.LOSS_FUNC)(reduction="mean")  # Training
        self.loss_fun_pred = losses.get_loss_func(self.cfg.MODEL.LOSS_FUNC)(reduction="none")  # Prediction
        self.continual_eval_freq = cfg.CONTINUAL_EVAL.FREQ
        self.plotting_log_freq = cfg.CONTINUAL_EVAL.PLOTTING_FREQ

        # Sanity modes/debugging
        self.enable_prepost_comparing = True  # Compare loss before/after update

        # Dataloader
        self.train_loader = construct_trainstream_loader(self.cfg, shuffle=False)

        # State tracker of stream
        self.stream_state = StreamStateTracker(
            self.train_loader, pretrain_action_sets=cfg.COMPUTED_PRETRAIN_ACTION_SETS
        )

        # Pretrain loader: Only samples that both verb and noun have been seen in pretrain
        self.predict_phase_load_idxs = []
        for stream_sample_idx, action in enumerate(self.stream_state.sample_idx_to_action_list):
            verb, noun = action
            if verb in self.stream_state.pretrain_verb_set and noun in self.stream_state.pretrain_noun_set:
                self.predict_phase_load_idxs.append(stream_sample_idx)

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
                                                   seen_action_set=self.stream_state.stream_seen_action_set,
                                                   total_capacity=self.cfg.CONTINUAL_EVAL.FUTURE_SAMPLE_CAPACITY)
        self.past_stream_sampler = PastSampler(mode='uniform_action_uniform_instance',
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
                RunningAvgOnlineLossMetric(TAG_BATCH, loss_fun=self.loss_fun, mode=mode),

                # ADAPT METRICS
                OnlineAdaptationGainMetric(
                    TAG_BATCH, self.loss_fun_pred, self.stream_state.sample_idx_to_pretrain_loss, loss_mode=mode),
                RunningAvgOnlineAdaptationGainMetric(
                    TAG_BATCH, self.loss_fun_pred, self.stream_state.sample_idx_to_pretrain_loss, loss_mode=mode),
                CumulativeOnlineAdaptationGainMetric(
                    TAG_BATCH, self.loss_fun_pred, self.stream_state.sample_idx_to_pretrain_loss, loss_mode=mode),
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
            ('action', self.stream_state.stream_seen_action_set, self.stream_state.user_action_freq_dict,
             self.stream_state.pretrain_action_set),
            ('verb', self.stream_state.stream_seen_verb_set, self.stream_state.user_verb_freq_dict,
             self.stream_state.pretrain_verb_set),
            ('noun', self.stream_state.stream_seen_noun_set, self.stream_state.user_noun_freq_dict,
             self.stream_state.pretrain_noun_set),
        ]:
            batch_metrics.extend([  # Seen actions (history part of stream) vs full user stream actions
                CountMetric(observed_set_name="seen", observed_set=seen_set,
                            ref_set_name="stream", ref_set=user_ref_set,
                            mode=mode
                            ),
                # Seen actions (history part of stream) vs all actions seen during pretraining phase
                CountMetric(observed_set_name="seen", observed_set=seen_set,
                            ref_set_name="pretrain", ref_set=pretrain_ref_set,
                            mode=mode
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
                    TAG_PAST, self.loss_fun_pred, self.stream_state.sample_idx_to_pretrain_loss, loss_mode=action_mode),
                RunningAvgOnlineAdaptationGainMetric(
                    TAG_PAST, self.loss_fun_pred, self.stream_state.sample_idx_to_pretrain_loss, loss_mode=action_mode),
                CumulativeOnlineAdaptationGainMetric(
                    TAG_PAST, self.loss_fun_pred, self.stream_state.sample_idx_to_pretrain_loss, loss_mode=action_mode),

                # ACTION METRICS
                OnlineTopkAccMetric(TAG_PAST, k=1, mode=action_mode),
                OnlineLossMetric(TAG_PAST, loss_fun=self.loss_fun, mode=action_mode),

                # TODO: Accuracy (unbalanced)
                # TODO Track on re-exposure for scatterplot
                # ReexposureForgettingAccMetric(
                #     k=1, action_mode=action_mode, keep_action_results_over_time=True),
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

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """
        Override to alter or apply batch augmentations to your batch before it is transferred to the device.
        e.g. Replay adds samples.
        Happens before on_train_batch_start (although differently documented).
        """
        # Reset metrics
        self.batch_metric_results = {}
        for metric in self.all_metrics:
            if metric.reset_before_batch:
                metric.reset()

        # Set state
        self._set_current_batch_states(batch, batch_idx=dataloader_idx)

        # Update batch
        altered_batch = self.method.on_before_batch_transfer(batch, dataloader_idx)
        return altered_batch

    def _set_current_batch_states(self, batch: Any, batch_idx: int):
        self.stream_state.batch_idx = batch_idx

        # Eval or plot at this iteration
        self.stream_state.plot_this_step = batch_idx % self.plotting_log_freq == 0 or batch_idx == len(
            self.train_loader)
        self.stream_state.eval_this_step = self.trainer.logger_connector.should_update_logs
        logger.debug(f"Continual eval on batch "
                     f"{batch_idx}/{len(self.train_loader)} = {self.stream_state.eval_this_step}")

        # Observed idxs update before batch is altered
        _, labels, _, stream_sample_idxs = batch
        self.stream_state.stream_batch_labels = labels
        self.stream_state.stream_batch_idxs = stream_sample_idxs.tolist()
        self.stream_state.stream_batch_size = len(self.stream_state.stream_batch_idxs)

        # Get new actions/verbs current batch
        self.stream_state.batch_action_set = set()
        self.stream_state.batch_verb_set = set()
        self.stream_state.batch_noun_set = set()
        for ((verb, noun), sample_idx) in zip(labels.tolist(), self.stream_state.stream_batch_idxs):
            action = verbnoun_to_action(verb, noun)
            self.stream_state.batch_action_set.add(action)
            self.stream_state.batch_verb_set.add(verbnoun_format(verb))
            self.stream_state.batch_noun_set.add(verbnoun_format(noun))

        logger.debug(f"current_batch_sample_idxs={self.stream_state.stream_batch_idxs}")

    def _log_current_batch_states(self, batch_idx: int):
        """ Log stats for current iteration stream. """
        # Before-update counts
        metric_results = {
            get_metric_tag(TAG_BATCH, base_metric_name=f"history_sample_count"):
                len(self.stream_state.seen_samples_idxs),
            get_metric_tag(TAG_BATCH, base_metric_name=f"future_sample_count"):
                self.stream_state.total_stream_sample_count - len(self.stream_state.seen_samples_idxs),
        }

        self.log_step_metrics(metric_results)
        logger.debug(f"Logging state of batch_idx={batch_idx}/{len(self.train_loader)}: "
                     f"{pprint.pformat(metric_results)}")

    def training_step(self, batch, batch_idx):
        """ Before update: Forward and define loss. """
        # Log current batch state (only possible starting from training_step)
        self._log_current_batch_states(batch_idx=batch_idx)

        # PREDICTIONS + LOSS
        metric_results = {}
        inputs, labels, video_names, _ = batch

        # Do method callback (Get losses etc)
        # This is the loss used for updates and the outputs of the entire batch.
        # Note that for metrics: e.g. replay we don't want to use all outputs, but only the new ones from the stream
        fwd_inputs = inputs
        if self.enable_prepost_comparing:  # Make copy as inputs are adapted in-place in SlowFast
            fwd_inputs = [inputs[i].clone() for i in range(len(inputs))]

        loss, verbnoun_outputs, step_results = self.method.training_step(
            fwd_inputs, labels, current_batch_stream_idxs=self.stream_state.stream_batch_idxs
        )
        self.add_to_dict_(metric_results, step_results)

        # Perform additional eval
        if self.stream_state.eval_this_step:
            logger.debug(f"Starting PRE-UPDATE evaluation: batch_idx={batch_idx}/{len(self.train_loader)}")
            self.eval_current_stream_batch_preupdate_(
                metric_results,
                [verbnoun_outputs[i][:self.stream_state.stream_batch_size] for i in range(2)],
                labels[:self.stream_state.stream_batch_size]
            )
            self.eval_future_data_(metric_results, batch_idx)

        # LOG results
        self.log_step_metrics(metric_results)
        logger.debug(f"PRE-UPDATE Results for batch_idx={batch_idx}/{len(self.train_loader)}: "
                     f"{pprint.pformat(metric_results)}")

        # Only loss should be used and stored for entire epoch (stream)
        return loss

    def on_after_backward(self):
        # Log gradients possibly
        if (self.cfg.LOG_GRADIENT_PERIOD >= 0 and
                self.trainer.global_step % self.cfg.LOG_GRADIENT_PERIOD == 0):
            for name, weight in self.model.named_parameters():
                if weight is not None:
                    self.logger.experiment.add_histogram(
                        name, weight, self.trainer.global_step
                    )
                    if weight.grad is not None:
                        self.logger.experiment.add_histogram(
                            f"{name}.grad", weight.grad, self.trainer.global_step
                        )

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        metric_results = {}
        inputs, labels, video_names, _ = batch

        # Measure difference of pre-update results of current batch (e.g. forward second time)
        self.eval_current_stream_batch_postupdate_(metric_results, batch)

        # Do post-update evaluation of the past
        if self.stream_state.eval_this_step:
            logger.debug(f"Starting POST-UPDATE evaluation on batch_idx={batch_idx}/{len(self.train_loader)}")
            self.eval_past_data_(metric_results, batch_idx)

            # (optionally) Save metrics after batch
            for metric in self.all_metrics:
                metric.save_result_to_history(current_batch_idx=batch_idx)

        # Update counts etc
        self._update_seen_state(labels, batch_idx)

        # Plot metrics if possible
        if self.stream_state.plot_this_step:
            self._log_plotting_metrics()

        # LOG results
        self.log_step_metrics(metric_results)
        logger.debug(f"POST-UPDATE Results for batch_idx={batch_idx}/{len(self.train_loader)}: "
                     f"{pprint.pformat(metric_results)}")

    def _update_seen_state(self, labels, batch_idx):
        self.stream_state.stream_seen_action_set.update(self.stream_state.batch_action_set)
        self.stream_state.stream_seen_verb_set.update(self.stream_state.batch_verb_set)
        self.stream_state.stream_seen_noun_set.update(self.stream_state.batch_noun_set)

        # Only iterate stream batch (not replay samples)
        for ((verb, noun), sample_idx) in zip(labels.tolist(), self.stream_state.stream_batch_idxs):
            action = verbnoun_to_action(verb, noun)
            self.stream_state.action_to_batches[action].append(batch_idx)  # Possibly add multiple time batch_idx
            self.stream_state.seen_action_to_stream_idxs[action].append(sample_idx)
            self.stream_state.batch_to_actions[batch_idx].append(action)

        # Update Task states
        self.stream_state.seen_samples_idxs.extend(self.stream_state.stream_batch_idxs)
        assert len(self.stream_state.seen_samples_idxs) == len(np.unique(self.stream_state.seen_samples_idxs)), \
            f"Duplicate visited samples in {self.stream_state.seen_samples_idxs}"

    def _log_plotting_metrics(self):
        """ Iterate over metrics to get Image plots. """

        # Collect loggers
        tb_loggers = [result_logger for result_logger in self.logger if isinstance(result_logger, TensorBoardLogger)]
        if len(tb_loggers) == 0:
            logger.info(f"No tensorboard logger found, skipping image plotting.")
            return
        elif len(tb_loggers) > 1:
            raise Exception(f"Multiple tensorboard loggers found, should only define one: {tb_loggers}")
        tb_logger = tb_loggers[0]

        logger.info("Collecting figures for metric plots")
        plot_dict = {}
        for metric in self.all_metrics:
            metric_plot_dict = metric.plot()
            if metric_plot_dict is not None and len(metric_plot_dict) > 0:
                self.add_to_dict_(plot_dict, metric_plot_dict)

        # Log them
        logger.info("Plotting tensorboard figures")
        for name, mpl_figure in plot_dict.items():
            tb_logger.experiment.add_figure(
                tag=name, figure=mpl_figure
            )
        plt.close('all')

    def on_train_end(self) -> None:
        """Dump any additional stats about the training."""
        dump_dict = {
            "batch_to_actions": self.stream_state.batch_to_actions,
            "action_to_batches": self.stream_state.action_to_batches,
            "dataset_all_entries_ordered": self.train_dataloader().dataset.seq_input_list,
        }
        for metric in self.all_metrics:
            metric_dict = metric.dump()
            if metric_dict is not None and len(metric_dict) > 0:
                self.add_to_dict_(dump_dict, metric_dict)

        torch.save(dump_dict, self.dumpfile)
        logger.debug(f"Logged stream info to dumpfile {self.dumpfile}")

    # ---------------------
    # PER-STEP EVALUATION
    # ---------------------
    def log_step_metrics(self, log_dict):
        self.add_to_dict_(self.batch_metric_results, log_dict)  # Update state of all metrics in all phases
        for logname, logval in log_dict.items():
            self.log(logname, float(logval), on_step=True, on_epoch=False)

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
            metric.update(self.stream_state.batch_idx, verbnoun_outputs, labels, self.stream_state.stream_batch_idxs)

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
        post_loss_action, post_loss_verb, post_loss_noun = self.method.get_losses_from_preds(
            post_update_preds, stream_labels, loss_fun=self.method.loss_fun_train
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
        all_future_idxs = list(range(min(self.stream_state.stream_batch_idxs), len(self.train_loader.dataset)))
        sampled_future_idxs = self.future_stream_sampler(all_future_idxs)
        logger.debug(f"SAMPLED {len(sampled_future_idxs)} from all_future_idxs interval = "
                     f"[{all_future_idxs[0]},..., {all_future_idxs[-1]}]")

        # Create new dataloader
        future_dataloader = self._get_train_dataloader_subset(
            self.train_loader,
            batch_size=self.cfg.CONTINUAL_EVAL.BATCH_SIZE,
            subset_indices=sampled_future_idxs,  # Future data, including current
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
            batch_size=self.cfg.CONTINUAL_EVAL.BATCH_SIZE,
            subset_indices=sampled_past_idxs,  # Previous data, not including current
        )
        result_dict = self._get_metric_results_over_dataloader(past_dataloader, metrics=self.past_metrics)
        self.add_to_dict_(step_result, result_dict)

    # ---------------------
    # HELPER METHODS
    # ---------------------
    @staticmethod
    def add_to_dict_(source_dict: dict, dict_to_add: dict, key_exist_ok=False):
        """In-place add to dict"""
        for k, v in dict_to_add.items():
            if not key_exist_ok and k in source_dict:
                raise ValueError(f'dict_to_add is overwriting source_dict, existing key={k}')
            source_dict[k] = v

    def _get_train_dataloader_subset(self, train_dataloader: torch.utils.data.DataLoader,
                                     subset_indices: Union[List, Tuple],
                                     batch_size: int = None):
        """ Get a subset of the training dataloader's dataset.

        !Warning!: DONT COPY SAMPLER from train_dataloader to new dataloader as __len__ is re-used
        from the parent train_dataloader in the new dataloader (which may not match the Dataset).
        """
        dataset = train_dataloader.dataset

        if batch_size is None:
            batch_size = train_dataloader.batch_size

        if subset_indices is not None:
            dataset = torch.utils.data.Subset(dataset, indices=subset_indices)

        batch_size = min(len(dataset), batch_size)  # Effective batch size

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=self.cfg.DATA_LOADER.PIN_MEMORY,
            collate_fn=None,
        )
        return loader

    @torch.no_grad()
    def _get_metric_results_over_dataloader(self, dataloader, metrics: List[Metric]):

        # Update metrics over dataloader data
        logger.debug(f"Iterating dataloader and transferring to device: {self.device}")
        for batch_idx, (inputs, labels, _, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Slowfast inputs (list):
            # inputs[0].shape = torch.Size([32, 3, 8, 224, 224]) -> Slow net
            # inputs[1].shape =  torch.Size([32, 3, 32, 224, 224]) -> Fast net

            labels = labels.to(self.device)
            inputs = [x.to(self.device) for x in inputs] if isinstance(inputs, list) \
                else inputs.to(self.device)
            preds = self.forward(inputs)

            for metric in metrics:
                metric.update(
                    self.stream_state.batch_idx, preds, labels,
                    stream_batch_labels=self.stream_state.stream_batch_labels
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
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        """ Collect per-sample stats such as the loss. """
        inputs, labels, video_names, stream_sample_idxs = batch

        # Loss per sample
        _, _, sample_to_results = self.method.prediction_step(inputs, labels, stream_sample_idxs.tolist())
        for k, v in sample_to_results.items():
            self.stream_state.sample_idx_to_pretrain_loss[k] = v

    @torch.no_grad()
    def on_predict_end(self) -> None:
        """ Get uniform (max entropy) classifier predictions over seen verbs/nouns in pretraining.
        As these prototypes are unseen during pretrain, we need a proxy to calculate the delta in Adaptation Gain.
        The proxy baseline performance we choose is this from classifier always predicting a uniform distribution,
        or a maximum entropy classifier.
        We only consider the pretraining prototypes of the pretraining phase for the uniform predictor.
        As later adaptation with incremental classes is also seen as an improvement over the original pretrain model.
        """
        logger.info(f"Predict collected over stream: {pprint.pformat(self.stream_state.sample_idx_to_pretrain_loss)}")

        nb_verbs_total, nb_nouns_total = self.cfg.MODEL.NUM_CLASSES  # [ 115, 478 ]

        # retrieve how many seen during pretrain
        nb_verbs_seen_pretrain = len(self.stream_state.pretrain_verb_set)
        nb_nouns_seen_pretrain = len(self.stream_state.pretrain_noun_set)
        logger.info(f"Pretraining seen verbs = {nb_verbs_seen_pretrain}/{nb_verbs_total}, "
                    f"nouns= {nb_nouns_seen_pretrain}/{nb_nouns_total}")

        # Random label in unifom prediction distribution
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
            self.stream_state.sample_idx_to_pretrain_loss[unseen_in_pretrain_idx] = {
                get_metric_tag(TAG_BATCH, train_mode='pred', action_mode='action', base_metric_name='loss'):
                    loss_action_uniform,
                get_metric_tag(TAG_BATCH, train_mode='pred', action_mode='verb', base_metric_name='loss'):
                    loss_verb_uniform,
                get_metric_tag(TAG_BATCH, train_mode='pred', action_mode='noun', base_metric_name='loss'):
                    loss_noun_uniform,
            }

        logger.info(
            f"ALL Predict including uniform classifier loss: {pprint.pformat(self.stream_state.sample_idx_to_pretrain_loss)}")

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def forward(self, inputs):
        return self.model(inputs)

    def setup(self, stage):
        """For distributed processes, init anything shared outside nn.Modules here. """
        # Setup is called immediately after the distributed processes have been
        # registered. We can now setup the distributed process groups for each machine
        # and create the distributed data loaders.
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
                 seen_action_set: set,
                 total_capacity=None):
        assert mode in self.modes
        self.mode = mode
        self.total_capacity = total_capacity
        self.stream_idx_to_action_list = stream_idx_to_action_list
        self.seen_action_set = seen_action_set

    def __call__(self, all_future_idxs: list, *args, **kwargs) -> list:
        if self.mode == 'full' or len(all_future_idxs) <= self.total_capacity:
            logger.debug(f"Returning all remaining future samples: {len(all_future_idxs)}")
            return all_future_idxs

        elif self.mode == 'FIFO_split_seen_unseen':
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

    modes = ['full', 'uniform_action_uniform_instance']

    def __init__(self, mode,
                 seen_action_to_stream_idxs: dict[tuple, list],
                 total_capacity=None):
        assert mode in self.modes
        self.mode = mode
        self.total_capacity = total_capacity
        self.seen_action_to_stream_idxs = seen_action_to_stream_idxs  # Mapping of all past actions to past stream idxs

    def __call__(self, all_past_idxs: list, *args, **kwargs) -> list:
        if self.mode == 'full' or len(all_past_idxs) <= self.total_capacity:
            logger.debug(f"Returning all remaining past samples: {len(all_past_idxs)}")
            return all_past_idxs

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

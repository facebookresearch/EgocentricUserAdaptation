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
from continual_ego4d.metrics.batch_metrics import Metric, OnlineTopkAccMetric, RunningAvgOnlineTopkAccMetric, \
    CountMetric, TAG_BATCH
from continual_ego4d.metrics.adapt_metrics import OnlineAdaptationGainMetric, RunningAvgOnlineAdaptationGainMetric, \
    CumulativeOnlineAdaptationGainMetric
from continual_ego4d.metrics.future_metrics import GeneralizationTopkAccMetric, FWTTopkAccMetric, \
    GeneralizationLossMetric, FWTLossMetric
from continual_ego4d.metrics.past_metrics import FullOnlineForgettingAccMetric, ReexposureForgettingAccMetric, \
    CollateralForgettingAccMetric, FullOnlineForgettingLossMetric, ReexposureForgettingLossMetric, \
    CollateralForgettingLossMetric
from continual_ego4d.datasets.continual_action_recog_dataset import verbnoun_to_action, verbnoun_format

from pytorch_lightning.core import LightningModule
from typing import List, Tuple, Union, Any, Optional, Dict
from ego4d.utils import logging

logger = logging.get_logger(__name__)


class ContinualMultiTaskClassificationTask(LightningModule):
    """
    Training mode: Visit samples in stream sequentially, update, and evaluate per update.
    Validation mode: Disabled.
    Test mode: Disabled (No held-out test sets available for online learning)
    Predict mode: Visit the stream and collect per-sample stats such as the loss. No learning is performed.

    For all lightning hooks, see: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
    """

    def __init__(self, cfg, future_metrics=None, past_metrics=None, shuffle_stream=False):
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
        self.model = build_model(cfg)
        self.loss_fun = losses.get_loss_func(self.cfg.MODEL.LOSS_FUNC)(reduction="mean")  # Training
        self.loss_fun_pred = losses.get_loss_func(self.cfg.MODEL.LOSS_FUNC)(reduction="none")  # Prediction
        self.continual_eval_freq = cfg.CONTINUAL_EVAL.FREQ

        # Pretraining stats
        self.pretrain_action_sets = cfg.COMPUTED_PRETRAIN_ACTION_SETS

        # Dataloader
        self.train_loader = construct_trainstream_loader(self.cfg, shuffle=shuffle_stream)
        self.predict_loader = construct_predictstream_loader(self.train_loader, self.cfg)
        self.total_stream_sample_count = len(self.train_loader.dataset)

        # Predict phase:
        # If we first run predict phase, we can fill this dict with the results, this can then be used in trainphase
        self.run_predict_before_train = True
        self.sample_idx_to_pretrain_loss = {}
        self.sample_idx_to_action_list = [None] * self.total_stream_sample_count

        # Store vars of observed part of stream (Don't reassign, use ref)
        self.seen_samples_idxs = []
        self.seen_action_set = set()
        self.seen_verb_set = set()
        self.seen_noun_set = set()
        self.seen_action_to_stream_idxs = defaultdict(list)  # On-the-fly: For each action keep observed stream ids

        # State vars (single batch)
        self.current_batch_stream_idxs = None
        self.eval_this_step = False
        self.stream_batch_size = None  # Size of the new data batch sampled from the stream (exclusive replay samples)
        self.stream_batch_labels = None  # Ref for Re-exposure based forgetting

        # For data stream info dump
        self.dumpfile = self.cfg.COMPUTED_USER_DUMP_FILE
        self.action_to_batches = defaultdict(list)  # On-the-fly: For each action keep all ids when it was observed
        self.batch_to_actions = defaultdict(list)

        # Stream samplers
        self.future_stream_sampler = FutureSampler(mode='FIFO_split_seen_unseen',
                                                   stream_idx_to_action_list=self.sample_idx_to_action_list,
                                                   seen_action_set=self.seen_action_set,
                                                   total_capacity=self.cfg.CONTINUAL_EVAL.FUTURE_SAMPLE_CAPACITY)
        self.past_stream_sampler = PastSampler(mode='uniform_action_uniform_instance',
                                               seen_action_to_stream_idxs=self.seen_action_to_stream_idxs,
                                               total_capacity=self.cfg.CONTINUAL_EVAL.PAST_SAMPLE_CAPACITY)

        # Method
        self.method: Method = build_method(cfg, self)

        # Count-sets: Current stream
        user_verb_freq_dict = self.train_loader.dataset.verb_freq_dict
        user_noun_freq_dict = self.train_loader.dataset.noun_freq_dict
        user_action_freq_dict = self.train_loader.dataset.action_freq_dict

        # Count-sets: Pretrain stream
        # From JSON: {'ACTION_LABEL': {'name': "ACTION_NAME", 'count': "ACTION_COUNT"}}
        pretrain_verb_set = {verbnoun_format(x) for x in self.pretrain_action_sets['verb_to_name_dict'].keys()}
        pretrain_noun_set = {verbnoun_format(x) for x in self.pretrain_action_sets['noun_to_name_dict'].keys()}
        pretrain_action_set = {verbnoun_to_action(*str(action).split('-'))  # Json format to tuple for actions
                               for action in self.pretrain_action_sets['action_to_name_dict'].keys()}

        # Metrics
        # CURRENT BATCH METRICS
        verbnoun_metrics = [[OnlineTopkAccMetric(k=k, mode=m), RunningAvgOnlineTopkAccMetric(k=k, mode=m)]
                            for k, m in product([1, 5], ['verb', 'noun'])]
        verbnoun_metrics = [metric for metric_list in verbnoun_metrics for metric in metric_list]

        action_metrics = [
            OnlineTopkAccMetric(k=1, mode='action'),
            RunningAvgOnlineTopkAccMetric(k=1, mode='action')
        ]

        adapt_metrics = [
            [
                OnlineAdaptationGainMetric(
                    self.loss_fun_pred, self.sample_idx_to_pretrain_loss, loss_mode=loss_mode),
                RunningAvgOnlineAdaptationGainMetric(
                    self.loss_fun_pred, self.sample_idx_to_pretrain_loss, loss_mode=loss_mode),
                CumulativeOnlineAdaptationGainMetric(
                    self.loss_fun_pred, self.sample_idx_to_pretrain_loss, loss_mode=loss_mode),
            ] for loss_mode in ['action', 'verb', 'noun']
        ]
        adapt_metrics = [metric for metric_list in adapt_metrics for metric in metric_list]  # Flatten

        count_metrics = [
            [  # Seen actions (history part of stream) vs full user stream actions
                CountMetric(observed_set_name="seen", observed_set=seen_set,
                            ref_set_name="stream", ref_set=user_ref_set,
                            mode=mode
                            ),
                # Seen actions (history part of stream) vs all actions seen during pretraining phase
                CountMetric(observed_set_name="seen", observed_set=seen_set,
                            ref_set_name="pretrain", ref_set=pretrain_ref_set,
                            mode=mode
                            ),
            ] for mode, seen_set, user_ref_set, pretrain_ref_set in [
                ('action', self.seen_action_set, user_action_freq_dict, pretrain_action_set),
                ('verb', self.seen_verb_set, user_verb_freq_dict, pretrain_verb_set),
                ('noun', self.seen_noun_set, user_noun_freq_dict, pretrain_noun_set),
            ]
        ]
        count_metrics = [metric for metric_list in count_metrics for metric in metric_list]  # Flatten

        self.current_batch_metrics = [*verbnoun_metrics, *action_metrics, *adapt_metrics, *count_metrics]

        # FUTURE METRICS
        self.future_metrics = future_metrics
        if self.future_metrics is None:  # None = default
            action_metrics = [
                GeneralizationTopkAccMetric(
                    seen_action_set=self.seen_action_set, k=1, action_mode='action'),
                FWTTopkAccMetric(
                    seen_action_set=self.seen_action_set, k=1, action_mode='action'),
                GeneralizationLossMetric(
                    seen_action_set=self.seen_action_set, action_mode='action', loss_fun=self.loss_fun),
                FWTLossMetric(
                    seen_action_set=self.seen_action_set, action_mode='action', loss_fun=self.loss_fun),
            ]
            verb_metrics = [
                GeneralizationTopkAccMetric(
                    seen_action_set=self.seen_verb_set, k=1, action_mode='verb'),
                FWTTopkAccMetric(
                    seen_action_set=self.seen_verb_set, k=1, action_mode='verb'),
                GeneralizationLossMetric(
                    seen_action_set=self.seen_verb_set, action_mode='verb', loss_fun=self.loss_fun),
                FWTLossMetric(
                    seen_action_set=self.seen_verb_set, action_mode='verb', loss_fun=self.loss_fun),
            ]
            noun_metrics = [
                GeneralizationTopkAccMetric(
                    seen_action_set=self.seen_noun_set, k=1, action_mode='noun'),
                FWTTopkAccMetric(
                    seen_action_set=self.seen_noun_set, k=1, action_mode='noun'),
                GeneralizationLossMetric(
                    seen_action_set=self.seen_noun_set, action_mode='noun', loss_fun=self.loss_fun),
                FWTLossMetric(
                    seen_action_set=self.seen_noun_set, action_mode='noun', loss_fun=self.loss_fun),
            ]
            self.future_metrics = action_metrics + verb_metrics + noun_metrics

        # PAST METRICS
        self.past_metrics = past_metrics
        if self.past_metrics is None:  # None = default
            action_metrics = [
                FullOnlineForgettingAccMetric(k=1, action_mode='action'),
                ReexposureForgettingAccMetric(k=1, action_mode='action'),
                CollateralForgettingAccMetric(k=1, action_mode='action'),
                FullOnlineForgettingLossMetric(loss_fun=self.loss_fun, action_mode='action'),
                ReexposureForgettingLossMetric(loss_fun=self.loss_fun, action_mode='action'),
                CollateralForgettingLossMetric(loss_fun=self.loss_fun, action_mode='action'),
            ]
            verb_metrics = [
                FullOnlineForgettingAccMetric(k=1, action_mode='verb'),
                ReexposureForgettingAccMetric(k=1, action_mode='verb'),
                CollateralForgettingAccMetric(k=1, action_mode='verb'),
                FullOnlineForgettingLossMetric(loss_fun=self.loss_fun, action_mode='verb'),
                ReexposureForgettingLossMetric(loss_fun=self.loss_fun, action_mode='verb'),
                CollateralForgettingLossMetric(loss_fun=self.loss_fun, action_mode='verb'),
            ]
            noun_metrics = [
                FullOnlineForgettingAccMetric(k=1, action_mode='noun'),
                ReexposureForgettingAccMetric(k=1, action_mode='noun'),
                CollateralForgettingAccMetric(k=1, action_mode='noun'),
                FullOnlineForgettingLossMetric(loss_fun=self.loss_fun, action_mode='noun'),
                ReexposureForgettingLossMetric(loss_fun=self.loss_fun, action_mode='noun'),
                CollateralForgettingLossMetric(loss_fun=self.loss_fun, action_mode='noun'),
            ]
            self.past_metrics = action_metrics + verb_metrics + noun_metrics

        logger.debug(f'Initialized {self.__class__.__name__}')

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
    def on_train_batch_start(self, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        # Reset metrics
        for metric in [*self.current_batch_metrics, *self.future_metrics, *self.past_metrics]:
            if metric.reset_before_batch:
                metric.reset()

        self.eval_this_step = self.trainer.logger_connector.should_update_logs
        logger.debug(f"Continual eval on batch {batch_idx} = {self.eval_this_step}")

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Override to alter or apply batch augmentations to your batch before it is transferred to the device."""
        # Observed idxs update before batch is altered
        _, labels, _, stream_sample_idxs = batch
        self.stream_batch_labels = labels
        self.current_batch_stream_idxs = stream_sample_idxs.tolist()
        self.stream_batch_size = len(self.current_batch_stream_idxs)
        logger.debug(f"current_batch_sample_idxs={self.current_batch_stream_idxs}")

        # Alter batch (e.g. Replay adds new samples)
        altered_batch = self.method.on_before_batch_transfer(batch, dataloader_idx)
        return altered_batch

    def training_step(self, batch, batch_idx):
        """ Before update: Forward and define loss. """
        metric_results = {}

        # PREDICTIONS + LOSS
        inputs, labels, video_names, _ = batch

        # Before-update counts
        metric_results = {**metric_results, **{
            get_metric_tag(TAG_BATCH, base_metric_name=f"history_sample_count"):
                len(self.seen_samples_idxs),
            get_metric_tag(TAG_BATCH, base_metric_name=f"future_sample_count"):
                self.total_stream_sample_count - len(self.seen_samples_idxs),
        }}

        # Do method callback (Get losses etc)
        loss, outputs, step_results = self.method.training_step(
            inputs, labels, current_batch_stream_idxs=self.current_batch_stream_idxs
        )
        metric_results = {**metric_results, **step_results}

        # Perform additional eval
        if self.eval_this_step:
            logger.debug(f"Starting PRE-UPDATE evaluation: batch_idx={batch_idx}")
            self.eval_current_batch_(metric_results, outputs, labels)
            self.eval_future_data_(metric_results, batch_idx)

        # LOG results
        self.log_step_metrics(metric_results)
        logger.debug(f"PRE-UPDATE Results for batch_idx={batch_idx}: {pprint.pformat(metric_results)}")

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
        inputs, labels, video_names, _ = batch

        # Do post-update evaluation of the past
        if self.eval_this_step:
            logger.debug(f"Starting POST-UPDATE evaluation on batch_idx={batch_idx}")
            metric_results = {}
            self.eval_past_data_(metric_results, batch_idx)

            # LOG results
            self.log_step_metrics(metric_results)
            logger.debug(f"POST-UPDATE Results for batch_idx={batch_idx}: {pprint.pformat(metric_results)}")

            # (optionally) Save metrics after batch
            for metric in [*self.current_batch_metrics, *self.future_metrics, *self.past_metrics]:
                metric.save_result_to_history()

        # Update counts etc
        self._update_state(labels, batch_idx)

    def _update_state(self, labels, batch_idx):
        # Only iterate stream batch (not replay samples)
        for ((verb, noun), sample_idx) in zip(labels.tolist(), self.current_batch_stream_idxs):
            action = verbnoun_to_action(verb, noun)
            self.seen_action_set.add(action)
            self.seen_verb_set.add(verb)
            self.seen_noun_set.add(noun)
            self.action_to_batches[action].append(batch_idx)  # Possibly add multiple time batch_idx
            self.seen_action_to_stream_idxs[action].append(sample_idx)
            self.batch_to_actions[batch_idx].append(action)

        # Update Task states
        self.seen_samples_idxs.extend(self.current_batch_stream_idxs)
        assert len(self.seen_samples_idxs) == len(np.unique(self.seen_samples_idxs)), \
            f"Duplicate visited samples in {self.seen_samples_idxs}"

    def on_train_end(self) -> None:
        """Dump any additional stats about the training."""
        torch.save({
            "batch_to_actions": self.batch_to_actions,
            "action_to_batches": self.action_to_batches,
            "dataset_all_entries_ordered": self.train_dataloader().dataset.seq_input_list,
        },
            self.dumpfile)
        logger.debug(f"Logged stream info to dumpfile {self.dumpfile}")

    # ---------------------
    # PER-STEP EVALUATION
    # ---------------------
    def log_step_metrics(self, log_dict):
        # logger.debug(f"LOGGING: {log_dict}")
        for logname, logval in log_dict.items():
            self.log(logname, float(logval), on_step=True, on_epoch=False)

    @torch.no_grad()
    @_eval_in_train_decorator
    def eval_current_batch_(self, step_result, outputs, labels):
        """Add additional metrics for current batch in-place to the step_result dict."""
        logger.debug(f"Gathering online results")

        # Update metrics
        for metric in self.current_batch_metrics:
            metric.update(outputs, labels, self.current_batch_stream_idxs)

        # Gather results from metrics
        results = {}
        for metric in self.current_batch_metrics:
            results = {**results, **metric.result()}

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
        all_future_idxs = list(range(min(self.current_batch_stream_idxs), len(self.train_loader.dataset)))
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

        all_past_idxs = np.unique(self.seen_samples_idxs).tolist()
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
    def add_to_dict_(source_dict: dict, dict_to_add: dict):
        """In-place add to dict"""
        for k, v in dict_to_add.items():
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
                    preds, labels, stream_batch_labels=self.stream_batch_labels
                )

        # Gather results
        avg_metric_result_dict = {}
        for metric in metrics:
            avg_metric_result_dict = {**avg_metric_result_dict, **metric.result()}

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
            self.sample_idx_to_pretrain_loss[k] = v

        # Actions in stream
        for stream_sample_idx, label in zip(stream_sample_idxs.tolist(), labels.tolist()):
            self.sample_idx_to_action_list[stream_sample_idx] = tuple(label)

    def on_predict_end(self) -> None:
        logger.info(f"Predict collected over stream: {pprint.pformat(self.sample_idx_to_pretrain_loss)}")

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
        pass

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

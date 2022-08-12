import torch
from fvcore.nn.precise_bn import get_bn_modules

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
from continual_ego4d.datasets.continual_dataloader import construct_trainstream_loader
import os.path as osp

from continual_ego4d.utils.meters import AverageMeter
from continual_ego4d.methods.build import build_method
from continual_ego4d.methods.method_callbacks import Method
from continual_ego4d.metrics.batch_metrics import Metric, OnlineTopkAccMetric, RunningAvgOnlineTopkAccMetric, \
    CountMetric
from continual_ego4d.metrics.future_metrics import GeneralizationTopkAccMetric, FWTTopkAccMetric
from continual_ego4d.metrics.past_metrics import FullOnlineForgettingMetric, ReexposureForgettingMetric, \
    CollateralForgettingMetric
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

    def __init__(self, cfg, future_metrics=None, past_metrics=None):
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
        self.method: Method = build_method(cfg, self)
        self.continual_eval_freq = cfg.TRAIN.CONTINUAL_EVAL_FREQ

        # Pretraining stats
        self.pretrain_action_sets = cfg.COMPUTED_PRETRAIN_ACTION_SETS

        # Predict phase:
        # If we first run predict phase, we can fill this dict with the results, this can then be used in trainphase
        self.run_predict_before_train = True
        self.sample_to_pretrain_loss = {}

        # Dataloader
        self.train_loader = construct_trainstream_loader(self.cfg, shuffle=False)
        self.total_stream_sample_count = len(self.train_loader.dataset)

        # Store vars (Don't reassign, use ref)
        self.seen_samples_idxs = []
        self.seen_action_set = set()
        self.seen_verb_set = set()
        self.seen_noun_set = set()

        # State vars (single batch)
        self.current_batch_sample_idxs = None
        self.eval_this_step = False

        # For data stream info dump
        self.dumpfile = self.cfg.COMPUTED_USER_DUMP_FILE
        self.action_to_batches = defaultdict(list)  # On-the-fly: For each action keep all ids when it was observed
        self.batch_to_actions = defaultdict(list)

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

        count_metrics = [
            [  # Seen actions (history part of stream) vs full user stream actions
                CountMetric(observed_set_name="seen", observed_set=seen_set,
                            ref_set_name="user", ref_set=user_ref_set,
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

        self.current_batch_metrics = [*verbnoun_metrics, *action_metrics, *count_metrics]

        # FUTURE METRICS
        self.future_metrics = future_metrics
        if self.future_metrics is None:  # None = default
            action_metrics = [
                GeneralizationTopkAccMetric(seen_action_set=self.seen_action_set, k=1, mode='action'),
                FWTTopkAccMetric(seen_action_set=self.seen_action_set, k=1, mode='action'),
            ]
            verb_metrics = [
                GeneralizationTopkAccMetric(seen_action_set=self.seen_action_set, k=1, mode='verb'),
                FWTTopkAccMetric(seen_action_set=self.seen_action_set, k=1, mode='verb'),
            ]
            noun_metrics = [
                GeneralizationTopkAccMetric(seen_action_set=self.seen_action_set, k=1, mode='noun'),
                FWTTopkAccMetric(seen_action_set=self.seen_action_set, k=1, mode='noun'),
            ]
            self.future_metrics = action_metrics + verb_metrics + noun_metrics

        # PAST METRICS
        self.past_metrics = past_metrics
        if self.past_metrics is None:  # None = default
            action_metrics = [
                FullOnlineForgettingMetric(k=1, mode='action'),
                ReexposureForgettingMetric(k=1, mode='action'),
                CollateralForgettingMetric(k=1, mode='action'),
            ]
            verb_metrics = [
                FullOnlineForgettingMetric(k=1, mode='verb'),
                ReexposureForgettingMetric(k=1, mode='verb'),
                CollateralForgettingMetric(k=1, mode='verb'),
            ]
            noun_metrics = [
                FullOnlineForgettingMetric(k=1, mode='noun'),
                ReexposureForgettingMetric(k=1, mode='noun'),
                CollateralForgettingMetric(k=1, mode='noun'),
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

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Override to alter or apply batch augmentations to your batch before it is transferred to the device."""
        altered_batch = self.method.on_before_batch_transfer(batch, dataloader_idx)
        return altered_batch

    def training_step(self, batch, batch_idx):
        """ Before update: Forward and define loss. """
        metric_results = {}

        # PREDICTIONS + LOSS
        inputs, labels, video_names, stream_sample_idxs = batch

        # Observed idxs update
        self.current_batch_sample_idxs = stream_sample_idxs.tolist()
        logger.debug(f"current_batch_sample_idxs={self.current_batch_sample_idxs}")
        metric_results = {**metric_results, **{
            "history_sample_count": len(self.seen_samples_idxs),
            "future_sample_count": self.total_stream_sample_count - len(self.seen_samples_idxs)
        }}

        # Do method callback (Get losses etc)
        loss, outputs, step_results = self.method.training_step(inputs, labels)
        metric_results = {**metric_results, **step_results}

        # Perform additional eval
        if self.eval_this_step:
            logger.debug(f"Starting PRE-UPDATE evaluation: batch_idx={batch_idx}")
            self.eval_current_batch_(metric_results, outputs, labels)
            self.eval_future_data_(metric_results, batch_idx)

        # LOG results
        self.log_metrics(metric_results)
        logger.debug(f"PRE-UPDATE Results for batch_idx={batch_idx}: {metric_results}")

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
            self.log_metrics(metric_results)
            logger.debug(f"POST-UPDATE Results for batch_idx={batch_idx}: {metric_results}")

            # (optionally) Save metrics after batch
            for metric in [*self.current_batch_metrics, *self.future_metrics, *self.past_metrics]:
                metric.save_result_to_history()

        # Derive action from verbs,nouns
        for (verb, noun) in labels.tolist():
            action = verbnoun_to_action(verb, noun)
            self.seen_action_set.add(action)
            self.seen_verb_set.add(verb)
            self.seen_noun_set.add(noun)
            self.action_to_batches[action].append(batch_idx)  # Possibly add multiple time batch_idx
            self.batch_to_actions[batch_idx].append(action)

        # Update Task states
        self.seen_samples_idxs.extend(self.current_batch_sample_idxs)
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
    def log_metrics(self, log_dict):
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
            metric.update(outputs, labels)

        # Gather results from metrics
        results = {}
        for metric in self.current_batch_metrics:
            results = {**results, **metric.result()}

        self.add_to_dict_(step_result, results, prefix='online')

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
        unseen_idxs = list(range(min(self.current_batch_sample_idxs), len(self.train_loader.dataset)))
        logger.debug(f"unseen_idxs interval = [{unseen_idxs[0]},..., {unseen_idxs[-1]}]")

        # Create new dataloader
        future_dataloader = self._get_train_dataloader_subset(
            self.train_loader,
            batch_size=self.cfg.TRAIN.CONTINUAL_EVAL_BATCH_SIZE,
            subset_indices=unseen_idxs,  # Future data, including current
        )

        result_dict = self._get_metric_results_over_dataloader(future_dataloader, metrics=self.future_metrics)

        self.add_to_dict_(step_result, result_dict, prefix='future')

    @torch.no_grad()
    @_eval_in_train_decorator
    def eval_past_data_(self, step_result, batch_idx):
        if len(self.past_metrics) == 0:
            return
        if batch_idx == 0:  # first batch
            logger.debug(f"Skipping first batch past data results (no observed data yet)")
            return
        logger.debug(f"Gathering results on past data")

        seen_idxs = np.unique(self.seen_samples_idxs)
        logger.debug(f"seen_idxs interval = [{seen_idxs[0]},..., {seen_idxs[-1]}]")

        obs_dataloader = self._get_train_dataloader_subset(
            self.train_loader,
            batch_size=self.cfg.TRAIN.CONTINUAL_EVAL_BATCH_SIZE,
            subset_indices=seen_idxs,  # Previous data, not including current
        )
        result_dict = self._get_metric_results_over_dataloader(obs_dataloader, metrics=self.past_metrics)

        self.add_to_dict_(step_result, result_dict, prefix='past')

    # ---------------------
    # HELPER METHODS
    # ---------------------
    @staticmethod
    def add_to_dict_(source_dict: dict, dict_to_add: dict, prefix: str = "", ):
        if len(prefix) > 0:
            prefix = f"{prefix}_"
        for k, v in dict_to_add.items():
            source_dict[f"{prefix}{k}"] = v

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
        for batch_idx, (inputs, labels, _, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Slowfast inputs (list):
            # inputs[0].shape = torch.Size([32, 3, 8, 224, 224]) -> Slow net
            # inputs[1].shape =  torch.Size([32, 3, 32, 224, 224]) -> Fast net

            labels = labels.to(self.device)
            inputs = [x.to(self.device) for x in inputs] if isinstance(inputs, list) \
                else inputs.to(self.device)
            preds = self.forward(inputs)

            for metric in metrics:
                metric.update(preds, labels)

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
        losses, outputs, step_results = self.method.prediction_step(inputs, labels)

        for sample_idx, sample_loss in zip(stream_sample_idxs.tolist(), losses.tolist()):
            self.sample_to_pretrain_loss[sample_idx] = sample_loss

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
        return self.train_loader

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None

    def validation_step(self, *args, **kwargs):
        raise NotImplementedError()

    def test_step(self, batch, batch_idx):
        raise NotImplementedError()

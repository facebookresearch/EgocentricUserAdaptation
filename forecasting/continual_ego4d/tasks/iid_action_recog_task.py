import copy
import pprint

import torch
import wandb
from fvcore.nn.precise_bn import get_bn_modules
from collections import Counter
from continual_ego4d.tasks.continual_action_recog_task import ContinualMultiTaskClassificationTask, PretrainState, \
    StreamStateTracker

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


class IIDMultiTaskClassificationTask(LightningModule):
    """
    """

    def __init__(self, cfg):
        """
        """
        logger.debug('Starting init IIDMultiTaskClassificationTask')
        super().__init__()

        # Disable automatic updates after training_step(), instead do manually
        # We need to do this for alternative update schedules (e.g. multiple iters per batch)
        # See: https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html#automatic-optimization
        self.automatic_optimization = True

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

        # For data stream info dump. Is used as token for finishing the job
        self.dumpfile = self.cfg.COMPUTED_USER_DUMP_FILE

        # Predict phase:
        self.run_predict_before_train: bool = False
        """ Triggers running prediction phase before starting training on the stream. 
        This allows preprocessing on the entire stream (e.g. collect pretraining losses and stream stats)."""

        # Need stream state for head masking
        self.pretrain_state = PretrainState(cfg.COMPUTED_PRETRAIN_ACTION_SETS)
        self.cfg.COMPUTED_PRETRAIN_STATE = self.pretrain_state  # Set for dataset creation

        # Dataloader
        self.train_loader = construct_trainstream_loader(self.cfg, shuffle=True)  # SHUFFLE LEARNING EPOCHS

        # Stream state
        self.stream_state = StreamStateTracker(self.train_loader, self.pretrain_state)  # State tracker of stream

        logger.debug(f'Initialized {self.__class__.__name__}')

    # ---------------------
    # TRAINING FLOW CALLBACKS
    # ---------------------
    def training_step(self, batch, batch_idx):
        """ Before update: Forward and define loss. """

        # PREDICTIONS + LOSS
        inputs, labels, video_names, _ = batch

        preds = self.forward(inputs, return_feats=False)
        loss_action_m, loss_verb_m, loss_noun_m = OnlineLossMetric.get_losses_from_preds(
            preds, labels, self.loss_fun_unred, mean=True
        )

        # add model trainable params
        (train_p, total_p), (head_train_p, head_total_p) = model_trainable_summary(self.model)

        log_results = {
            # Losses
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='action', base_metric_name='loss'):
                loss_action_m.item(),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='verb', base_metric_name='loss'):
                loss_verb_m.item(),
            get_metric_tag(TAG_BATCH, train_mode='train', action_mode='noun', base_metric_name='loss'):
                loss_noun_m.item(),

            # Trainable
            get_metric_tag(TAG_BATCH, base_metric_name=f"model_trainable_params"): train_p,
            get_metric_tag(TAG_BATCH, base_metric_name=f"model_all_params"): total_p,
            get_metric_tag(TAG_BATCH, base_metric_name=f"head_trainable_params"): head_train_p,
            get_metric_tag(TAG_BATCH, base_metric_name=f"head_all_params"): head_total_p,
        }

        self.log_step_metric_results(log_results)
        return loss_action_m

    def on_train_end(self) -> None:
        """Dump any additional stats about the training."""
        wandb_logger: WandbLogger = ContinualMultiTaskClassificationTask.get_logger_instance(self.logger, WandbLogger)
        assert wandb_logger is not None, "Must have wandb logger to finish run!"

        torch.save({}, self.dumpfile)  # Save token
        logger.debug(f"Logged stream info to dumpfile {self.dumpfile}")

        # Let WandB logger know that run is fully executed
        wandb_logger.experiment.log({"finished_run": True})

    # ---------------------
    # PER-STEP EVALUATION
    # ---------------------
    def log_step_metric_results(self, log_dict):
        for logname, logval in log_dict.items():
            self.log(logname, float(logval), on_step=True, on_epoch=False)

    # ---------------------
    # PREDICTION FLOW CALLBACKS
    # ---------------------
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        raise NotImplementedError()

    def validation_step(self, *args, **kwargs):
        raise NotImplementedError()

    def test_step(self, batch, batch_idx):
        raise NotImplementedError()

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def forward(self, inputs, return_feats=False):
        return self.model(inputs, return_feats=return_feats)

    def setup(self, stage):
        """
        This is called in train/predict/test phase inits.
        For distributed processes, init anything shared outside nn.Modules here.
        """
        # Setup is called immediately after the distributed processes have been
        # registered. We can now setup the distributed process groups for each machine
        # and create the distributed data loaders.
        ContinualMultiTaskClassificationTask.configure_head(self.model, self.stream_state)

    def configure_optimizers(self):
        steps_in_epoch = len(self.train_loader)
        return lr_scheduler.lr_factory(
            self.model, self.cfg, steps_in_epoch, self.cfg.SOLVER.LR_POLICY
        )

    # ---------------------
    # LOADERS
    # ---------------------
    def train_dataloader(self):
        return self.train_loader

    def predict_dataloader(self):
        return None

    def test_dataloader(self):
        return None

    def val_dataloader(self):
        return None

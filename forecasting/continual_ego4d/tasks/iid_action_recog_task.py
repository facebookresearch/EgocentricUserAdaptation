from ego4d.utils import distributed as du
from ego4d.utils import logging
from continual_ego4d.datasets.continual_dataloader import construct_trainstream_loader
from typing import List, Tuple, Union, Any, Optional, Dict
from continual_ego4d.tasks.continual_action_recog_task import ContinualMultiTaskClassificationTask
from ego4d.evaluation import lta_metrics as metrics
import torch
from fvcore.nn.precise_bn import get_bn_modules

logger = logging.get_logger(__name__)


class IIDMultiTaskClassificationTask(ContinualMultiTaskClassificationTask):
    """
    # TODO: IID baseline makes no sense for streaming performance?
    Unless retrain every iteration = O(N^2)

    Can use CL Task, but set loader to shuffle + allow multiple epochs.
    We still don't use validation, as it's a direct baseline.

    How to use this class:
        1) trainer.fit()
         -> first train the model on the training-stream (cfg.SOLVER.MAX_EPOCH repeat cycles of the stream)
         -> Gives online metrics. These are NOT directly comparable to the online learning, as the stream is now shuffled.
         This means that tracked online metrics just give an idea of the learning progress on the shuffled stream.
        2) trainer.test()
        for IID baseline: model tested after trainer.fit()
        for pretrain model: model test directly (no trainer.fit())
    """

    def __init__(self, cfg):
        super().__init__(cfg, future_metrics=[], past_metrics=[], shuffle_stream=True)  # No future/past metrics
        self.test_mode = "train_stream"  # Can configure later
        self.run_predict_before_train = False

    def training_step(self, batch, batch_idx):
        inputs, labels, _, _ = batch
        preds = self.forward(inputs)
        loss1 = self.loss_fun(preds[0], labels[:, 0])
        loss2 = self.loss_fun(preds[1], labels[:, 1])
        loss = loss1 + loss2
        top1_err_verb, top5_err_verb = metrics.distributed_topk_errors(
            preds[0], labels[:, 0], (1, 5)
        )
        top1_err_noun, top5_err_noun = metrics.distributed_topk_errors(
            preds[1], labels[:, 1], (1, 5)
        )

        top1_err_action: torch.FloatTensor = metrics.distributed_twodistr_top1_errors(
            preds[0], preds[1], labels[:, 0], labels[:, 1]
        )

        metric_results = {
            "train_action_loss": loss.item(),
            "train_verb_loss": loss1.item(),
            "train_noun_loss": loss2.item(),
            "train_top1_action_err": top1_err_action.item(),
            "train_top1_verb_err": top1_err_verb.item(),
            "train_top5_verb_err": top5_err_verb.item(),
            "train_top1_noun_err": top1_err_noun.item(),
            "train_top5_noun_err": top5_err_noun.item(),
        }

        self.log_step_metrics(metric_results)
        logger.debug(f"PRE-UPDATE Results for batch_idx={batch_idx}: {metric_results}")

        return {**{"loss": loss}, **{metric_results}}

    def training_epoch_end(self, outputs):
        """ Avg collected metrics over stream. """
        logger.debug(f"Starting training_epoch_end")

        keys = [x for x in outputs[0].keys() if x is not "loss"]
        for key in keys:
            metric = torch.tensor([x[key] for x in outputs]).mean()
            self.log(f'{key}_avg', metric)
            logger.info(f"END TRAIN EPOCH {self.current_epoch}: {key}={metric}")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def test_dataloader(self):
        if self.test_mode == "train_stream":
            return self.train_loader
        else:
            raise ValueError()

    def test_step(self, batch, batch_idx):
        inputs, labels, _, _ = batch
        preds = self.forward(inputs)

        # Get losses
        loss_verb = self.lightning_module.loss_fun(preds[0], labels[:, 0])  # Verbs
        loss_noun = self.lightning_module.loss_fun(preds[1], labels[:, 1])  # Nouns
        loss_action = loss_verb + loss_noun  # Avg losses

        log_results = {
            "action_loss": loss_action.item(),
            "verb_loss": loss_verb.item(),
            "noun_loss": loss_noun.item(),
        }

        # Get accuracies
        top1_acc_verb, top5_acc_verb = metrics.distributed_topk_errors(
            preds[0], labels[:, 0], (1, 5), return_mode='acc'
        )
        top1_acc_noun, top5_acc_noun = metrics.distributed_topk_errors(
            preds[1], labels[:, 1], (1, 5), return_mode='acc'
        )
        action_top1_acc = metrics.distributed_twodistr_top1_errors(
            preds[0], preds[1], labels[:, 0], labels[:, 1], return_mode='acc')

        log_results = {**log_results, **{
            "test_top1_verb_acc": top1_acc_verb.item(),
            "test_top5_verb_acc": top5_acc_verb.item(),
            "test_top1_noun_acc": top1_acc_noun.item(),
            "test_top5_noun_acc": top5_acc_noun.item(),
            "test_action_top1_acc": action_top1_acc.item(),
        }}

        return log_results

    def test_epoch_end(self, outputs):
        """ Collect over all metric-results collected over steps. """
        metric_names = outputs[0].keys()  # First step keys
        for metric_name in metric_names:
            metric = torch.tensor([step_output[metric_name] for step_output in outputs], dtype=torch.float).mean()
            self.log(metric_name, metric)
            logger.info(f"END TESTING: {metric_name}={metric}")

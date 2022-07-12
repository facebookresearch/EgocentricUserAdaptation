import torch
from fvcore.nn.precise_bn import get_bn_modules

from ego4d.evaluation import lta_metrics as metrics
from ego4d.utils import misc
from ego4d.models import losses
from ego4d.optimizers import lr_scheduler
from ego4d.utils import distributed as du
from ego4d.utils import logging
from ego4d.datasets import loader
from ego4d.models import build_model
from pytorch_lightning.core import LightningModule

logger = logging.get_logger(__name__)


class ContinualVideoTask(LightningModule):
    def __init__(self, cfg):
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

        self.cfg = cfg
        self.save_hyperparameters()  # Save cfg to '
        self.model = build_model(cfg)
        self.loss_fun = losses.get_loss_func(self.cfg.MODEL.LOSS_FUNC)(reduction="mean")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def training_step_end(self, training_step_outputs):
        if self.cfg.SOLVER.ACCELERATOR in ["dp", "gpu"]:
            training_step_outputs["loss"] = training_step_outputs["loss"].mean()
        return training_step_outputs

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def forward(self, inputs):
        return self.model(inputs)

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def setup(self, stage):
        # Setup is called immediately after the distributed processes have been
        # registered. We can now setup the distributed process groups for each machine
        # and create the distributed data loaders.
        # if not self.cfg.FBLEARNER:
        if self.cfg.SOLVER.ACCELERATOR not in ["dp", "gpu"]:
            du.init_distributed_groups(self.cfg)

        self.train_loader = loader.construct_loader(self.cfg, "train")
        self.val_loader = None
        self.test_loader = None

    def configure_optimizers(self):
        steps_in_epoch = len(self.train_loader)
        return lr_scheduler.lr_factory(
            self.model, self.cfg, steps_in_epoch, self.cfg.SOLVER.LR_POLICY
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def on_after_backward(self):
        if (self.cfg.LOG_GRADIENT_PERIOD >= 0 and
                self.trainer.global_step % self.cfg.LOG_GRADIENT_PERIOD == 0
        ):
            for name, weight in self.model.named_parameters():
                if weight is not None:
                    self.logger.experiment.add_histogram(
                        name, weight, self.trainer.global_step
                    )
                    if weight.grad is not None:
                        self.logger.experiment.add_histogram(
                            f"{name}.grad", weight.grad, self.trainer.global_step
                        )


class ContinualMultiTaskClassificationTask(ContinualVideoTask):
    checkpoint_metric = "val_top1_noun_err"

    def training_step(self, batch, batch_idx):
        """Before update.
        """

        # PREDICTIONS + LOSS
        inputs, labels, _, _ = batch
        preds = self.forward(inputs)
        loss1 = self.loss_fun(preds[0], labels[:, 0])  # Verbs
        loss2 = self.loss_fun(preds[1], labels[:, 1])  # Nouns
        loss = loss1 + loss2  # Avg losses

        step_result = {
            "loss": loss,
            "train_loss": loss.item()
        }

        # CURRENT BATCH METRICS
        step_result = {**step_result, **self.get_current_batch_metrics(preds, labels)}

        # PAST METRICS
        step_result = {**step_result, **self.get_observed_data_metrics(preds, labels)}

        # FUTURE METRICS
        step_result = {**step_result, **self.get_unseen_data_metrics(preds, labels)}

        return step_result

    def get_current_batch_metrics(self, preds, labels):
        """ Use current data observed training data in mini-batch."""
        top1_err_verb, top5_err_verb = metrics.distributed_topk_errors(
            preds[0], labels[:, 0], (1, 5)
        )
        top1_err_noun, top5_err_noun = metrics.distributed_topk_errors(
            preds[1], labels[:, 1], (1, 5)
        )

        return {
            "train_top1_verb_err": top1_err_verb.item(),
            "train_top5_verb_err": top5_err_verb.item(),
            "train_top1_noun_err": top1_err_noun.item(),
            "train_top5_noun_err": top5_err_noun.item(),
        }

    def get_observed_data_metrics(self, preds, labels):
        """ Stability measure of previous data. """
        # Create new dataloader
        pass

    def get_unseen_data_metrics(self, preds, labels):
        """ Zero-shot and generalization performance of future data (including current just-observed mini-batch."""
        # Create new dataloaders
        pass

    def training_epoch_end(self, outputs):
        """ End of stream."""
        raise NotImplementedError()  # TODO

        if self.cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(self.model)) > 0:
            misc.calculate_and_update_precise_bn(
                self.train_loader, self.model, self.cfg.BN.NUM_BATCHES_PRECISE
            )
        _ = misc.aggregate_split_bn_stats(self.model)

        keys = [x for x in outputs[0].keys() if x is not "loss"]
        for key in keys:
            metric = torch.tensor([x[key] for x in outputs]).mean()
            self.log(key, metric)

    def test_step(self, batch, batch_idx):
        raise NotImplementedError(
            """We have no test 'phase' in user-specific action-recognition.
        Rerun the continual 'train' experiments with the testing user-subset."""
        )

    def test_epoch_end(self, outputs):
        raise NotImplementedError(
            """We have no test 'phase' in user-specific action-recognition.
        Rerun the continual 'train' experiments with the testing user-subset."""
        )

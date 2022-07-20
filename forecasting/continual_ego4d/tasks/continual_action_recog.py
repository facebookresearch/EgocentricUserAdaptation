import torch
from fvcore.nn.precise_bn import get_bn_modules

from collections import defaultdict

from ego4d.evaluation import lta_metrics as metrics
from ego4d.utils import misc
from ego4d.models import losses
from ego4d.optimizers import lr_scheduler
from ego4d.utils import distributed as du
from ego4d.utils import logging
from ego4d.datasets import loader
from ego4d.models import build_model

from continual_ego4d.utils.meters import AverageMeter
from continual_ego4d.methods.build import build_method
from continual_ego4d.methods.method_callbacks import Method
from pytorch_lightning.core import LightningModule
from typing import List, Tuple, Union, Any

logger = logging.get_logger(__name__)


class ContinualVideoTask(LightningModule):
    """
    For all lightning hooks, see: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
    """

    def __init__(self, cfg):
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

        self.cfg = cfg
        self.save_hyperparameters()  # Save cfg to '
        self.model = build_model(cfg)
        self.loss_fun = losses.get_loss_func(self.cfg.MODEL.LOSS_FUNC)(reduction="mean")

        self.method: Method = build_method(cfg, self)

        # State vars
        self.seen_samples_idxs = []
        self.first_unseen_sample_idx = 0
        logger.debug('Initialized ContinualVideoTask')

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

        self.train_loader = loader.construct_loader(self.cfg, "continual")
        # self.val_loader = None
        # self.test_loader = None

    def configure_optimizers(self):
        steps_in_epoch = len(self.train_loader)
        return lr_scheduler.lr_factory(
            self.model, self.cfg, steps_in_epoch, self.cfg.SOLVER.LR_POLICY
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None

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


class ContinualMultiTaskClassificationTask(LightningModule):

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Override to alter or apply batch augmentations to your batch before it is transferred to the device."""
        altered_batch = self.method.on_before_batch_transfer(batch, dataloader_idx)
        return altered_batch

    def training_step(self, batch, batch_idx):
        """ Before update: Forward and define loss. """
        # PREDICTIONS + LOSS
        inputs, labels, video_names, sample_idxs = batch
        sample_idxs = sample_idxs.tolist()
        self.first_unseen_sample_idx = min(sample_idxs)  # First unseen in sequence is min of batch

        # Do method callback
        step_result, outputs = self.method.training_step(inputs, labels)

        # Perform additional eval
        step_result = self.eval_current_batch(step_result, outputs, labels, batch_idx, sample_idxs)

        # Update seen samples
        self.seen_samples_idxs.extend(sample_idxs)

        return step_result

    def eval_current_batch(self, step_result, outputs, labels, batch_idx, sample_idxs):
        logger.debug(f"Starting evaluation on current iteration: batch_idx={batch_idx}")

        # SET TO EVAL MODE
        self.model.train(False)
        torch.set_grad_enabled(False)

        #############################################
        # METRICS

        # CURRENT BATCH METRICS
        logger.debug(f"Gathering online results -> batch {batch_idx} SAMPLE IDXS = {sample_idxs}")
        step_result = {**step_result,
                       **self.add_prefix_to_keys(prefix='online',
                                                 source_dict=self._get_metric_results(outputs, labels))}
        # PAST METRICS
        logger.debug(f"Gathering results on past data")
        step_result = {**step_result,
                       **self.add_prefix_to_keys(prefix='seen',
                                                 source_dict=self._get_observed_data_metrics(batch_idx))}

        # FUTURE METRICS
        logger.debug(f"Gathering results on future data")
        step_result = {**step_result,
                       **self.add_prefix_to_keys(prefix='unseen',
                                                 source_dict=self._get_unseen_data_metrics(batch_idx))}
        #############################################

        # SET BACK TO TRAIN MODE
        self.model.train()
        torch.set_grad_enabled(True)

        logger.debug(f"Results for batch_idx={batch_idx}: {step_result}")
        return step_result

    def add_prefix_to_keys(self, prefix: str, source_dict: dict, ):
        return {f"{prefix}_{k}": v for k, v in source_dict.items()}

    @torch.no_grad()
    def _get_metric_results(self, preds, labels):
        """ Get metrics for predictions and labels."""
        top1_err_verb, top5_err_verb = metrics.distributed_topk_errors(
            preds[0], labels[:, 0], (1, 5)
        )
        top1_err_noun, top5_err_noun = metrics.distributed_topk_errors(
            preds[1], labels[:, 1], (1, 5)
        )

        # TODO get total action error (merge the two)

        return {
            f"top1_verb_err": top1_err_verb.item(),
            f"top5_verb_err": top5_err_verb.item(),
            f"top1_noun_err": top1_err_noun.item(),
            f"top5_noun_err": top5_err_noun.item(),
        }

    @torch.no_grad()
    def _get_observed_data_metrics(self, batch_idx):
        """ Stability measure of previous data. """
        if batch_idx == 0:  # first batch
            logger.debug(f"Skipping first batch (no observed data yet)")
            return {}

        seen_idxs = self.seen_samples_idxs
        logger.debug(f"seen_idxs interval = [{seen_idxs[0]},..., {seen_idxs[-1]}]")

        obs_dataloader = self._get_train_dataloader_subset(
            self.train_loader,
            batch_size=self.cfg.TRAIN.CONTINUAL_EVAL_BATCH_SIZE,
            subset_indices=seen_idxs,  # Previous data, not including current
        )

        result_dict = self._get_average_metrics_for_dataloader(obs_dataloader)
        return result_dict

    @torch.no_grad()
    def _get_unseen_data_metrics(self, batch_idx):
        """ Zero-shot and generalization performance of future data (including current just-observed mini-batch."""
        if batch_idx == len(self.train_loader):  # last batch
            return {}

        unseen_idxs = list(range(self.first_unseen_sample_idx, len(self.train_loader.dataset)))
        logger.debug(f"unseen_idxs interval = [{unseen_idxs[0]},..., {unseen_idxs[-1]}]")

        # Create new dataloader
        future_dataloader = self._get_train_dataloader_subset(
            self.train_loader,
            batch_size=self.cfg.TRAIN.CONTINUAL_EVAL_BATCH_SIZE,
            subset_indices=unseen_idxs,  # Future data, including current
        )

        result_dict = self._get_average_metrics_for_dataloader(future_dataloader)
        return result_dict

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
    def _get_average_metrics_for_dataloader(self, dataloader):

        avg_metric_result_dict = defaultdict(AverageMeter)
        for batch_idx, (inputs, labels, _, _) in enumerate(dataloader):
            # Slowfast inputs (list):
            # inputs[0].shape = torch.Size([32, 3, 8, 224, 224]) -> Slow net
            # inputs[1].shape =  torch.Size([32, 3, 32, 224, 224]) -> Fast net

            labels = labels.to(self.device)
            inputs = [x.to(self.device) for x in inputs] if isinstance(inputs, list) \
                else inputs.to(self.device)

            batch_size = labels.shape[0]
            preds = self.forward(inputs)

            metric_result_dict = self._get_metric_results(preds, labels)
            for k, v in metric_result_dict.items():
                avg_metric_result_dict[k].update(v, weight=batch_size)

        # Avg per metric
        for metric_name, avg_meter in avg_metric_result_dict.items():
            avg_metric_result_dict[metric_name] = avg_meter.avg
        return avg_metric_result_dict

    def training_epoch_end(self, outputs):
        """ End of stream."""
        return
        raise NotImplementedError()  # TODO make results/logs dump

        if self.cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(self.model)) > 0:
            misc.calculate_and_update_precise_bn(
                self.train_loader, self.model, self.cfg.BN.NUM_BATCHES_PRECISE
            )
        _ = misc.aggregate_split_bn_stats(self.model)

        keys = [x for x in outputs[0].keys() if x != "loss"]
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

from abc import ABC, abstractmethod
from continual_ego4d.methods.build import build_method, METHOD_REGISTRY
from typing import Dict, Tuple
from torch import Tensor
from pytorch_lightning.core import LightningModule


class Method:
    def __init__(self, cfg, lightning_module: LightningModule):
        self.cfg = cfg  # For method-specific params
        self.lightning_module = lightning_module
        self.trainer = self.lightning_module.trainer

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return batch

    def training_step(self, inputs, labels) -> Tuple[Tensor, Tensor, Dict]:
        """ Training step for the method when observing a new batch.
        Return Loss,  prediction outputs,a nd dictionary of result metrics to log."""

        preds = self.lightning_module.forward(inputs)
        loss1 = self.lightning_module.loss_fun(preds[0], labels[:, 0])  # Verbs
        loss2 = self.lightning_module.loss_fun(preds[1], labels[:, 1])  # Nouns
        loss = loss1 + loss2  # Avg losses

        log_results = {
            "loss": loss,
            "train_loss": loss.item(),
            "verb_loss": loss1.item(),
            "noun_loss": loss2.item(),
        }
        return loss, preds, log_results


@METHOD_REGISTRY.register()
class Finetuning(Method):
    def __init__(self, cfg, lightning_module):
        super().__init__(cfg, lightning_module)

    def training_step(self, inputs, labels):
        return super().training_step(inputs, labels)


@METHOD_REGISTRY.register()
class Replay(Method):
    def __init__(self, cfg, lightning_module):
        super().__init__(cfg, lightning_module)
        self.memory_dataloader_idxs = []
        self.mem_input = None
        self.mem_labels = None

    def on_before_batch_transfer(self, batch, dataloader_idx):
        # TODO add replay samples
        pass

    def training_step(self, inputs, labels):
        loss, preds, log_results = super().training_step(inputs, labels)

        # TODO add samples to buffer

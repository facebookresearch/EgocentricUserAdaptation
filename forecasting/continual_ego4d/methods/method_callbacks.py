from abc import ABC, abstractmethod
from continual_ego4d.methods.build import build_method, METHOD_REGISTRY
from typing import Dict, Tuple
from torch import Tensor


class Method:
    def __init__(self, cfg, trainer):
        self.cfg = cfg  # For method-specific params
        self.trainer = trainer

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return batch

    def training_step(self, inputs, labels) -> Tuple[Dict, Tensor]:
        """ Training step for the method when observing a new batch.
        Return dictionary of result metrics and the prediction outputs."""

        preds = self.trainer.forward(inputs)
        loss1 = self.trainer.loss_fun(preds[0], labels[:, 0])  # Verbs
        loss2 = self.trainer.loss_fun(preds[1], labels[:, 1])  # Nouns
        loss = loss1 + loss2  # Avg losses

        step_result = {
            "loss": loss,
            "train_loss": loss.item(),
            "verb_loss": loss1.item(),
            "noun_loss": loss2.item(),
        }
        return step_result, preds


@METHOD_REGISTRY.register()
class Finetuning(Method):
    def __init__(self, cfg, trainer):
        super().__init__(cfg, trainer)

    def training_step(self, inputs, labels):
        return super().training_step(inputs, labels)


@METHOD_REGISTRY.register()
class Replay(Method):
    def __init__(self, cfg, trainer):
        super().__init__(cfg, trainer)
        self.memory_dataloader_idxs = []
        self.mem_input = None
        self.mem_labels = None

    def on_before_batch_transfer(self, batch, dataloader_idx):
        # TODO add replay samples
        pass

    def training_step(self, inputs, labels):
        step_result, preds = super().training_step(inputs, labels)

        # TODO add samples to buffer

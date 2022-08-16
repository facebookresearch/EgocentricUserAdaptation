import torch
from typing import Dict, Set, Union, Tuple
from continual_ego4d.utils.meters import AverageMeter
from continual_ego4d.metrics.metric import AvgMeterMetric, Metric, get_metric_tag
from ego4d.evaluation import lta_metrics as metrics

TAG_BATCH = 'batch'


class OnlineTopkAccMetric(AvgMeterMetric):
    reset_before_batch = True

    def __init__(self, k: int = 1, mode="action"):
        super().__init__(mode=mode)
        self.k = k
        self.name = get_metric_tag(parent_tag=TAG_BATCH, action_mode=mode, base_metric_name=f"top{self.k}_acc")
        self.avg_meter = AverageMeter()

        # Checks
        if self.mode == 'action':
            assert self.k == 1, f"Action mode only supports top1, not top-{self.k}"

    @torch.no_grad()
    def update(self, preds, labels, *args, **kwargs):
        """Update metric from predictions and labels."""
        assert preds[0].shape[0] == labels.shape[0], f"Batch sizes not matching!"
        batch_size = labels.shape[0]

        # Verb/noun errors
        if self.mode in ['verb', 'noun']:
            topk_acc: torch.FloatTensor = metrics.distributed_topk_errors(
                preds[self.label_idx], labels[:, self.label_idx], [self.k], return_mode='acc'
            )[0]  # Unpack self.k
        elif self.mode in ['action']:
            topk_acc: torch.FloatTensor = metrics.distributed_twodistr_top1_errors(
                preds[0], preds[1], labels[:, 0], labels[:, 1], return_mode='acc')

        self.avg_meter.update(topk_acc.item(), weight=batch_size)


class RunningAvgOnlineTopkAccMetric(OnlineTopkAccMetric):
    reset_before_batch = False

    def __init__(self, k: int = 1, mode="action"):
        super().__init__(k, mode)
        self.name = f"{self.name}_running_avg"


class CountMetric(Metric):
    reset_before_batch = False

    modes = ["verb", "noun", "action"]

    def __init__(
            self,
            observed_set_name: str,
            observed_set: set,
            ref_set_name: str = None,
            ref_set: set = None,
            mode="action"
    ):
        self.observed_set_name = observed_set_name
        self.observed_set = observed_set

        self.ref_set = ref_set
        if self.ref_set is not None:
            assert ref_set_name is not None
            self.ref_set_name = ref_set_name

        self.mode = mode  # Action/verb/noun
        assert self.mode in self.modes

    @torch.no_grad()
    def update(self, preds, labels, *args, **kwargs):
        """Update metric from predictions and labels."""
        pass

    @torch.no_grad()
    def reset(self):
        """Reset the metric."""
        pass

    @torch.no_grad()
    def result(self) -> Dict:
        """Get the metric(s) with name in dict format."""
        ret = []

        # Count in observed set
        ret.append(
            (get_metric_tag(TAG_BATCH, action_mode=self.mode, base_metric_name=f"{self.observed_set_name}_count"),
             len(self.observed_set))
        )

        if self.ref_set is not None:
            intersect_set = {x for x in self.observed_set if x in self.ref_set}

            # Count in reference set
            ret.append(
                (get_metric_tag(TAG_BATCH, action_mode=self.mode, base_metric_name=f"{self.ref_set_name}_count"),
                 len(self.ref_set))
            )

            # Count of intersection
            ret.append(
                (get_metric_tag(TAG_BATCH, action_mode=self.mode,
                                base_metric_name=f"intersect_{self.observed_set_name}_VS_{self.ref_set_name}_count"),
                 len(intersect_set))
            )

            # Fraction intersection/observed set
            if len(self.observed_set) > 0:
                ret.append((
                    get_metric_tag(
                        TAG_BATCH, action_mode=self.mode, base_metric_name=
                        f"intersect_{self.observed_set_name}_VS_{self.ref_set_name}_{self.observed_set_name}-fract"),
                    len(intersect_set) / len(self.observed_set)
                ))

            # Fraction intersection/ref set
            if len(self.ref_set) > 0:
                ret.append((
                    get_metric_tag(
                        TAG_BATCH, action_mode=self.mode, base_metric_name=
                        f"intersect_{self.observed_set_name}_VS_{self.ref_set_name}_{self.ref_set_name}-fract"),
                    len(intersect_set) / len(self.ref_set)
                ))

        return {k: v for k, v in ret}

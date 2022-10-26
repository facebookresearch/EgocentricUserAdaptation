import unittest

from continual_ego4d.metrics.past_metrics import ConditionalOnlineForgettingMetric
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from continual_ego4d.metrics.meters import ConditionalAverageMeterDict

class PlotConditionalOnlineForgettingMetricCase(unittest.TestCase):

    def setUp(self) -> None:
        self.metric = ConditionalOnlineForgettingMetric(
            base_metric_mode='acc', action_mode="action",
            do_plot=True,

            # Optional
            main_metric_name="TESTCASE_METRIC",
            k=1,
        )

        self.metric.action_results_over_time = {
            "prev_batch_idx": defaultdict(list),  # <action, list(<prev_iter,current_iter,delta_value>)
            "current_batch_idx": defaultdict(list),
            "delta": defaultdict(list),
        }

        # for action, prev_batch_idx, current_batch_idx, delta in [
        #     (0, 1, 2, 10), (1, 1, 2, 20), (3, 1, 20, 30), (10, 15, 20, -10)
        # ]:
        #     self.metric.action_results_over_time['prev_batch_idx'][action].append(prev_batch_idx)
        #     self.metric.action_results_over_time['current_batch_idx'][action].append(current_batch_idx)
        #     self.metric.action_results_over_time['delta'][action].append(delta)

        self.fake_stream = [
            torch.LongTensor([(0,2)])
        ]

    def test_updates_action_subsequent_actions(self):
        self.fake_stream = [
            torch.LongTensor(
                [(0,2)])
        ]

    def test_updated_dict(self):
        d = ConditionalAverageMeterDict()

        d.update(True,True)
        d.update(True,[True])
        d.update([False],True)
        d.update([False],[True])

        self.assertEqual(d.result(),0.5)


if __name__ == '__main__':
    unittest.main()

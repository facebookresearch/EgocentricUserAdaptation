import unittest

from continual_ego4d.metrics.past_metrics import ConditionalOnlineForgettingMetric
from collections import defaultdict
import matplotlib.pyplot as plt


class PlotConditionalOnlineForgettingMetricCase(unittest.TestCase):

    def setUp(self) -> None:
        self.metric = ConditionalOnlineForgettingMetric(
            base_metric_mode='acc', action_mode="action",
            keep_action_results_over_time=True, do_plot=True,

            # Optional
            main_metric_name="TESTCASE_METRIC",
            is_current_batch_cond_set=False,
            in_cond_set=False,
            k=1,
        )

        self.metric.action_results_over_time = {
            "prev_batch_idx": defaultdict(list),  # <action, list(<prev_iter,current_iter,delta_value>)
            "current_batch_idx": defaultdict(list),
            "delta": defaultdict(list),
        }

        for action, prev_batch_idx, current_batch_idx, delta in [
            (0, 1, 2, 10), (1, 1, 2, 20), (3, 1, 20, 30), (10, 15, 20, -10)
        ]:
            self.metric.action_results_over_time['prev_batch_idx'][action].append(prev_batch_idx)
            self.metric.action_results_over_time['current_batch_idx'][action].append(current_batch_idx)
            self.metric.action_results_over_time['delta'][action].append(delta)

    def test_plotting(self):
        fig_dict = self.metric.plot()

        for figure in fig_dict.values():
            figure.show()
            plt.clf()


if __name__ == '__main__':
    unittest.main()

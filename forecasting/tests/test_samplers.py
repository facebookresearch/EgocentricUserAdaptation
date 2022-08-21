import unittest

from continual_ego4d.tasks.continual_action_recog_task import PastSampler, FutureSampler


class PastSamplerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.seen_action_to_stream_idxs = {}
        offset = 0
        for action, action_len in [
            ((0, 0), 10),
            ((0, 1), 10),
        ]:
            self.seen_action_to_stream_idxs[action] = [x + offset for x in list(range(action_len))]
            offset += action_len
        self.all_past_idxs = list(range(offset))

    def test_full(self):
        total_mem_capacity = 10

        self.sampler = PastSampler(
            mode="full",
            seen_action_to_stream_idxs=self.seen_action_to_stream_idxs,
            total_capacity=total_mem_capacity,
        )

        ret = self.sampler(self.all_past_idxs)
        self.assertEqual(ret, self.all_past_idxs, "Should be equal for full mode even when limited mem_capcity")

    def test_uniform_action_uniform_instance(self):
        total_mem_capacity = 10

        self.sampler = PastSampler(
            mode="uniform_action_uniform_instance",
            seen_action_to_stream_idxs=self.seen_action_to_stream_idxs,
            total_capacity=total_mem_capacity,
        )

        ret = self.sampler(self.all_past_idxs)

        for el in ret:
            self.assertIsInstance(el, int)

        self.assertEqual(len(ret), total_mem_capacity)
        self.assertEqual(len(set(ret)), len(ret), "Only unique idxs!")


class FutureSamplerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.total_mem_capacity = 10

        self.past_stream = [(0, 2)] * 100
        self.future_stream = [(0, 0)] * 4 + [(0, 1)] * 5 + [(0, 0)] * 10
        self.stream_idx_to_action_list = self.past_stream + self.future_stream
        self.all_future_idxs = list(range(len(self.past_stream), len(self.stream_idx_to_action_list)))

        self.seen_action_set_all_seen = set(self.stream_idx_to_action_list)

    def test_full(self):
        sampler = FutureSampler(
            mode='full',
            stream_idx_to_action_list=self.stream_idx_to_action_list,
            seen_action_set=self.seen_action_set_all_seen,
            total_capacity=self.total_mem_capacity,
        )

        ret = sampler(
            all_future_idxs=self.all_future_idxs
        )

        self.assertEqual(ret, self.all_future_idxs)

    def test_split_seen_unseen(self):
        sampler = FutureSampler(
            mode='FIFO_split_seen_unseen',
            stream_idx_to_action_list=self.stream_idx_to_action_list,
            seen_action_set=self.seen_action_set_all_seen,
            total_capacity=self.total_mem_capacity,
        )

        ret = sampler(
            all_future_idxs=self.all_future_idxs
        )

        for el in ret:
            self.assertIsInstance(el, int)
        self.assertEqual(len(ret), self.total_mem_capacity)


if __name__ == '__main__':
    unittest.main()

import unittest

from forecasting.ego4d.config.defaults import get_cfg, convert_cfg_to_dict
import pprint


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.default_cfg = get_cfg()

    def test_config_to_dict(self):
        cfg_dict = convert_cfg_to_dict(self.default_cfg)

        print(pprint.pprint(cfg_dict))


if __name__ == '__main__':
    unittest.main()

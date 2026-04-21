from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapcal.config import TrainingConfig


class ConfigTestCase(unittest.TestCase):
    def test_training_config_extends_base(self):
        config = TrainingConfig.from_json(ROOT / "configs/training/vit_b16_segmented.json")
        self.assertEqual(config.model_name, "vit_b16")
        self.assertEqual(config.dataset_variant, "segmented")
        self.assertEqual(config.batch_size, 64)


if __name__ == "__main__":
    unittest.main()

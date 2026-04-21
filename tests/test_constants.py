from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapcal.constants import FOOD101_CLASSES, MANIFEST_COLUMNS, USDA_MAPPING_COLUMNS


class ConstantsTestCase(unittest.TestCase):
    def test_food101_class_count(self):
        self.assertEqual(len(FOOD101_CLASSES), 101)
        self.assertEqual(len(set(FOOD101_CLASSES)), 101)

    def test_schema_columns_are_stable(self):
        self.assertIn("image_path", MANIFEST_COLUMNS)
        self.assertIn("food101_class", USDA_MAPPING_COLUMNS)


if __name__ == "__main__":
    unittest.main()

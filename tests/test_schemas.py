from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapcal.schemas import NutritionFacts


class SchemasTestCase(unittest.TestCase):
    def test_scaled_nutrition(self):
        nutrition = NutritionFacts(
            serving_size_g=100.0,
            serving_unit="serving",
            calories_kcal=200.0,
            protein_g=10.0,
            carbs_g=15.0,
            fat_g=5.0,
        )
        scaled = nutrition.scaled(2.0)
        self.assertEqual(scaled.calories_kcal, 400.0)
        self.assertEqual(scaled.protein_g, 20.0)


if __name__ == "__main__":
    unittest.main()

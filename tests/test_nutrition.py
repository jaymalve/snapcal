from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapcal.nutrition import NutritionLookup


class NutritionTestCase(unittest.TestCase):
    def test_build_response_scales_macros(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "mapping.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "food101_class,usda_food_name,fdc_id,serving_size_g,serving_unit,calories_kcal,protein_g,carbs_g,fat_g,mapping_confidence,notes",
                        "apple_pie,Apple Pie,local-1,125,serving,320,2.5,46,14,manual,Example row",
                    ]
                ),
                encoding="utf-8",
            )
            lookup = NutritionLookup.from_csv(csv_path)
            response = lookup.build_response(
                ranked_predictions=[("apple_pie", 0.91)],
                portion_multiplier=1.5,
                model_version="test-bundle",
            )
            self.assertEqual(response.selected_class, "apple_pie")
            self.assertAlmostEqual(response.top_predictions[0].nutrition_adjusted.calories_kcal or 0.0, 480.0)


if __name__ == "__main__":
    unittest.main()

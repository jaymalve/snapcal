import unittest

from snapcal.nutrition import build_adjusted_nutrition, build_requested_portion
from snapcal.schemas import NutritionFacts


class PortionMathTests(unittest.TestCase):
    def test_standard_serving_request(self) -> None:
        requested = build_requested_portion("serving", None)

        self.assertEqual(requested.label, "Standard serving")
        self.assertIsNone(requested.grams)
        self.assertFalse(requested.approximate)

    def test_solid_request_formats_pounds_for_large_values(self) -> None:
        requested = build_requested_portion("oz", 24)

        self.assertEqual(requested.label, "24 oz (1 lb 8 oz)")
        self.assertEqual(requested.grams, 680.39)
        self.assertFalse(requested.approximate)

    def test_fluid_ounce_request_is_marked_approximate(self) -> None:
        requested = build_requested_portion("fl_oz", 8)

        self.assertEqual(requested.label, "8 fl oz")
        self.assertEqual(requested.grams, 236.59)
        self.assertTrue(requested.approximate)

    def test_non_standard_portion_scales_from_requested_grams(self) -> None:
        per_serving = NutritionFacts(
            serving_size_g=100.0,
            serving_unit="cup",
            calories_kcal=200.0,
            protein_g=10.0,
            carbs_g=20.0,
            fat_g=5.0,
        )

        adjusted = build_adjusted_nutrition(per_serving, build_requested_portion("oz", 8))

        self.assertEqual(adjusted.serving_size_g, 226.8)
        self.assertEqual(adjusted.serving_unit, "8 oz")
        self.assertEqual(adjusted.calories_kcal, 453.6)
        self.assertEqual(adjusted.protein_g, 22.68)
        self.assertEqual(adjusted.carbs_g, 45.36)
        self.assertEqual(adjusted.fat_g, 11.34)

    def test_non_standard_portion_preserves_requested_grams_when_mapping_is_incomplete(self) -> None:
        per_serving = NutritionFacts(
            serving_size_g=None,
            serving_unit="serving",
            calories_kcal=None,
            protein_g=None,
            carbs_g=None,
            fat_g=None,
        )

        adjusted = build_adjusted_nutrition(per_serving, build_requested_portion("fl_oz", 4))

        self.assertEqual(adjusted.serving_size_g, 118.29)
        self.assertEqual(adjusted.serving_unit, "4 fl oz")
        self.assertIsNone(adjusted.calories_kcal)


if __name__ == "__main__":
    unittest.main()

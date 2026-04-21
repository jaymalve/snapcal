from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapcal.evaluation import summarize_predictions


class EvaluationTestCase(unittest.TestCase):
    def test_summary_metrics(self):
        labels = [0, 1, 2]
        ranked_predictions = [
            [0, 2, 1],
            [2, 1, 0],
            [2, 0, 1],
        ]
        report = summarize_predictions(labels, ranked_predictions, class_names=("a", "b", "c"))
        self.assertAlmostEqual(report.top1_accuracy, 0.666667, places=5)
        self.assertAlmostEqual(report.top5_accuracy, 1.0, places=5)
        self.assertEqual(report.sample_count, 3)


if __name__ == "__main__":
    unittest.main()

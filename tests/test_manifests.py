from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapcal.manifests import build_manifest_rows


class ManifestTestCase(unittest.TestCase):
    def test_build_manifest_rows_creates_train_val_test(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "food-101"
            meta_root = dataset_root / "meta"
            meta_root.mkdir(parents=True)
            (meta_root / "train.txt").write_text(
                "apple_pie/100\napple_pie/101\nbaby_back_ribs/200\nbaby_back_ribs/201\n",
                encoding="utf-8",
            )
            (meta_root / "test.txt").write_text(
                "apple_pie/300\nbaby_back_ribs/400\n",
                encoding="utf-8",
            )
            rows = build_manifest_rows(dataset_root=dataset_root, val_ratio=0.5, seed=7)
            splits = {row.split for row in rows}
            self.assertEqual(splits, {"train", "val", "test"})
            self.assertEqual(len(rows), 6)


if __name__ == "__main__":
    unittest.main()

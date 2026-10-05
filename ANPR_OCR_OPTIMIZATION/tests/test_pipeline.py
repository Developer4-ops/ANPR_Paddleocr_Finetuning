import unittest
import tempfile
import shutil
import pandas as pd
from pathlib import Path

from dataset_pipeline.ingestion import PredictionRecord
from dataset_pipeline.plate_validator import IndianPlateValidator
from dataset_pipeline.confidence_sorter import ConfidenceSorter, Tier
from dataset_pipeline.label_manager import LabelManager
from dataset_pipeline.dataset_builder import DatasetBuilder
from evaluation.metrics import OCRMetrics

class TestANPROCRPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_prediction_record(self):
        rec = PredictionRecord("p1.jpg", Path("/tmp/p1.jpg"), "ts09ab1234", 0.95)
        self.assertEqual(rec.prediction, "TS09AB1234")
        self.assertEqual(rec.confidence, 0.95)

    def test_indian_plate_validator(self):
        validator = IndianPlateValidator()
        
        # Valid formats
        v1 = validator.validate("TS09AB1234")
        self.assertTrue(v1.is_valid)
        self.assertEqual(v1.format_type, "standard")
        self.assertEqual(v1.state_code, "TS")

        v2 = validator.validate("21BH1234AB")
        self.assertTrue(v2.is_valid)
        self.assertEqual(v2.format_type, "bh_series")

        # Invalid format (O instead of 0)
        v3 = validator.validate("TSO9AB1234")
        self.assertFalse(v3.is_valid)
        self.assertEqual(v3.reason, "pattern_mismatch")

    def test_confidence_sorter(self):
        sorter = ConfidenceSorter(auto_accept_threshold=0.90, reject_threshold=0.70)
        
        # Priority sort candidate
        r1 = PredictionRecord("p1.jpg", Path("/tmp/p1.jpg"), "TS09AB1234", 0.95)
        s1 = sorter.sort_record(r1)
        self.assertEqual(s1.tier, Tier.AUTO_ACCEPT)

        # Queue candidate (lower confidence)
        r2 = PredictionRecord("p2.jpg", Path("/tmp/p2.jpg"), "TS09AB1234", 0.85)
        s2 = sorter.sort_record(r2)
        self.assertEqual(s2.tier, Tier.REVIEW_QUEUE)

    def test_label_manager_and_ground_truth(self):
        labels_dir = self.temp_dir / "labels"
        mgr = LabelManager(str(labels_dir))

        # Create dummy image
        img_path = self.temp_dir / "plate1.jpg"
        img_path.write_text("dummy image content")

        # Create queue & approve
        rec = PredictionRecord("plate1.jpg", img_path, "TS09AB1234", 0.96)
        sorter = ConfidenceSorter()
        s_res = sorter.sort_record(rec)

        mgr.create_review_queue([s_res])
        self.assertTrue(mgr.review_queue_csv.exists())

        mgr.approve_batch(auto_approve_high_confidence=True, min_confidence=0.90)
        self.assertTrue(mgr.approved_labels_csv.exists())

        df_gt = mgr.compile_final_ground_truth()
        self.assertTrue(mgr.final_ground_truth_csv.exists())
        self.assertEqual(len(df_gt), 1)
        self.assertEqual(df_gt.iloc[0]["ground_truth_label"], "TS09AB1234")

    def test_dataset_builder_splits(self):
        data_base = self.temp_dir / "data"
        builder = DatasetBuilder(base_dir=str(data_base), train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

        # Create 10 dummy samples
        rows = []
        for i in range(10):
            p = self.temp_dir / f"p{i}.jpg"
            p.write_text("dummy")
            rows.append({"image_path": str(p), "ground_truth_label": f"TS09AB120{i}"})

        df_gt = pd.DataFrame(rows)
        splits = builder.build_dataset(df_gt)

        self.assertTrue(splits["train_txt"].exists())
        self.assertTrue(splits["val_txt"].exists())
        self.assertTrue(splits["test_txt"].exists())
        self.assertTrue(splits["character_dict"].exists())

        # Check character dictionary restriction (A-Z, 0-9)
        dict_content = splits["character_dict"].read_text()
        for char in dict_content.splitlines():
            self.assertTrue(char.isalnum())
            self.assertTrue(char.isupper() or char.isdigit())

    def test_ocr_metrics(self):
        metrics_calc = OCRMetrics()
        preds = ["TS09AB1234", "MH12DE1433", "TSO9AB1234"]
        targets = ["TS09AB1234", "MH12DE1433", "TS09AB1234"]
        confs = [0.95, 0.90, 0.85]
        lats = [12.5, 10.0, 15.0]
        
        res = metrics_calc.evaluate(preds, targets, confidences=confs, inference_times_ms=lats)
        self.assertEqual(res.total_plates, 3)
        self.assertEqual(res.exact_matches, 2)
        self.assertAlmostEqual(res.avg_confidence, 0.90, places=2)
        self.assertAlmostEqual(res.avg_inference_time_ms, 12.5, places=1)
        self.assertEqual(res.confusion_pair_stats["O<->0"]["errors"], 1)

if __name__ == "__main__":
    unittest.main()

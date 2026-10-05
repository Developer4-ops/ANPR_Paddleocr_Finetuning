"""
ANPR OCR Optimization Pipeline
evaluation/evaluator.py

Wrapper for running PaddleOCR models on validation and held-out test sets,
collecting predictions, measuring confidence scores and inference latency (ms).
"""

import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd

from .metrics import OCRMetrics, MetricsResult

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates baseline or fine-tuned PaddleOCR models against ground-truth manifests.
    Calculates Accuracy, CER, Average Confidence, and Average Inference Latency (ms).
    """

    def __init__(self, ocr_engine=None):
        self.ocr = ocr_engine
        self.metrics_calculator = OCRMetrics()

    def evaluate_manifest(self, manifest_txt_path: Path) -> Tuple[MetricsResult, pd.DataFrame]:
        """
        Reads train/val/test manifest, runs inference on image samples, and computes evaluation metrics.
        """
        if not manifest_txt_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_txt_path}")

        records = []
        with open(manifest_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    records.append((parts[0], parts[1]))

        predictions = []
        targets = []
        confidences = []
        latencies_ms = []
        results_data = []

        for img_path_str, target_label in records:
            pred_label, score, latency_ms = self._predict_image(Path(img_path_str))
            predictions.append(pred_label)
            targets.append(target_label)
            confidences.append(score)
            latencies_ms.append(latency_ms)

            results_data.append({
                "image_path": img_path_str,
                "target": target_label,
                "prediction": pred_label,
                "confidence": round(score, 4),
                "latency_ms": round(latency_ms, 2),
                "is_exact_match": pred_label == target_label
            })

        metrics = self.metrics_calculator.evaluate(
            predictions=predictions,
            targets=targets,
            confidences=confidences,
            inference_times_ms=latencies_ms
        )
        df_results = pd.DataFrame(results_data)

        return metrics, df_results

    def _predict_image(self, img_path: Path) -> Tuple[str, float, float]:
        """Runs single image inference, measuring latency in milliseconds."""
        t_start = time.perf_counter()
        score = 0.0
        text = ""

        if self.ocr is not None:
            try:
                result = self.ocr.ocr(str(img_path), cls=False)
                if result and result[0]:
                    res = result[0][0]
                    text = str(res[1][0]).strip().upper()
                    score = float(res[1][1])
            except Exception as e:
                logger.warning(f"PaddleOCR inference warning on {img_path}: {e}")

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        return text, score, latency_ms

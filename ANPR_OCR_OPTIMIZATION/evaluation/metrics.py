"""
ANPR OCR Optimization Pipeline
evaluation/metrics.py

Calculates OCR evaluation metrics:
- Exact Plate Accuracy (exact full string match)
- Character Accuracy (normalized total character matches)
- Character Error Rate (CER) via Levenshtein Edit Distance
- Average Model Prediction Confidence
- Average Inference Time per Image (milliseconds)
- Per-Confusion Pair Accuracy (specifically for target pairs like O vs 0, I vs 1, etc.)
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

try:
    import editdistance
    def levenshtein_dist(s1: str, s2: str) -> int:
        return editdistance.eval(s1, s2)
except ImportError:
    def levenshtein_dist(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return levenshtein_dist(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

logger = logging.getLogger(__name__)

TARGET_CONFUSION_PAIRS = [
    ("O", "0"),
    ("I", "1"),
    ("B", "8"),
    ("S", "5"),
    ("Z", "2"),
    ("G", "6"),
]


@dataclass
class MetricsResult:
    total_plates: int
    exact_matches: int
    plate_accuracy: float
    total_characters: int
    char_matches: int
    char_accuracy: float
    total_edit_distance: int
    cer: float
    avg_confidence: float
    avg_inference_time_ms: float
    confusion_pair_stats: Dict[str, Dict[str, int]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_plates": self.total_plates,
            "exact_matches": self.exact_matches,
            "plate_accuracy": round(self.plate_accuracy, 4),
            "total_characters": self.total_characters,
            "char_matches": self.char_matches,
            "char_accuracy": round(self.char_accuracy, 4),
            "total_edit_distance": self.total_edit_distance,
            "cer": round(self.cer, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_inference_time_ms": round(self.avg_inference_time_ms, 2),
            "confusion_pair_stats": self.confusion_pair_stats,
        }


class OCRMetrics:
    """
    Computes exact accuracy, character accuracy, CER, average confidence, latency, and target confusion metrics.
    """

    def __init__(self, confusion_pairs: List[Tuple[str, str]] = None):
        self.pairs = confusion_pairs or TARGET_CONFUSION_PAIRS

    def evaluate(
        self,
        predictions: List[str],
        targets: List[str],
        confidences: Optional[List[float]] = None,
        inference_times_ms: Optional[List[float]] = None
    ) -> MetricsResult:
        if len(predictions) != len(targets):
            raise ValueError(f"Length mismatch: {len(predictions)} preds vs {len(targets)} targets")

        total_plates = len(targets)
        if total_plates == 0:
            return MetricsResult(0, 0, 0.0, 0, 0, 0.0, 0, 0.0, 0.0, 0.0, {})

        exact_matches = 0
        total_chars = 0
        char_matches = 0
        total_edit_dist = 0

        # Targeted confusion stats
        pair_stats = {
            f"{c1}<->{c2}": {"occurrences": 0, "errors": 0}
            for c1, c2 in self.pairs
        }

        for pred, gt in zip(predictions, targets):
            pred_clean = str(pred).strip().upper()
            gt_clean = str(gt).strip().upper()

            if pred_clean == gt_clean:
                exact_matches += 1

            dist = levenshtein_dist(pred_clean, gt_clean)
            total_edit_dist += dist
            total_chars += len(gt_clean)

            matches = max(0, len(gt_clean) - dist)
            char_matches += matches

            for c1, c2 in self.pairs:
                pair_key = f"{c1}<->{c2}"
                gt_count = gt_clean.count(c1) + gt_clean.count(c2)
                if gt_count > 0:
                    pair_stats[pair_key]["occurrences"] += gt_count
                    min_len = min(len(pred_clean), len(gt_clean))
                    for i in range(min_len):
                        gt_char = gt_clean[i]
                        pred_char = pred_clean[i]
                        if gt_char in (c1, c2):
                            if (gt_char == c1 and pred_char == c2) or (gt_char == c2 and pred_char == c1):
                                pair_stats[pair_key]["errors"] += 1

        plate_acc = exact_matches / float(total_plates)
        char_acc = char_matches / float(total_chars) if total_chars > 0 else 0.0
        cer = total_edit_dist / float(total_chars) if total_chars > 0 else 0.0

        avg_conf = (sum(confidences) / len(confidences)) if confidences and len(confidences) > 0 else 0.0
        avg_latency = (sum(inference_times_ms) / len(inference_times_ms)) if inference_times_ms and len(inference_times_ms) > 0 else 0.0

        return MetricsResult(
            total_plates=total_plates,
            exact_matches=exact_matches,
            plate_accuracy=plate_acc,
            total_characters=total_chars,
            char_matches=char_matches,
            char_accuracy=char_acc,
            total_edit_distance=total_edit_dist,
            cer=cer,
            avg_confidence=avg_conf,
            avg_inference_time_ms=avg_latency,
            confusion_pair_stats=pair_stats
        )

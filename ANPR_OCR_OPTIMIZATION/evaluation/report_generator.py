"""
ANPR OCR Optimization Pipeline
evaluation/report_generator.py

Generates Markdown and JSON comparative reports for baseline vs. fine-tuned model runs:
- Baseline -> Fine Tuned -> Improvement %
- Average Confidence & Inference Latency (ms)
- Targeted Character Confusion Pair Error Reduction
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from .metrics import MetricsResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Produces human-readable markdown reports and structured JSON metrics comparisons.
    """

    def __init__(self, reports_dir: str = "experiments/reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_comparison_report(
        self,
        baseline_metrics: MetricsResult,
        finetuned_metrics: MetricsResult,
        experiment_name: str = "v1"
    ) -> Path:
        """
        Generates comparison markdown report with: Baseline -> Fine Tuned -> Improvement %
        """
        report_path = self.reports_dir / f"comparison_{experiment_name}.md"

        # Percentage Relative Improvements
        plate_acc_b = baseline_metrics.plate_accuracy * 100
        plate_acc_f = finetuned_metrics.plate_accuracy * 100
        plate_impr_pct = ((plate_acc_f - plate_acc_b) / plate_acc_b * 100) if plate_acc_b > 0 else 0.0

        char_acc_b = baseline_metrics.char_accuracy * 100
        char_acc_f = finetuned_metrics.char_accuracy * 100
        char_impr_pct = ((char_acc_f - char_acc_b) / char_acc_b * 100) if char_acc_b > 0 else 0.0

        cer_b = baseline_metrics.cer
        cer_f = finetuned_metrics.cer
        cer_impr_pct = ((cer_b - cer_f) / cer_b * 100) if cer_b > 0 else 0.0

        conf_b = baseline_metrics.avg_confidence
        conf_f = finetuned_metrics.avg_confidence
        conf_impr_pct = ((conf_f - conf_b) / conf_b * 100) if conf_b > 0 else 0.0

        lat_b = baseline_metrics.avg_inference_time_ms
        lat_f = finetuned_metrics.avg_inference_time_ms
        lat_diff = lat_f - lat_b

        content = f"""# ANPR OCR Fine-Tuning Performance Report ({experiment_name})

## 📊 Summary Metrics Comparison

| Metric | Baseline Model | Fine-Tuned Model ({experiment_name}) | Relative Improvement % |
|---|---|---|---|
| **Exact Plate Accuracy** | {plate_acc_b:.2f}% | {plate_acc_f:.2f}% | **{plate_impr_pct:+.2f}%** |
| **Character Accuracy** | {char_acc_b:.2f}% | {char_acc_f:.2f}% | **{char_impr_pct:+.2f}%** |
| **Character Error Rate (CER)** | {cer_b:.4f} | {cer_f:.4f} | **{cer_impr_pct:+.2f}% error reduction** |
| **Average Prediction Confidence** | {conf_b:.4f} | {conf_f:.4f} | **{conf_impr_pct:+.2f}%** |
| **Avg Inference Time per Image** | {lat_b:.2f} ms | {lat_f:.2f} ms | **{lat_diff:+.2f} ms** |
| Total Held-Out Test Samples | {baseline_metrics.total_plates} | {finetuned_metrics.total_plates} | - |

---

## 🔤 Targeted Character Confusion Analysis Report

Evaluation of priority confusion pairs (`O↔0`, `I↔1`, `B↔8`, `S↔5`, `Z↔2`, `G↔6`):

| Confusion Pair | Baseline Errors | Fine-Tuned Errors | Error Reduction % |
|---|---|---|---|
"""
        b_pairs = baseline_metrics.confusion_pair_stats
        f_pairs = finetuned_metrics.confusion_pair_stats

        for pair_key in b_pairs:
            b_err = b_pairs[pair_key]["errors"]
            f_err = f_pairs.get(pair_key, {}).get("errors", 0)
            diff_err = b_err - f_err
            pct_red = (diff_err / float(b_err) * 100) if b_err > 0 else 0.0
            content += f"| `{pair_key}` | {b_err} | {f_err} | **{pct_red:.1f}% reduction ({diff_err} fewer errors)** |\n"

        content += "\n---\n*Report generated automatically by ANPR OCR Optimization Platform*\n"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Save JSON artifact
        json_path = self.reports_dir / f"comparison_{experiment_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "experiment": experiment_name,
                "baseline": baseline_metrics.to_dict(),
                "finetuned": finetuned_metrics.to_dict(),
                "improvement_percentages": {
                    "plate_accuracy": round(plate_impr_pct, 2),
                    "character_accuracy": round(char_impr_pct, 2),
                    "cer_reduction": round(cer_impr_pct, 2),
                    "confidence_improvement": round(conf_impr_pct, 2)
                }
            }, f, indent=2)

        logger.info(f"Comparison report generated at {report_path}")
        return report_path

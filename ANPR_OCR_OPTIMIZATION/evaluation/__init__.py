"""
ANPR OCR Optimization Pipeline
evaluation/__init__.py

Exports all evaluation components.
"""

from .metrics import OCRMetrics, MetricsResult
from .confusion_analyzer import ConfusionAnalyzer
from .evaluator import ModelEvaluator
from .report_generator import ReportGenerator

__all__ = [
    "OCRMetrics",
    "MetricsResult",
    "ConfusionAnalyzer",
    "ModelEvaluator",
    "ReportGenerator",
]

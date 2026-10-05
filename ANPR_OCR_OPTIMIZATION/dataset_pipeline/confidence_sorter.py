"""
ANPR OCR Optimization Pipeline
dataset_pipeline/confidence_sorter.py

Three-Tier Sorting Logic:
- AUTO_ACCEPT: confidence >= auto_accept_threshold AND format-valid
- REVIEW_QUEUE: confidence between reject and auto_accept OR format invalid
- REJECT: confidence < reject_threshold AND format invalid
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from .ingestion import PredictionRecord
from .plate_validator import IndianPlateValidator, ValidationResult

logger = logging.getLogger(__name__)


class Tier(Enum):
    AUTO_ACCEPT = "auto_accept"
    REVIEW_QUEUE = "review_queue"
    REJECT = "reject"


@dataclass
class SorterResult:
    record: PredictionRecord
    tier: Tier
    validation: ValidationResult


class ConfidenceSorter:
    """
    Sorts predictions into three action tiers.
    """

    def __init__(
        self,
        auto_accept_threshold: float = 0.92,
        reject_threshold: float = 0.70,
        validator: IndianPlateValidator = None
    ):
        self.auto_accept_threshold = auto_accept_threshold
        self.reject_threshold = reject_threshold
        self.validator = validator or IndianPlateValidator()

    def sort_record(self, record: PredictionRecord) -> SorterResult:
        val = self.validator.validate(record.prediction)

        if record.confidence >= self.auto_accept_threshold and val.is_valid:
            tier = Tier.AUTO_ACCEPT
        elif record.confidence < self.reject_threshold and not val.is_valid:
            tier = Tier.REJECT
        else:
            tier = Tier.REVIEW_QUEUE

        return SorterResult(record=record, tier=tier, validation=val)

    def sort_records(self, records: List[PredictionRecord]) -> Dict[Tier, List[SorterResult]]:
        results = {
            Tier.AUTO_ACCEPT: [],
            Tier.REVIEW_QUEUE: [],
            Tier.REJECT: []
        }

        for r in records:
            res = self.sort_record(r)
            results[res.tier].append(res)

        logger.info(
            f"Sorting completed: {len(results[Tier.AUTO_ACCEPT])} auto-accepted, "
            f"{len(results[Tier.REVIEW_QUEUE])} in review queue, "
            f"{len(results[Tier.REJECT])} rejected."
        )
        return results

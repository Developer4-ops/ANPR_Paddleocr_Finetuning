"""
ANPR OCR Optimization Pipeline
dataset_pipeline/plate_validator.py

Validates prediction text structure according to Indian license plate conventions:
- Standard format: State(2) District(2) Series(1-3) Number(4) (e.g. TS09AB1234)
- BH-Series (2021+): YY BH NNNN XX (e.g. 21BH1234AB)
- Validates state codes against standard list.
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "GA", "GJ", "HR", "HP",
    "JK", "JH", "KA", "KL", "LD", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PY",
    "PB", "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "WB"
}


@dataclass
class ValidationResult:
    text: str
    is_valid: bool
    format_type: str  # "standard", "bh_series", or "invalid"
    state_code: Optional[str]
    district_code: Optional[str]
    series: Optional[str]
    number: Optional[str]
    reason: Optional[str] = None


class IndianPlateValidator:
    """
    Regex-based validator for Indian registration number patterns.
    """

    def __init__(self, valid_state_codes: Optional[List[str]] = None):
        self.state_codes = set(valid_state_codes) if valid_state_codes else DEFAULT_STATE_CODES

        # Pattern 1: Standard (e.g., TS09AB1234, DL1C1234, KA01M1234)
        self.standard_pattern = re.compile(r"^([A-Z]{2})([0-9]{2})([A-Z]{1,3})([0-9]{4})$")
        
        # Pattern 2: BH Series (e.g., 22BH1234A, 21BH9999AB)
        self.bh_pattern = re.compile(r"^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$")

    def validate(self, text: str) -> ValidationResult:
        clean_text = text.strip().upper().replace(" ", "").replace("-", "")

        if not clean_text:
            return ValidationResult(
                text=text, is_valid=False, format_type="invalid",
                state_code=None, district_code=None, series=None, number=None,
                reason="empty_string"
            )

        # Standard check
        match_std = self.standard_pattern.match(clean_text)
        if match_std:
            state, district, series, number = match_std.groups()
            if state not in self.state_codes:
                return ValidationResult(
                    text=clean_text, is_valid=False, format_type="standard",
                    state_code=state, district_code=district, series=series, number=number,
                    reason=f"invalid_state_code_{state}"
                )
            return ValidationResult(
                text=clean_text, is_valid=True, format_type="standard",
                state_code=state, district_code=district, series=series, number=number
            )

        # BH series check
        match_bh = self.bh_pattern.match(clean_text)
        if match_bh:
            year, number, series = match_bh.groups()
            return ValidationResult(
                text=clean_text, is_valid=True, format_type="bh_series",
                state_code="BH", district_code=year, series=series, number=number
            )

        return ValidationResult(
            text=clean_text, is_valid=False, format_type="invalid",
            state_code=None, district_code=None, series=None, number=None,
            reason="pattern_mismatch"
        )

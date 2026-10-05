"""
ANPR OCR Optimization Pipeline
dataset_pipeline/ingestion.py

Handles reading Hema's ANPR output format (images + predictions JSON)
into a structured internal representation for the pipeline.

Input contract (from Hema):
  - Cropped plate images in: data/input/images/
  - Predictions JSON: data/input/predictions.json

Predictions JSON format (single or list):
  {
    "image": "plate_001.jpg",
    "prediction": "TSO9AB1234",
    "confidence": 0.82,
    "camera_id": "CAM_001",
    "timestamp": "2026-09-28T10:30:00+05:30",
    "vehicle_type": "car"
  }
"""

import json
import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """
    Represents one input record from Hema's ANPR system.
    This is the atomic unit flowing through the entire pipeline.
    """
    image_filename: str           # e.g. plate_001.jpg
    image_path: Path              # Absolute path to the image
    prediction: str               # Hema's OCR output e.g. "TSO9AB1234"
    confidence: float             # Model confidence score [0.0, 1.0]
    camera_id: Optional[str] = None
    timestamp: Optional[str] = None
    vehicle_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize prediction to uppercase — plates are always uppercase."""
        self.prediction = self.prediction.strip().upper()
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be in [0.0, 1.0], got {self.confidence} "
                f"for image {self.image_filename}"
            )


class DataIngestion:
    """
    Reads Hema's ANPR output (images + predictions JSON) and produces
    a list of PredictionRecord objects for downstream pipeline stages.

    Usage:
        ingestion = DataIngestion(
            images_dir="data/input/images",
            predictions_json="data/input/predictions.json"
        )
        records = ingestion.load()
        print(f"Loaded {len(records)} records")
    """

    REQUIRED_FIELDS = {"image", "prediction", "confidence"}

    def __init__(self, images_dir: str, predictions_json: str):
        """
        Args:
            images_dir: Path to directory or zip file containing cropped plate images.
            predictions_json: Path to Hema's predictions JSON file.
        """
        self.images_dir = Path(images_dir)
        self.effective_images_dir = self.images_dir
        self.predictions_json = Path(predictions_json)
        self._records: List[PredictionRecord] = []
        self._temp_dir = None

    def load(self) -> List[PredictionRecord]:
        """
        Main entry point. Reads predictions JSON and resolves image paths.

        Returns:
            List of PredictionRecord objects, one per valid plate.

        Raises:
            FileNotFoundError: If images_dir or predictions_json don't exist.
            ValueError: If JSON structure is invalid.
        """
        self._validate_paths()
        raw_records = self._read_predictions_json()
        self._records = []

        skipped = 0
        for raw in raw_records:
            try:
                record = self._parse_record(raw)
                self._records.append(record)
            except (KeyError, ValueError, FileNotFoundError) as e:
                logger.warning(f"Skipping record — {e}")
                skipped += 1

        logger.info(
            f"Ingestion complete: {len(self._records)} loaded, "
            f"{skipped} skipped."
        )
        return self._records

    def load_from_dict(self, records: List[Dict[str, Any]]) -> List[PredictionRecord]:
        """
        Alternative loader for programmatic use (e.g., streaming from Hema's API).

        Args:
            records: List of raw prediction dictionaries.

        Returns:
            List of PredictionRecord objects.
        """
        self._records = []
        skipped = 0
        for raw in records:
            try:
                record = self._parse_record(raw)
                self._records.append(record)
            except (KeyError, ValueError, FileNotFoundError) as e:
                logger.warning(f"Skipping record — {e}")
                skipped += 1
        logger.info(
            f"Ingestion (dict mode): {len(self._records)} loaded, "
            f"{skipped} skipped."
        )
        return self._records

    def get_records(self) -> List[PredictionRecord]:
        """Returns previously loaded records."""
        return self._records

    def summary(self) -> Dict[str, Any]:
        """
        Returns a summary dict of the loaded dataset.

        Returns:
            Dict with counts, confidence stats, and missing image info.
        """
        if not self._records:
            return {"total": 0}

        confidences = [r.confidence for r in self._records]
        cameras = list({r.camera_id for r in self._records if r.camera_id})

        return {
            "total_records": len(self._records),
            "confidence_mean": round(sum(confidences) / len(confidences), 4),
            "confidence_min": round(min(confidences), 4),
            "confidence_max": round(max(confidences), 4),
            "unique_cameras": cameras,
            "unique_predictions": len({r.prediction for r in self._records}),
        }

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _validate_paths(self) -> None:
        """Validates that input paths exist before processing."""
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir}"
            )
        if self.images_dir.is_file() and zipfile.is_zipfile(self.images_dir):
            if self._temp_dir is None:
                self._temp_dir = tempfile.TemporaryDirectory()
                extract_path = Path(self._temp_dir.name)
                with zipfile.ZipFile(self.images_dir, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                if (extract_path / "images").exists() and (extract_path / "images").is_dir():
                    self.effective_images_dir = extract_path / "images"
                else:
                    self.effective_images_dir = extract_path
        if not self.predictions_json.exists():
            raise FileNotFoundError(
                f"Predictions JSON not found: {self.predictions_json}"
            )

    def _read_predictions_json(self) -> List[Dict[str, Any]]:
        """
        Reads the predictions JSON file. Supports both list and single-object format.
        """
        with open(self.predictions_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Single prediction — wrap in list for uniform handling
            return [data]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(
                f"predictions.json must be a JSON object or array, "
                f"got: {type(data).__name__}"
            )

    def _parse_record(self, raw: Dict[str, Any]) -> PredictionRecord:
        """
        Parses a single raw dict into a PredictionRecord.

        Raises:
            KeyError: If required fields are missing.
            FileNotFoundError: If the image file doesn't exist.
            ValueError: If confidence is out of range.
        """
        missing = self.REQUIRED_FIELDS - set(raw.keys())
        if missing:
            raise KeyError(f"Missing required fields: {missing}")

        image_filename = os.path.basename(raw["image"])
        image_path = self.effective_images_dir / image_filename

        if not image_path.exists():
            matches = list(self.effective_images_dir.rglob(image_filename))
            if matches:
                image_path = matches[0]
            else:
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

        # Collect any extra fields as metadata
        known_fields = {"image", "prediction", "confidence",
                        "camera_id", "timestamp", "vehicle_type"}
        extra = {k: v for k, v in raw.items() if k not in known_fields}

        return PredictionRecord(
            image_filename=image_filename,
            image_path=image_path,
            prediction=str(raw["prediction"]),
            confidence=float(raw["confidence"]),
            camera_id=raw.get("camera_id"),
            timestamp=raw.get("timestamp"),
            vehicle_type=raw.get("vehicle_type"),
            metadata=extra,
        )

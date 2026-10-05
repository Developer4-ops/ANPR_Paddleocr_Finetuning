"""
ANPR OCR Optimization Pipeline
dataset_pipeline/quality_checker.py

Filters bad images from raw intake before human verification or fine-tuning:
- Blur detection (Laplacian variance)
- Dimensional checks (width, height, aspect ratio)
- Brightness / Contrast validation
- Plate Visibility Check (detects tight crops, edge clipping, heavy occlusion, unreadable text)
- Duplicate detection (Perceptual Hashing using imagehash if available)
"""

import json
import logging
import shutil
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

from .ingestion import PredictionRecord

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    """Result of quality validation for a single image."""
    image_path: Path
    is_valid: bool
    blur_score: float
    width: int
    height: int
    aspect_ratio: float
    mean_brightness: float
    contrast_std: float
    phash: str
    rejection_reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "is_valid": self.is_valid,
            "blur_score": round(self.blur_score, 2),
            "width": self.width,
            "height": self.height,
            "aspect_ratio": round(self.aspect_ratio, 2),
            "mean_brightness": round(self.mean_brightness, 2),
            "contrast_std": round(self.contrast_std, 2),
            "phash": self.phash,
            "rejection_reasons": self.rejection_reasons,
        }


class ImageQualityChecker:
    """
    Evaluates license plate image quality, plate visibility/occlusion, and duplicate samples.
    """

    def __init__(
        self,
        min_blur_laplacian: float = 50.0,
        min_width: int = 60,
        min_height: int = 20,
        max_width: int = 2000,
        max_height: int = 800,
        min_aspect_ratio: float = 1.5,
        max_aspect_ratio: float = 10.0,
        min_brightness: float = 20.0,
        max_brightness: float = 235.0,
        min_contrast_std: float = 15.0,
        phash_distance_threshold: int = 8,
    ):
        self.min_blur = min_blur_laplacian
        self.min_w = min_width
        self.min_h = min_height
        self.max_w = max_width
        self.max_h = max_height
        self.min_ar = min_aspect_ratio
        self.max_ar = max_aspect_ratio
        self.min_bright = min_brightness
        self.max_bright = max_brightness
        self.min_contrast_std = min_contrast_std
        self.phash_threshold = phash_distance_threshold

        self._seen_hashes: List[Tuple[Any, Path]] = []

    def check_image(self, image_path: Path) -> QualityResult:
        """
        Runs full quality and plate visibility checks on a single image.
        """
        rejection_reasons = []

        if not image_path.exists():
            return QualityResult(
                image_path=image_path,
                is_valid=False,
                blur_score=0.0,
                width=0,
                height=0,
                aspect_ratio=0.0,
                mean_brightness=0.0,
                contrast_std=0.0,
                phash="",
                rejection_reasons=["file_not_found"]
            )

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return QualityResult(
                image_path=image_path,
                is_valid=False,
                blur_score=0.0,
                width=0,
                height=0,
                aspect_ratio=0.0,
                mean_brightness=0.0,
                contrast_std=0.0,
                phash="",
                rejection_reasons=["corrupt_image"]
            )

        h, w = img_bgr.shape[:2]
        aspect_ratio = w / float(h) if h > 0 else 0.0

        # Dimension checks
        if w < self.min_w or h < self.min_h:
            rejection_reasons.append(f"too_small_{w}x{h}")
        if w > self.max_w or h > self.max_h:
            rejection_reasons.append(f"too_large_{w}x{h}")
        if aspect_ratio < self.min_ar or aspect_ratio > self.max_ar:
            rejection_reasons.append(f"bad_aspect_ratio_{aspect_ratio:.2f}")

        # Gray conversion for blur & brightness
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Blur check (Laplacian variance)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < self.min_blur:
            rejection_reasons.append(f"blurry_score_{blur_score:.1f}")

        # Brightness check
        mean_brightness = float(np.mean(gray))
        if mean_brightness < self.min_bright:
            rejection_reasons.append(f"too_dark_{mean_brightness:.1f}")
        elif mean_brightness > self.max_bright:
            rejection_reasons.append(f"too_bright_{mean_brightness:.1f}")

        # Production Check: Plate Visibility & Occlusion Check
        contrast_std = float(np.std(gray))
        if contrast_std < self.min_contrast_std:
            rejection_reasons.append(f"low_contrast_occluded_{contrast_std:.1f}")

        # Border clipping / tight crop detection
        # Check if text characters touch the outer 2-pixel border
        border_mask = np.ones_like(gray, dtype=bool)
        border_mask[2:-2, 2:-2] = False
        border_edges = cv2.Canny(gray, 100, 200)[border_mask]
        if np.mean(border_edges) > 50:  # High edge density right at border
            rejection_reasons.append("cropped_too_tightly_clipped")

        # Perceptual hash for duplicate detection
        phash_str = ""
        if HAS_IMAGEHASH:
            try:
                pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                curr_hash = imagehash.phash(pil_img)
                phash_str = str(curr_hash)

                for existing_hash, existing_path in self._seen_hashes:
                    dist = curr_hash - existing_hash
                    if dist <= self.phash_threshold:
                        rejection_reasons.append(f"duplicate_of_{existing_path.name}_dist_{dist}")
                        break

                if not rejection_reasons:
                    self._seen_hashes.append((curr_hash, image_path))

            except Exception as e:
                logger.warning(f"Failed to calculate pHash for {image_path}: {e}")

        is_valid = len(rejection_reasons) == 0

        return QualityResult(
            image_path=image_path,
            is_valid=is_valid,
            blur_score=blur_score,
            width=w,
            height=h,
            aspect_ratio=aspect_ratio,
            mean_brightness=mean_brightness,
            contrast_std=contrast_std,
            phash=phash_str,
            rejection_reasons=rejection_reasons
        )

    def process_records(
        self, records: List[PredictionRecord]
    ) -> Tuple[List[PredictionRecord], List[Tuple[PredictionRecord, QualityResult]]]:
        """
        Filters a list of PredictionRecords into passed and rejected.
        """
        passed = []
        rejected = []

        for r in records:
            res = self.check_image(r.image_path)
            if res.is_valid:
                passed.append(r)
            else:
                rejected.append((r, res))

        logger.info(
            f"Quality Check completed: {len(passed)} passed, {len(rejected)} rejected"
        )
        return passed, rejected

    def filter_duplicates(
        self,
        quality_passed: List[PredictionRecord],
    ) -> Tuple[List[PredictionRecord], List[Tuple[PredictionRecord, str]]]:
        """
        Removes duplicate plate images from quality-passed records.

        Groups records by OCR prediction text. Within each group keeps the single
        best image ranked by:
          1. Highest OCR confidence.
          2. Highest Laplacian sharpness score (blur_score) as tie-breaker.

        Returns:
            unique_records  : one best record per plate text.
            duplicate_pairs : (record, rejection_reason) for all removed duplicates.
        """
        groups: Dict[str, List[PredictionRecord]] = {}
        for r in quality_passed:
            key = r.prediction.strip().upper()
            groups.setdefault(key, []).append(r)

        unique_records: List[PredictionRecord] = []
        duplicate_pairs: List[Tuple[PredictionRecord, str]] = []

        for pred_text, group in groups.items():
            if len(group) == 1:
                unique_records.append(group[0])
                continue

            # Compute sharpness for each record in the group
            scored = []
            for r in group:
                img = cv2.imread(str(r.image_path))
                blur = 0.0
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                scored.append((r, blur))

            # Sort: highest confidence first, then highest sharpness
            scored.sort(key=lambda x: (x[0].confidence, x[1]), reverse=True)

            best_record, _ = scored[0]
            unique_records.append(best_record)

            for r, _ in scored[1:]:
                reason = f"duplicate_plate:{pred_text}:kept={best_record.image_filename}"
                duplicate_pairs.append((r, reason))

        logger.info(
            f"Duplicate removal: {len(unique_records)} unique plates kept, "
            f"{len(duplicate_pairs)} duplicates removed."
        )
        return unique_records, duplicate_pairs

    def export_cleaned_and_rejected(
        self,
        unique_records: List[PredictionRecord],
        quality_rejected: List[Tuple[PredictionRecord, QualityResult]],
        duplicate_rejected: List[Tuple[PredictionRecord, str]],
        cleaned_images_dir: str,
        cleaned_predictions_path: str,
        rejected_blur_dir: str,
        rejected_bad_quality_dir: str,
        rejected_duplicate_dir: str,
        rejected_predictions_path: str,
    ) -> None:
        """
        Physically copies images into output folders and writes prediction.json manifests.

        Folder layout produced:
            data/cleaned/
                images/               <- unique, quality-passed plate images
                prediction.json       <- manifest for create-review stage
            data/rejected/
                blur/                 <- blurry images
                bad_quality/          <- tight-crop / low contrast / dimension failures
                duplicate/            <- duplicate plate images
                prediction.json       <- manifest of every rejected image + reason
        """
        cleaned_img_dir = Path(cleaned_images_dir)
        cleaned_img_dir.mkdir(parents=True, exist_ok=True)

        rej_blur = Path(rejected_blur_dir)
        rej_bad = Path(rejected_bad_quality_dir)
        rej_dup = Path(rejected_duplicate_dir)
        for d in (rej_blur, rej_bad, rej_dup):
            d.mkdir(parents=True, exist_ok=True)

        # ── Export cleaned unique images ──────────────────────────────────
        cleaned_manifest = []
        for r in unique_records:
            dest = cleaned_img_dir / r.image_filename
            if not dest.exists():
                shutil.copy2(str(r.image_path), str(dest))
            cleaned_manifest.append({
                "image": r.image_filename,
                "prediction": r.prediction,
                "confidence": r.confidence,
                "camera_id": r.camera_id,
                "timestamp": r.timestamp,
                "vehicle_type": r.vehicle_type,
            })

        with open(cleaned_predictions_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_manifest, f, indent=2)
        logger.info(
            f"Cleaned manifest written ({len(cleaned_manifest)} records) "
            f"-> {cleaned_predictions_path}"
        )

        # ── Export rejected images ────────────────────────────────────────
        rejected_manifest = []

        for r, res in quality_rejected:
            reasons = res.rejection_reasons
            is_blur = any("blurry" in rr for rr in reasons)
            dest_dir = rej_blur if is_blur else rej_bad
            dest = dest_dir / r.image_filename
            if not dest.exists():
                shutil.copy2(str(r.image_path), str(dest))
            rejected_manifest.append({
                "image": r.image_filename,
                "prediction": r.prediction,
                "confidence": r.confidence,
                "rejection_reason": ";".join(reasons),
                "rejected_to": str(dest_dir),
            })

        for r, reason in duplicate_rejected:
            dest = rej_dup / r.image_filename
            if not dest.exists():
                shutil.copy2(str(r.image_path), str(dest))
            rejected_manifest.append({
                "image": r.image_filename,
                "prediction": r.prediction,
                "confidence": r.confidence,
                "rejection_reason": reason,
                "rejected_to": str(rej_dup),
            })

        with open(rejected_predictions_path, "w", encoding="utf-8") as f:
            json.dump(rejected_manifest, f, indent=2)
        logger.info(
            f"Rejected manifest written ({len(rejected_manifest)} records) "
            f"-> {rejected_predictions_path}"
        )

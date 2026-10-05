"""
ANPR OCR Optimization Pipeline
dataset_pipeline/label_manager.py

Production Human-in-the-Loop Workflow:
Prediction -> Prioritization -> Human Verification -> Approved Labels -> Final Ground Truth

Files managed:
- review_queue.xlsx: Prioritized queue of predictions for human inspection (Excel workbook)
- approved_labels.csv: Human-verified/corrected labels
- final_ground_truth.csv: Sole source-of-truth dataset for model fine-tuning
"""

import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

import openpyxl
from openpyxl.styles import (
    Font, PatternFill, Alignment
)
from openpyxl.formatting.rule import FormulaRule
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

from .ingestion import PredictionRecord
from .confidence_sorter import SorterResult

logger = logging.getLogger(__name__)


class LabelManager:
    """
    Manages human-in-the-loop review queues, approved label storage, and final ground truth compilation.
    """

    def __init__(self, labels_dir: str):
        self.labels_dir = Path(labels_dir)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.review_queue_xlsx = self.labels_dir / "review_queue.xlsx"
        self.approved_labels_csv = self.labels_dir / "approved_labels.csv"
        self.final_ground_truth_csv = self.labels_dir / "final_ground_truth.csv"

    def create_review_queue(
        self,
        sorter_results: List[SorterResult],
        cleaned_images_dir: str = None,
    ) -> Path:
        """
        Creates a prioritized review_queue.xlsx Excel workbook for human review.
        Confidence is used ONLY for priority sorting (lowest confidence / invalid format first).

        Args:
            sorter_results: Sorted prediction records from ConfidenceSorter.
            cleaned_images_dir: Optional path to the cleaned images directory.
                                When provided, image_path cells will use this directory
                                so reviewers get paths to the quality-checked images.

        Sheet: 'Review Queue'
        Columns: image_filename, image_path, predicted_label, verified_label,
                 confidence, status, reject_reason, duplicate_flag, duplicate_of,
                 plate_valid, comments

        Features:
        - Clickable hyperlinks on image_path cells
        - Data validation dropdowns on status, reject_reason, duplicate_flag, plate_valid
        - Conditional row coloring: green=verified, yellow=corrected, red=rejected, blue=dup
        - Frozen header row (freeze_panes='A2')
        - Bold header row
        - Auto-sized column widths
        """
        cleaned_dir = Path(cleaned_images_dir) if cleaned_images_dir else None

        # ── 1. Build sorted row data ──────────────────────────────────────────
        queue_data = []
        for res in sorter_results:
            priority = (1.0 - res.record.confidence) + (0.5 if not res.validation.is_valid else 0.0)

            # Use cleaned image path when available; fall back to record's own path
            if cleaned_dir is not None:
                img_path = str(cleaned_dir / res.record.image_filename)
            else:
                img_path = str(res.record.image_path)

            queue_data.append({
                "image_filename": res.record.image_filename,
                "image_path": img_path,
                "predicted_label": res.record.prediction,
                "verified_label": "",
                "confidence": round(res.record.confidence, 4),
                "status": "pending",
                "reject_reason": "",
                "duplicate_flag": "NO",
                "duplicate_of": "",
                "plate_valid": "",
                "comments": "",
                "_priority": round(priority, 3),
            })

        queue_data.sort(key=lambda r: r["_priority"], reverse=True)
        for row in queue_data:
            del row["_priority"]

        # ── 2. Column definitions (order matters) ─────────────────────────────
        COLUMNS = [
            "image_filename",
            "image_path",
            "predicted_label",
            "verified_label",
            "confidence",
            "status",
            "reject_reason",
            "duplicate_flag",
            "duplicate_of",
            "plate_valid",
            "comments",
        ]

        # ── 3. Create workbook & sheet ─────────────────────────────────────────
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Review Queue"

        # ── 4. Header row ──────────────────────────────────────────────────────
        header_font = Font(bold=True)
        header_fill = PatternFill(fill_type="solid", fgColor="2D3748")   # dark slate
        header_font_color = Font(bold=True, color="FFFFFF")
        for col_idx, col_name in enumerate(COLUMNS, start=1):
            cell = ws.cell(row=1, column=col_idx, value=col_name)
            cell.font = header_font_color
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=False)

        # ── 5. Data rows ───────────────────────────────────────────────────────
        for row_idx, row_data in enumerate(queue_data, start=2):
            for col_idx, col_name in enumerate(COLUMNS, start=1):
                value = row_data.get(col_name, "")

                if col_name == "image_path" and value:
                    # Clickable hyperlink cell
                    cell = ws.cell(row=row_idx, column=col_idx, value=value)
                    cell.hyperlink = value
                    cell.font = Font(color="0563C1", underline="single")
                else:
                    cell = ws.cell(row=row_idx, column=col_idx, value=value)

                cell.alignment = Alignment(vertical="center")

        # ── 6. Data Validation dropdowns ──────────────────────────────────────
        total_rows = len(queue_data) + 1   # +1 for header
        max_row = max(total_rows, 2)       # at least row 2 even if empty

        def _make_dv(formula: str, col_letter: str, prompt_title: str, prompt_body: str) -> DataValidation:
            dv = DataValidation(
                type="list",
                formula1=formula,
                allow_blank=True,
                showDropDown=False,
                showInputMessage=True,
                promptTitle=prompt_title,
                prompt=prompt_body,
            )
            dv.sqref = f"{col_letter}2:{col_letter}{max_row}"
            return dv

        status_col   = get_column_letter(COLUMNS.index("status") + 1)
        reject_col   = get_column_letter(COLUMNS.index("reject_reason") + 1)
        dupflag_col  = get_column_letter(COLUMNS.index("duplicate_flag") + 1)
        pvalid_col   = get_column_letter(COLUMNS.index("plate_valid") + 1)

        dv_status = _make_dv(
            '"pending,verified,corrected,rejected"',
            status_col,
            "Status",
            "Select the review status for this record."
        )
        dv_reject = _make_dv(
            '"unreadable,blurry,partial_plate,wrong_crop,duplicate,not_plate,other"',
            reject_col,
            "Reject Reason",
            "Select a reject reason (only if status = rejected)."
        )
        dv_dupflag = _make_dv(
            '"YES,NO"',
            dupflag_col,
            "Duplicate Flag",
            "Mark YES if this plate is a duplicate of another record."
        )
        dv_pvalid = _make_dv(
            '"YES,NO"',
            pvalid_col,
            "Plate Valid",
            "Mark YES if the plate format is valid."
        )

        for dv in (dv_status, dv_reject, dv_dupflag, dv_pvalid):
            ws.add_data_validation(dv)

        # ── 7. Conditional Formatting ──────────────────────────────────────────
        last_col_letter = get_column_letter(len(COLUMNS))
        full_range = f"A2:{last_col_letter}{max_row}"

        # Colors
        green_fill  = PatternFill(fill_type="solid", fgColor="C6EFCE")   # verified
        yellow_fill = PatternFill(fill_type="solid", fgColor="FFEB9C")   # corrected
        red_fill    = PatternFill(fill_type="solid", fgColor="FFC7CE")   # rejected
        blue_fill   = PatternFill(fill_type="solid", fgColor="BDD7EE")   # duplicate

        # Formula references $F2 = status column (col 6), $H2 = duplicate_flag (col 8)
        status_ref   = f"${status_col}2"
        dupflag_ref  = f"${dupflag_col}2"

        ws.conditional_formatting.add(full_range, FormulaRule(
            formula=[f'{status_ref}="verified"'],   fill=green_fill))
        ws.conditional_formatting.add(full_range, FormulaRule(
            formula=[f'{status_ref}="corrected"'],  fill=yellow_fill))
        ws.conditional_formatting.add(full_range, FormulaRule(
            formula=[f'{status_ref}="rejected"'],   fill=red_fill))
        ws.conditional_formatting.add(full_range, FormulaRule(
            formula=[f'{dupflag_ref}="YES"'],        fill=blue_fill))

        # ── 8. Freeze header row ───────────────────────────────────────────────
        ws.freeze_panes = "A2"

        # ── 9. Column widths ───────────────────────────────────────────────────
        COLUMN_WIDTHS = {
            "image_filename":  20,
            "image_path":      42,
            "predicted_label": 18,
            "verified_label":  18,
            "confidence":      12,
            "status":          14,
            "reject_reason":   20,
            "duplicate_flag":  16,
            "duplicate_of":    20,
            "plate_valid":     12,
            "comments":        35,
        }
        for col_idx, col_name in enumerate(COLUMNS, start=1):
            ws.column_dimensions[get_column_letter(col_idx)].width = COLUMN_WIDTHS.get(col_name, 16)

        # ── 10. Row height for header ──────────────────────────────────────────
        ws.row_dimensions[1].height = 20

        # ── 11. Save workbook & CSV ───────────────────────────────────────────
        wb.save(self.review_queue_xlsx)
        df_queue = pd.DataFrame(queue_data, columns=COLUMNS)
        df_queue.to_csv(self.labels_dir / "review_queue.csv", index=False)
        logger.info(
            f"Generated review queue ({len(queue_data)} records) -> {self.review_queue_xlsx} and {self.labels_dir / 'review_queue.csv'}"
        )
        return self.review_queue_xlsx

    def approve_batch(self, auto_approve_high_confidence: bool = False, min_confidence: float = 0.95) -> Path:
        """
        Processes review_queue.xlsx (or legacy review_queue.csv) and outputs approved_labels.csv.
        If auto_approve_high_confidence is True, valid high confidence predictions are pre-approved.
        Humans verify/correct labels in review_queue.xlsx and set status to 'approved' or 'corrected'.
        """
        # Support both xlsx (new) and csv (legacy fallback)
        review_queue_csv_legacy = self.labels_dir / "review_queue.csv"
        if self.review_queue_xlsx.exists():
            df_queue = pd.read_excel(self.review_queue_xlsx, sheet_name="Review Queue", engine="openpyxl")
        elif review_queue_csv_legacy.exists():
            df_queue = pd.read_csv(review_queue_csv_legacy)
        else:
            logger.warning(
                f"Review queue file does not exist: {self.review_queue_xlsx}"
            )
            return self.approved_labels_csv

        if df_queue.empty:
            logger.warning("Review queue is empty.")
            return self.approved_labels_csv

        approved_rows = []
        for _, row in df_queue.iterrows():
            status = str(row.get("status", "pending")).lower()
            verified = str(row.get("verified_label", "")).strip().upper()
            predicted = str(row.get("predicted_label", "")).strip().upper()
            conf = float(row.get("confidence", 0.0))
            fmt_valid = bool(row.get("format_valid", False))

            if status in ["approved", "corrected"] and verified:
                approved_rows.append({
                    "image_path": row["image_path"],
                    "verified_label": verified,
                    "verification_source": "human_verified",
                    "status": status
                })
            elif auto_approve_high_confidence and conf >= min_confidence and fmt_valid:
                # Pre-approved candidates subject to human audit
                approved_rows.append({
                    "image_path": row["image_path"],
                    "verified_label": predicted,
                    "verification_source": "pre_approved_high_conf",
                    "status": "approved"
                })

        df_approved = pd.DataFrame(approved_rows)
        df_approved.to_csv(self.approved_labels_csv, index=False)
        logger.info(f"Approved labels exported ({len(df_approved)} records) -> {self.approved_labels_csv}")
        return self.approved_labels_csv

    def compile_final_ground_truth(self) -> pd.DataFrame:
        """
        Generates final_ground_truth.csv as the single source-of-truth for training dataset creation.
        Ensures strict data cleaning and alphanumeric normalization.
        """
        if not self.approved_labels_csv.exists():
            # If approved_labels does not exist yet, build from queue directly if marked approved
            self.approve_batch(auto_approve_high_confidence=True)

        df_approved = pd.read_csv(self.approved_labels_csv)
        if df_approved.empty:
            logger.warning("No approved labels available to compile final ground truth.")
            return pd.DataFrame(columns=["image_path", "ground_truth_label", "verification_source"])

        # Filter and clean final ground truth
        clean_rows = []
        for _, row in df_approved.iterrows():
            img_path = str(row["image_path"])
            label = str(row["verified_label"]).strip().upper()
            # Strict alphanumeric cleaning (A-Z, 0-9)
            cleaned_label = "".join([c for c in label if c.isalnum()])

            if Path(img_path).exists() and len(cleaned_label) > 0:
                clean_rows.append({
                    "image_path": img_path,
                    "ground_truth_label": cleaned_label,
                    "verification_source": row.get("verification_source", "human_verified")
                })

        df_gt = pd.DataFrame(clean_rows)
        df_gt.to_csv(self.final_ground_truth_csv, index=False)
        logger.info(f"Final Ground Truth compiled ({len(df_gt)} records) -> {self.final_ground_truth_csv}")

        return df_gt

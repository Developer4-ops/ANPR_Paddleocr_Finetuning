#!/usr/bin/env python3
"""
ANPR OCR Optimization Pipeline - Phase 2
scripts/create_ground_truth.py

Converts Human Verified Review Queue Excel workbook into final_ground_truth.csv.

Filtering Rules:
  - Include: status == 'verified' or status == 'corrected'
  - Exclude: status == 'rejected' or duplicate_flag == 'YES'

Validation:
  - Verify image file existence on disk
  - Remove empty labels
  - Remove duplicate labels
  - Log summary statistics (total, accepted, rejected, missing)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("create_ground_truth")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert human-verified Excel review queue into final_ground_truth.csv"
    )
    parser.add_argument(
        "--excel-path",
        type=str,
        default="/home/snr5/Desktop/ANPR_Human_verification/review_queue.xl.xlsx",
        help="Path to the human-verified review queue Excel file."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/ground_truth/final_ground_truth.csv",
        help="Path for the generated final_ground_truth.csv"
    )
    return parser.parse_args()


def process_ground_truth(excel_path: str, output_path: str):
    excel_file = Path(excel_path)
    out_file = Path(output_path)

    if not excel_file.exists():
        logger.error(f"Excel file does not exist: {excel_file}")
        sys.exit(1)

    logger.info(f"Reading human verified Excel file from: {excel_file}")
    df = pd.read_excel(excel_file, engine="openpyxl")

    total_rows = len(df)
    logger.info(f"Total rows loaded: {total_rows}")

    required_cols = ["image_filename", "image_path", "predicted_label", "verified_label", "status", "duplicate_flag"]
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing required column in Excel file: '{col}'")
            sys.exit(1)

    accepted_rows = []
    rejected_count = 0
    missing_images_count = 0
    empty_labels_count = 0
    seen_labels = set()
    duplicate_labels_count = 0

    for idx, row in df.iterrows():
        status = str(row.get("status", "")).strip().lower()
        duplicate_flag = str(row.get("duplicate_flag", "")).strip().upper()
        
        # Determine the effective ground truth label
        verified_val = row.get("verified_label")
        predicted_val = row.get("predicted_label")

        if pd.notna(verified_val) and str(verified_val).strip() != "":
            label = str(verified_val).strip().upper()
        elif pd.notna(predicted_val) and str(predicted_val).strip() != "":
            label = str(predicted_val).strip().upper()
        else:
            label = ""

        # Normalize label to remove whitespace / non-printable chars
        label = "".join([c for c in label if c.isalnum()])

        # Filtering rules
        is_included_status = status in ["verified", "corrected"]
        is_duplicate = duplicate_flag in ["YES", "Y", "TRUE", "1"]
        is_rejected_status = status == "rejected"

        if is_rejected_status or is_duplicate or not is_included_status:
            rejected_count += 1
            continue

        if not label:
            logger.warning(f"Row {idx} ({row.get('image_filename')}): Empty label after cleaning. Excluding.")
            empty_labels_count += 1
            rejected_count += 1
            continue

        # Check duplicate label
        if label in seen_labels:
            logger.warning(f"Row {idx} ({row.get('image_filename')}): Duplicate label '{label}' encountered. Excluding.")
            duplicate_labels_count += 1
            rejected_count += 1
            continue

        # Validate image file exists
        raw_image_path = str(row.get("image_path", "")).strip()
        img_path = Path(raw_image_path)
        
        # Check relative and absolute resolution
        if not img_path.exists():
            # Try resolving relative to workspace if needed
            alt_path = Path.cwd() / raw_image_path
            if alt_path.exists():
                img_path = Path(raw_image_path)
            else:
                logger.warning(f"Row {idx} ({row.get('image_filename')}): Image file not found at '{raw_image_path}'.")
                missing_images_count += 1
                rejected_count += 1
                continue

        seen_labels.add(label)
        accepted_rows.append({
            "image_path": str(img_path),
            "label": label
        })

    accepted_count = len(accepted_rows)

    # Create output directory
    out_file.parent.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(accepted_rows, columns=["image_path", "label"])
    df_out.to_csv(out_file, index=False)

    print("\n" + "=" * 50)
    print(" ANPR GROUND TRUTH EXTRACTION SUMMARY")
    print("=" * 50)
    print(f" Total rows in Excel:    {total_rows}")
    print(f" Accepted samples:       {accepted_count}")
    print(f" Rejected samples:       {rejected_count}")
    print(f" Missing images:         {missing_images_count}")
    print(f" Empty labels:           {empty_labels_count}")
    print(f" Duplicate labels:       {duplicate_labels_count}")
    print("=" * 50)
    print(f" Saved Ground Truth to:  {out_file.resolve()}")
    print("=" * 50 + "\n")

    return df_out


if __name__ == "__main__":
    args = parse_args()
    process_ground_truth(args.excel_path, args.output_path)

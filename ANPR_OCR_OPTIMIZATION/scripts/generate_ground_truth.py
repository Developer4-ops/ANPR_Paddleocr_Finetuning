"""
scripts/generate_ground_truth.py

Phase 2 — Ground Truth Generation Script
-----------------------------------------
Reads the human-verified review_queue.xlsx and produces:
    data/ground_truth/final_ground_truth.csv

Rules:
  INCLUDE:  status = verified  → use verified_label
            status = corrected → use verified_label (human correction)
  EXCLUDE:  status = rejected
            duplicate_flag = YES

Output format:
    image_path,label
"""

import sys
import pandas as pd
from pathlib import Path
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
XLSX_PATH      = PROJECT_ROOT / "data" / "labelled" / "review_queue.xlsx"
OUTPUT_PATH    = PROJECT_ROOT / "data" / "ground_truth" / "final_ground_truth.csv"
CLEANED_IMAGES = PROJECT_ROOT / "data" / "cleaned" / "images"
SHEET_NAME     = "Review Queue"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def clean_label(raw: str) -> str:
    """Strip whitespace, uppercase, keep only alphanumeric (A-Z, 0-9)."""
    return "".join(c for c in str(raw).strip().upper() if c.isalnum())


def main():
    # ── Load workbook ──────────────────────────────────────────────────────────
    print(f"Loading: {XLSX_PATH}")
    if not XLSX_PATH.exists():
        print(f"ERROR: {XLSX_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
    total_rows = len(df)

    # Normalise column values
    df["status"]         = df["status"].astype(str).str.strip().str.lower()
    df["duplicate_flag"] = df["duplicate_flag"].astype(str).str.strip().str.upper()
    df["verified_label"] = df["verified_label"].astype(str).str.strip()
    df["predicted_label"]= df["predicted_label"].astype(str).str.strip()

    # ── Status distribution (full picture) ────────────────────────────────────
    status_counts    = df["status"].value_counts(dropna=False)
    verified_count   = int(status_counts.get("verified",  0))
    corrected_count  = int(status_counts.get("corrected", 0))
    rejected_count   = int(status_counts.get("rejected",  0))
    pending_count    = int(status_counts.get("pending",   0))

    dup_yes_count = int((df["duplicate_flag"] == "YES").sum())

    # ── Filter: include only verified + corrected, exclude dup=YES ─────────────
    mask_include = df["status"].isin(["verified", "corrected"])
    mask_exclude_dup = df["duplicate_flag"] != "YES"
    df_accepted = df[mask_include & mask_exclude_dup].copy()

    # ── Build ground truth rows ────────────────────────────────────────────────
    rows = []
    missing_images  = []
    empty_labels    = []
    label_errors    = []

    for _, row in df_accepted.iterrows():
        raw_label = row["verified_label"]

        # For verified: verified_label should match predicted; use it.
        # For corrected: verified_label is the human correction.
        # In both cases we use verified_label.
        if raw_label in ("", "nan", "NaN", "None"):
            # Fallback: if human left it blank on a "verified" row, use predicted
            if row["status"] == "verified":
                raw_label = row["predicted_label"]
            else:
                empty_labels.append(row["image_filename"])
                continue

        label = clean_label(raw_label)
        if len(label) == 0:
            empty_labels.append(row["image_filename"])
            continue

        # Resolve image path
        img_path_str = str(row["image_path"]).strip()
        img_path_abs = PROJECT_ROOT / img_path_str

        if not img_path_abs.exists():
            # Try resolved via CLEANED_IMAGES / filename
            alt = CLEANED_IMAGES / row["image_filename"]
            if alt.exists():
                img_path_str = str(alt.relative_to(PROJECT_ROOT))
            else:
                missing_images.append(img_path_str)
                continue

        rows.append({
            "image_path": img_path_str,
            "label": label,
            "status": row["status"],
        })

    # ── Save CSV ───────────────────────────────────────────────────────────────
    df_gt = pd.DataFrame(rows)
    # Save with only the two required columns
    df_gt[["image_path", "label"]].to_csv(OUTPUT_PATH, index=False)
    final_count = len(df_gt)

    # ── Analytics ──────────────────────────────────────────────────────────────
    verified_in_gt  = int((df_gt["status"] == "verified").sum())
    corrected_in_gt = int((df_gt["status"] == "corrected").sum())

    # Character vocabulary
    all_chars = sorted(set("".join(df_gt["label"].tolist())))

    # Plate length distribution
    df_gt["length"] = df_gt["label"].str.len()
    length_dist = df_gt["length"].value_counts().sort_index().to_dict()

    # State code distribution (first 2 chars if alpha)
    def get_state(label):
        if len(label) >= 2 and label[:2].isalpha():
            return label[:2]
        return "OTHER"
    state_dist = Counter(df_gt["label"].apply(get_state))
    top_states = state_dist.most_common(15)

    # ── Print Report ───────────────────────────────────────────────────────────
    sep = "=" * 60
    print()
    print(sep)
    print("  PHASE 2 — GROUND TRUTH GENERATION REPORT")
    print(sep)
    print(f"  Source       : {XLSX_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Output       : {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(sep)
    print()
    print("── REVIEW QUEUE SUMMARY ──────────────────────────────────")
    print(f"  Total review rows     : {total_rows}")
    print(f"  Verified samples      : {verified_count}")
    print(f"  Corrected samples     : {corrected_count}")
    print(f"  Rejected samples      : {rejected_count}")
    print(f"  Pending samples       : {pending_count}")
    print(f"  Duplicate flag=YES    : {dup_yes_count}")
    print()
    print("── GROUND TRUTH OUTPUT ───────────────────────────────────")
    print(f"  Final training samples: {final_count}")
    print(f"    ↳ from verified      : {verified_in_gt}")
    print(f"    ↳ from corrected     : {corrected_in_gt}")
    print(f"  Missing images skipped: {len(missing_images)}")
    print(f"  Empty labels skipped  : {len(empty_labels)}")
    print()

    if missing_images:
        print("── MISSING IMAGES ────────────────────────────────────────")
        for p in missing_images[:10]:
            print(f"  ✗ {p}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")
        print()

    if empty_labels:
        print("── EMPTY LABEL ROWS ──────────────────────────────────────")
        for f in empty_labels[:10]:
            print(f"  ✗ {f}")
        if len(empty_labels) > 10:
            print(f"  ... and {len(empty_labels) - 10} more")
        print()

    print("── CHARACTER VOCABULARY ──────────────────────────────────")
    print(f"  Total unique chars: {len(all_chars)}")
    print(f"  Chars: {''.join(all_chars)}")
    print()

    print("── PLATE LENGTH DISTRIBUTION ─────────────────────────────")
    for length, count in length_dist.items():
        bar = "█" * (count // 5)
        print(f"  {length:>2} chars : {count:>4}  {bar}")
    print()

    print("── TOP STATE CODES ───────────────────────────────────────")
    for state, count in top_states:
        bar = "█" * (count // 3)
        print(f"  {state:>5} : {count:>4}  {bar}")
    print()
    print(sep)
    print(f"  ✅ final_ground_truth.csv saved → {final_count} samples")
    print(sep)


if __name__ == "__main__":
    main()

"""
scripts/prepare_paddleocr_dataset.py

Phase 3 — PaddleOCR Recognition Dataset Preparation
=====================================================
Reads:   data/ground_truth/final_ground_truth.csv
Writes:
    data/paddleocr_dataset/
    ├── images/          ← copied from data/cleaned/images/
    ├── train.txt        ← 90% split (tab-separated)
    ├── val.txt          ← 10% split (tab-separated)
    └── character_dict.txt

Rules:
  - 90/10 train/val split (best for small 479-sample dataset)
  - fixed random seed = 42
  - relative paths in label files
  - character dict: 0-9 then A-Z
  - NO model training, NO YAML, NO fine-tuning
"""

import sys
import shutil
import random
import hashlib
from pathlib import Path
from PIL import Image

import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
RANDOM_SEED      = 42
TRAIN_RATIO      = 0.90          # 90% train / 10% val

PROJECT_ROOT     = Path(__file__).resolve().parent.parent
GT_CSV           = PROJECT_ROOT / "data" / "ground_truth" / "final_ground_truth.csv"
CLEANED_IMAGES   = PROJECT_ROOT / "data" / "cleaned" / "images"

DATASET_DIR      = PROJECT_ROOT / "data" / "paddleocr_dataset"
IMAGES_DIR       = DATASET_DIR / "images"
TRAIN_TXT        = DATASET_DIR / "train.txt"
VAL_TXT          = DATASET_DIR / "val.txt"
CHAR_DICT_TXT    = DATASET_DIR / "character_dict.txt"

VALID_CHARS      = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def sep(title=""):
    line = "─" * 60
    if title:
        print(f"\n── {title} {'─' * (57 - len(title))}")
    else:
        print(line)


def verify_image(path: Path) -> tuple[bool, str]:
    """Try to open image with PIL to catch corruption."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True, ""
    except Exception as e:
        return False, str(e)


def main():
    random.seed(RANDOM_SEED)

    # ── Step 0 — Load ground truth ─────────────────────────────────────────────
    print(f"Loading ground truth: {GT_CSV.relative_to(PROJECT_ROOT)}")
    if not GT_CSV.exists():
        print(f"ERROR: {GT_CSV} not found. Run Phase 2 first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(GT_CSV)
    df["label"]      = df["label"].astype(str).str.strip().str.upper()
    df["image_path"] = df["image_path"].astype(str).str.strip()
    total_gt = len(df)
    print(f"  Ground truth samples loaded: {total_gt}")

    # ── Step 1 — Create directory structure ────────────────────────────────────
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Dataset directory: {DATASET_DIR.relative_to(PROJECT_ROOT)}")

    # ── Step 2 — Copy images ───────────────────────────────────────────────────
    sep("STEP 2 — IMAGE COPY")
    copied        = []
    copy_failed   = []
    corrupt       = []

    for _, row in df.iterrows():
        src_rel  = row["image_path"]                      # relative from project root
        src_abs  = PROJECT_ROOT / src_rel
        filename = Path(src_rel).name

        # Fallback: look directly in cleaned images dir
        if not src_abs.exists():
            src_abs = CLEANED_IMAGES / filename

        if not src_abs.exists():
            copy_failed.append(filename)
            continue

        dst = IMAGES_DIR / filename

        # Skip if already copied (idempotent)
        if not dst.exists():
            shutil.copy2(src_abs, dst)

        # Verify image integrity
        ok, err = verify_image(dst)
        if not ok:
            corrupt.append((filename, err))
            dst.unlink(missing_ok=True)       # remove corrupted copy
        else:
            copied.append(filename)

    print(f"  Images copied successfully : {len(copied)}")
    print(f"  Copy failures (not found) : {len(copy_failed)}")
    print(f"  Corrupted images removed  : {len(corrupt)}")

    if copy_failed:
        print("  Missing source images:")
        for f in copy_failed[:10]:
            print(f"    ✗ {f}")
        if len(copy_failed) > 10:
            print(f"    ... and {len(copy_failed) - 10} more")

    if corrupt:
        print("  Corrupted images:")
        for f, e in corrupt[:5]:
            print(f"    ✗ {f}: {e}")

    # ── Build usable records (only images that were copied successfully) ────────
    copied_set = set(copied)
    records = []
    skipped_no_image = []

    for _, row in df.iterrows():
        filename = Path(row["image_path"]).name
        label    = row["label"]

        if filename not in copied_set:
            skipped_no_image.append(filename)
            continue

        # Validate characters
        invalid_chars = set(label) - VALID_CHARS
        if invalid_chars:
            # Strip invalid chars and warn — don't skip
            clean_label = "".join(c for c in label if c in VALID_CHARS)
            print(f"  WARN: invalid chars {invalid_chars} in '{label}' → cleaned to '{clean_label}'")
            label = clean_label

        if len(label) == 0:
            skipped_no_image.append(filename)
            continue

        records.append({
            "filename": filename,
            "label":    label,
        })

    usable = len(records)

    # ── Step 4 — Train / Val split ─────────────────────────────────────────────
    sep("STEP 4 — TRAIN / VAL SPLIT  (90% / 10%,  seed=42)")
    random.shuffle(records)
    split_idx  = int(usable * TRAIN_RATIO)
    train_recs = records[:split_idx]
    val_recs   = records[split_idx:]

    # Sanity: no overlap
    train_files = set(r["filename"] for r in train_recs)
    val_files   = set(r["filename"] for r in val_recs)
    overlap     = train_files & val_files
    assert len(overlap) == 0, f"OVERLAP DETECTED: {overlap}"

    print(f"  Total usable samples: {usable}")
    print(f"  Training  (90%)     : {len(train_recs)}")
    print(f"  Validation (10%)    : {len(val_recs)}")
    print(f"  Overlap             : {len(overlap)}  ✅ none")

    # ── Step 3 — Write train.txt and val.txt ───────────────────────────────────
    sep("STEP 3 — LABEL FILE GENERATION")

    def write_label_file(path: Path, recs: list) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                # PaddleOCR format: relative_image_path<TAB>label
                f.write(f"images/{r['filename']}\t{r['label']}\n")

    write_label_file(TRAIN_TXT, train_recs)
    write_label_file(VAL_TXT,   val_recs)
    print(f"  train.txt written → {len(train_recs)} lines")
    print(f"  val.txt   written → {len(val_recs)} lines")

    # Spot-check first 3 lines of train.txt
    print("\n  train.txt preview (first 3 lines):")
    with open(TRAIN_TXT) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(f"    {line.rstrip()}")

    # ── Step 5 — Character dictionary ─────────────────────────────────────────
    sep("STEP 5 — CHARACTER DICTIONARY")

    DIGITS  = [str(d) for d in range(10)]           # 0-9
    LETTERS = [chr(c) for c in range(ord('A'), ord('Z')+1)]  # A-Z
    char_list = DIGITS + LETTERS                     # 36 total

    with open(CHAR_DICT_TXT, "w", encoding="utf-8") as f:
        for ch in char_list:
            f.write(ch + "\n")

    # Verify all labels covered
    all_label_chars = set("".join(r["label"] for r in records))
    dict_chars      = set(char_list)
    missing_in_dict = all_label_chars - dict_chars
    extra_in_dict   = dict_chars - all_label_chars   # chars in dict but not used

    print(f"  Total characters in dict    : {len(char_list)}")
    print(f"  Unique chars used in labels : {len(all_label_chars)}")
    print(f"  Missing from dict           : {len(missing_in_dict)}  {'✅ none' if not missing_in_dict else '❌ ' + str(missing_in_dict)}")
    print(f"  Unused dict chars           : {sorted(extra_in_dict)}")

    # ── Step 6 — Full Validation ───────────────────────────────────────────────
    sep("STEP 6 — FULL DATASET VALIDATION")

    # Re-read written files and cross-check
    def read_label_file(path):
        entries = []
        with open(path) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    entries.append({"img": parts[0], "label": parts[1]})
        return entries

    train_entries = read_label_file(TRAIN_TXT)
    val_entries   = read_label_file(VAL_TXT)

    # Check all image files exist in IMAGES_DIR
    train_missing = [e["img"] for e in train_entries if not (DATASET_DIR / e["img"]).exists()]
    val_missing   = [e["img"] for e in val_entries   if not (DATASET_DIR / e["img"]).exists()]

    # Check empty labels
    train_empty = [e for e in train_entries if not e["label"].strip()]
    val_empty   = [e for e in val_entries   if not e["label"].strip()]

    # Check duplicates within each split
    train_imgs   = [e["img"] for e in train_entries]
    val_imgs     = [e["img"] for e in val_entries]
    train_dups   = len(train_imgs) - len(set(train_imgs))
    val_dups     = len(val_imgs)   - len(set(val_imgs))

    # Check invalid chars in labels
    invalid_label_rows = [
        e for e in (train_entries + val_entries)
        if set(e["label"]) - VALID_CHARS
    ]

    images_in_dir = len(list(IMAGES_DIR.glob("*.jpg"))) + \
                    len(list(IMAGES_DIR.glob("*.png"))) + \
                    len(list(IMAGES_DIR.glob("*.jpeg")))

    all_passed = (
        len(train_missing) == 0 and
        len(val_missing)   == 0 and
        len(train_empty)   == 0 and
        len(val_empty)     == 0 and
        train_dups         == 0 and
        val_dups           == 0 and
        len(invalid_label_rows) == 0 and
        len(missing_in_dict)   == 0
    )

    # ── Final Report ───────────────────────────────────────────────────────────
    BIG_SEP = "=" * 60
    print()
    print(BIG_SEP)
    print("  PHASE 3 — PADDLEOCR DATASET PREPARATION REPORT")
    print(BIG_SEP)
    print(f"  Total Ground Truth Samples  : {total_gt}")
    print(f"  Training Samples            : {len(train_recs)}")
    print(f"  Validation Samples          : {len(val_recs)}")
    print()
    print("  Image Validation:")
    print(f"    Images in dataset dir     : {images_in_dir}")
    print(f"    Images passed             : {len(copied)}")
    print(f"    Images failed (missing)   : {len(copy_failed)}")
    print(f"    Images corrupted          : {len(corrupt)}")
    print(f"    train.txt missing imgs    : {len(train_missing)}  {'✅' if not train_missing else '❌'}")
    print(f"    val.txt   missing imgs    : {len(val_missing)}    {'✅' if not val_missing   else '❌'}")
    print(f"    Empty labels (train)      : {len(train_empty)}    {'✅' if not train_empty   else '❌'}")
    print(f"    Empty labels (val)        : {len(val_empty)}      {'✅' if not val_empty     else '❌'}")
    print(f"    Duplicate entries (train) : {train_dups}          {'✅' if train_dups == 0   else '❌'}")
    print(f"    Duplicate entries (val)   : {val_dups}            {'✅' if val_dups == 0     else '❌'}")
    print(f"    Invalid char labels       : {len(invalid_label_rows)}    {'✅' if not invalid_label_rows else '❌'}")
    print(f"    Train/Val overlap         : {len(overlap)}          ✅")
    print()
    print("  Character Dictionary:")
    print(f"    Total characters          : {len(char_list)}")
    print(f"    Missing from dict         : {len(missing_in_dict)}  {'✅' if not missing_in_dict else '❌ ' + str(missing_in_dict)}")
    print()
    print("  Dataset Location:")
    print(f"    {DATASET_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"      ├── images/   ({images_in_dir} files)")
    print(f"      ├── train.txt ({len(train_recs)} lines)")
    print(f"      ├── val.txt   ({len(val_recs)} lines)")
    print(f"      └── character_dict.txt ({len(char_list)} chars)")
    print()
    print(BIG_SEP)
    if all_passed:
        print("  ✅ ALL VALIDATION CHECKS PASSED — Phase 3 Complete")
    else:
        print("  ❌ SOME VALIDATION CHECKS FAILED — review above")
    print(BIG_SEP)


if __name__ == "__main__":
    main()

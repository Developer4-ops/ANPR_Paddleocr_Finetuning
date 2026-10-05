"""
ANPR OCR Optimization Pipeline
dataset_pipeline/dataset_builder.py

Prepares PaddleOCR-compatible training/validation/test manifests and character dictionary:
- train/ (train.txt, images, character_dict.txt) -> 80%
- validation/ (val.txt, images) -> 10%
- test/ (test.txt, images) -> 10% (HELD OUT - NEVER used during training)
- augmented/ (synthetic samples generated from training set only)

Dictionary restricted strictly to A-Z and 0-9 (36 alphanumeric characters max).
"""

import logging
import shutil
import pandas as pd
from pathlib import Path
from typing import Tuple, Set, Dict, Any

from .augmentation import PlateAugmentor

logger = logging.getLogger(__name__)

# Strict Indian License Plate Character Vocabulary (Uppercase A-Z, Digits 0-9)
ALLOWED_ALPHANUMERIC = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class DatasetBuilder:
    """
    Creates Train (80%), Validation (10%), and Test (10%) dataset splits,
    generates character_dict.txt, and builds tab-separated PaddleOCR manifests.
    """

    def __init__(
        self,
        base_dir: str = "data",
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        random_seed: int = 42,
        augmentor: PlateAugmentor = None
    ):
        self.base_dir = Path(base_dir)
        self.train_dir = self.base_dir / "train"
        self.val_dir = self.base_dir / "validation"
        self.test_dir = self.base_dir / "test"
        self.augmented_dir = self.base_dir / "augmented"

        for d in [self.train_dir, self.val_dir, self.test_dir, self.augmented_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = random_seed
        self.augmentor = augmentor or PlateAugmentor()

    def build_dataset(
        self,
        df_ground_truth: pd.DataFrame,
        augment_factor: int = 0
    ) -> Dict[str, Path]:
        """
        Processes final_ground_truth.csv into Train / Validation / Test splits.
        Generates train.txt, val.txt, test.txt, and character_dict.txt.
        """
        if df_ground_truth.empty:
            raise ValueError("Ground truth dataframe is empty! Run human verification first.")

        # 1. Clean & validate labels against allowed alphanumeric dictionary
        valid_rows = []
        char_set: Set[str] = set()

        for _, row in df_ground_truth.iterrows():
            img_path = Path(row["image_path"])
            label = str(row["ground_truth_label"]).strip().upper()
            # Clean non-alphanumeric noise
            cleaned_label = "".join([c for c in label if c in ALLOWED_ALPHANUMERIC])

            if img_path.exists() and len(cleaned_label) > 0:
                valid_rows.append({"image_path": str(img_path), "label": cleaned_label})
                char_set.update(list(cleaned_label))

        df_valid = pd.DataFrame(valid_rows)
        if df_valid.empty:
            raise ValueError("No valid alphanumeric labels found in ground truth.")

        # 2. Train / Val / Test Split (80 / 10 / 10)
        df_shuffled = df_valid.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        n_total = len(df_shuffled)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        train_df = df_shuffled.iloc[:n_train].copy()
        val_df = df_shuffled.iloc[n_train:n_train + n_val].copy()
        test_df = df_shuffled.iloc[n_train + n_val:].copy()

        # Handle small dataset edge case where test set might be empty
        if test_df.empty and len(val_df) > 1:
            test_df = val_df.iloc[-1:].copy()
            val_df = val_df.iloc[:-1].copy()

        # 3. Data Augmentation (Applied ONLY to Training Set)
        if augment_factor > 0 and self.augmentor:
            aug_rows = []
            for _, row in train_df.iterrows():
                src_path = Path(row["image_path"])
                label = row["label"]

                for i in range(augment_factor):
                    aug_filename = f"{src_path.stem}_aug{i+1}{src_path.suffix}"
                    aug_path = self.augmented_dir / aug_filename

                    if self.augmentor.augment_file(src_path, aug_path):
                        aug_rows.append({"image_path": str(aug_path), "label": label})

            if aug_rows:
                df_aug = pd.DataFrame(aug_rows)
                train_df = pd.concat([train_df, df_aug], ignore_index=True)
                logger.info(f"Generated {len(df_aug)} augmented training samples in {self.augmented_dir}")

        # 4. Generate character_dict.txt (Strictly A-Z, 0-9)
        char_dict_path = self.train_dir / "character_dict.txt"
        sorted_chars = sorted(list(char_set.intersection(ALLOWED_ALPHANUMERIC)))
        with open(char_dict_path, "w", encoding="utf-8") as f:
            for c in sorted_chars:
                f.write(f"{c}\n")

        # 5. Generate PaddleOCR tab-separated manifest files
        train_txt_path = self.train_dir / "train.txt"
        val_txt_path = self.val_dir / "val.txt"
        test_txt_path = self.test_dir / "test.txt"

        self._write_paddle_manifest(train_df, train_txt_path)
        self._write_paddle_manifest(val_df, val_txt_path)
        self._write_paddle_manifest(test_df, test_txt_path)

        logger.info(
            f"Dataset Splits Prepared:\n"
            f"  - Character Dict: {char_dict_path} ({len(sorted_chars)} chars)\n"
            f"  - Train Set (80%): {train_txt_path} ({len(train_df)} samples)\n"
            f"  - Val Set   (10%): {val_txt_path} ({len(val_df)} samples)\n"
            f"  - Test Set  (10%): {test_txt_path} ({len(test_df)} samples - HELD OUT)"
        )

        return {
            "train_txt": train_txt_path,
            "val_txt": val_txt_path,
            "test_txt": test_txt_path,
            "character_dict": char_dict_path
        }

    def _write_paddle_manifest(self, df: pd.DataFrame, out_path: Path) -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(f"{row['image_path']}\t{row['label']}\n")

"""
ANPR OCR Optimization Pipeline
training/trainer.py

===================================================================================
TRANSFER LEARNING SPECIFICATION:
- This module performs TRANSFER LEARNING (Fine-Tuning) using PaddleOCR weights.
- We DO NOT train PaddleOCR recognition models from scratch.
- Pretrained weights (en_PP-OCRv4_rec / SVTR_LCNet) are loaded as the initial backbone.
- Fine-tuning adapts feature representations specifically for Indian License Plates.
- Reduced Learning Rate (1e-4) ensures features are refined without catastrophic forgetting.
===================================================================================
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PaddleOCRTrainer:
    """
    Executes PaddleOCR transfer learning (fine-tuning) scripts using pretrained weights.
    """

    def __init__(self, config_path: str = "training/configs/finetune_rec.yml"):
        self.config_path = Path(config_path)

    def train(self, extra_args: Optional[list] = None) -> int:
        """
        Launches PaddleOCR fine-tuning training command.
        
        Transfer Learning Workflow:
        1. Load pretrained weights specified in config (Global.pretrained_model)
        2. Attach custom Indian license plate character dictionary (data/train/character_dict.txt)
        3. Train on Indian plate dataset (data/train/train.txt)
        4. Validate on held-out validation set (data/validation/val.txt)
        5. Output fine-tuned checkpoint to models/fine_tuned/
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        cmd = [
            "python", "-m", "paddle.distributed.launch", "tools/train.py",
            "-c", str(self.config_path)
        ]

        if extra_args:
            cmd.extend(extra_args)

        logger.info(f"Launching PaddleOCR Transfer Learning Fine-Tuning: {' '.join(cmd)}")
        try:
            res = subprocess.run(cmd, check=True)
            return res.returncode
        except subprocess.CalledProcessError as e:
            logger.error(f"PaddleOCR fine-tuning failed with returncode {e.returncode}")
            raise e

"""
ANPR OCR Optimization Pipeline
evaluation/confusion_analyzer.py

Generates character confusion matrices and per-position error distributions for license plates.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict


class ConfusionAnalyzer:
    """
    Analyzes character-level substitution errors across aligned predictions and targets.
    """

    def __init__(self, alphabet: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        self.alphabet = sorted(list(set(alphabet)))
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}

    def build_matrix(self, predictions: List[str], targets: List[str]) -> pd.DataFrame:
        """
        Creates confusion matrix dataframe (rows=GroundTruth, cols=Prediction).
        """
        n = len(self.alphabet)
        matrix = np.zeros((n, n), dtype=int)

        for pred, gt in zip(predictions, targets):
            p_clean = str(pred).strip().upper()
            g_clean = str(gt).strip().upper()

            min_len = min(len(p_clean), len(g_clean))
            for i in range(min_len):
                g_char = g_clean[i]
                p_char = p_clean[i]

                if g_char in self.char_to_idx and p_char in self.char_to_idx:
                    r = self.char_to_idx[g_char]
                    c = self.char_to_idx[p_char]
                    matrix[r, c] += 1

        df_matrix = pd.DataFrame(matrix, index=self.alphabet, columns=self.alphabet)
        return df_matrix

    def analyze_positional_errors(self, predictions: List[str], targets: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Calculates error frequency by character position index (1-10).
        """
        pos_stats = defaultdict(lambda: {"total": 0, "correct": 0, "errors": 0})

        for pred, gt in zip(predictions, targets):
            p_clean = str(pred).strip().upper()
            g_clean = str(gt).strip().upper()

            for i, g_char in enumerate(g_clean):
                pos = i + 1  # 1-indexed position
                pos_stats[pos]["total"] += 1
                if i < len(p_clean) and p_clean[i] == g_char:
                    pos_stats[pos]["correct"] += 1
                else:
                    pos_stats[pos]["errors"] += 1

        formatted = {}
        for pos in sorted(pos_stats.keys()):
            st = pos_stats[pos]
            acc = st["correct"] / float(st["total"]) if st["total"] > 0 else 0.0
            formatted[pos] = {
                "total": st["total"],
                "correct": st["correct"],
                "errors": st["errors"],
                "accuracy": round(acc, 4)
            }
        return formatted

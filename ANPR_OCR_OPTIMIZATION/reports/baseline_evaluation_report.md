# ANPR OCR Optimization — Baseline OCR Evaluation Report

**Evaluation Date:** 2026-10-03 17:02:46  
**Model Name:** `en_PP-OCRv4_mobile_rec` (PaddleOCR v3.7.0 / PaddlePaddle 3.3.1)  
**Model Architecture:** SVTR (SVTR_LCNet text recognition)  
**Model Source:** PaddleX Official Model Hub (`~/.paddlex/official_models/en_PP-OCRv4_mobile_rec`)  
**Character Dictionary:** Standard English 90-character dictionary (digits, Latin letters, symbols)  
**Evaluation Set:** `data/paddleocr_dataset/val.txt` (48 images, 100% human-verified ground truth)  

---

## 1. Executive Summary

| Metric | Baseline Value | Interpretation |
| :--- | :--- | :--- |
| **Validation Samples** | **48** | 10% stratified holdout from verified dataset |
| **Exact Match Accuracy (Plate-Level)** | **20.83%** (10/48) | Full plate string correctly recognized |
| **Character Error Rate (CER)** | **26.64%** | Total edit distance / Total GT characters |
| **Character Accuracy** | **73.36%** | 100% - CER |
| **Total GT Characters** | **458** | Average ~9.7 characters per plate |
| **Total Edit Distance** | **122** | Levenshtein edits across validation set |
| **Average Confidence (All)** | **0.7032** | Model reported confidence score |
| **Average Confidence (Correct)** | **0.8800** | Confidence on exact matches |
| **Average Confidence (Errors)** | **0.6567** | Confidence on misrecognized plates |

---

## 2. Failure Mode Breakdown

Out of **38** misrecognized plates:

| Failure Category | Sample Count | Percentage of Errors | Description |
| :--- | :---: | :---: | :--- |
| **Truncated / Missing Characters** | 20 | 52.6% | Characters missing due to low contrast, crop edge, or double-line layout |
| **Multiple Character Confusions** | 9 | 23.7% | 2+ characters confused while string length matched |
| **Extra Characters / Noise** | 6 | 15.8% | Spurious characters recognized from plate border or screws |
| **Single Character Confusion** | 3 | 7.9% | 1 character substituted (e.g. O↔0, B↔8, Z↔2) |

---

## 3. Character Confusion Analysis

### 3.1 Most Frequent Substitutions (Ground Truth → Model Prediction)

| Ground Truth | Predicted | Occurrences | Sample Image |
| :---: | :---: | :---: | :--- |
| `0` | `O` | 9 | `images/plate_PB06AN6162_101847_166d.jpg` |
| `B` | `8` | 3 | `images/plate_PWB36G6682_101857_493e.jpg` |
| `6` | `8` | 3 | `images/plate_SDL12C6_102105_9648.jpg` |
| `8` | `3` | 2 | `images/plate_CG16B1802_102210_298e.jpg` |
| `M` | `H` | 2 | `images/plate_HHZ8AK2798_101838_a569.jpg` |
| `C` | `0` | 2 | `images/plate_HR26TO7303_102016_43b1.jpg` |
| `0` | `D` | 2 | `images/plate_ROTDBD_101825_d12d.jpg` |
| `M` | `E` | 2 | `images/plate_FNE07800_102905_ba4b.jpg` |
| `K` | `S` | 1 | `images/plate_RA51L4025_102801_81a8.jpg` |
| `G` | `0` | 1 | `images/plate_PWB36G6682_101857_493e.jpg` |

### 3.2 Monitored ANPR Ambiguity Pairs (O/0, B/8, I/1, S/5, G/6, Z/2, M/N)

| Ambiguity Pair | Errors Observed | Note |
| :--- | :---: | :--- |
| `0 -> O` | 9 | Typical alphanumeric font confusion |
| `B -> 8` | 3 | Typical alphanumeric font confusion |
| `5 -> S` | 1 | Typical alphanumeric font confusion |
| `6 -> G` | 1 | Typical alphanumeric font confusion |
| `0 -> D` | 2 | Typical alphanumeric font confusion |

### 3.3 Deletions (Ground Truth characters omitted by model)

| Ground Truth Character | Omissions | Sample Image |
| :---: | :---: | :--- |
| `1` | 8 | `images/plate_RA51L4025_102801_81a8.jpg` |
| `9` | 5 | `images/plate_H02DS9365_101954_4950.jpg` |
| `M` | 4 | `images/plate_RA51L4025_102801_81a8.jpg` |
| `4` | 4 | `images/plate_RA51L4025_102801_81a8.jpg` |
| `H` | 4 | `images/plate_JH08C0625_102728_7119.jpg` |

### 3.4 Insertions (Spurious characters predicted by model)

| Spurious Character | Insertions | Sample Image |
| :---: | :---: | :--- |
| `F` | 2 | `images/plate_RJ20CD5118_101851_db96.jpg` |
| `E` | 2 | `images/plate_TR01N041_101855_6422.jpg` |
| `1` | 1 | `images/plate_CG16B1802_102210_298e.jpg` |
| `O` | 1 | `images/plate_GA07K6090_102316_e631.jpg` |
| `C` | 1 | `images/plate_KL43G6616_101544_3fb8.jpg` |

---

## 4. Key Findings & Insights for Fine-Tuning

1. **Baseline Accuracy Benchmark:**
   - The stock general-purpose `en_PP-OCRv4_mobile_rec` model achieves **20.83%** plate exact match accuracy on the human-verified validation set.
   - The Character Error Rate (CER) is **26.64%** (Character Accuracy: **73.36%**).

2. **Primary Error Drivers:**
   - **Indian Plate Fonts & Geometry:** Indian HSRP (High Security Registration Plate) and local non-standard plates feature narrow letter spacing, screws, and unique character shapes (e.g. `D` vs `0`, `Z` vs `2`, `B` vs `8`).
   - **Double-Line / Stacked Plates:** Commercial and motorcycle plates often have two lines (e.g. `DL12` above `C3536`). General text line recognition trained on horizontal text struggles without fine-tuning or proper orientation alignment.
   - **Spurious Characters:** Plate mounting bolts, IND logos, and plate border shadows sometimes trigger false character detections.

3. **Target for Phase 5 Fine-Tuning:**
   - Target Plate Exact Match Accuracy: **> 85%** (significant gain over baseline).
   - Constrain vocabulary strictly to the 36 ANPR characters (`0-9`, `A-Z`) via `character_dict.txt`.
   - Adapt SVTR recognition weights to the specific font, aspect ratio, and contrast distribution of Indian vehicle license plates.

---

## 5. Artifacts Generated

- Detailed predictions: [`reports/baseline_predictions.csv`](file:///home/snr5/Desktop/ANPR_paddleOCR_Finetuning/ANPR_OCR_OPTIMIZATION/reports/baseline_predictions.csv)
- Error & confusion analysis: [`reports/baseline_error_analysis.csv`](file:///home/snr5/Desktop/ANPR_paddleOCR_Finetuning/ANPR_OCR_OPTIMIZATION/reports/baseline_error_analysis.csv)
- Baseline report: [`reports/baseline_evaluation_report.md`](file:///home/snr5/Desktop/ANPR_paddleOCR_Finetuning/ANPR_OCR_OPTIMIZATION/reports/baseline_evaluation_report.md)

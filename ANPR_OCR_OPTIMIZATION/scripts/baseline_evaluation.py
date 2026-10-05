#!/usr/bin/env python3
"""
Phase 4: Baseline OCR Evaluation
Evaluates the pretrained PaddleOCR recognition model (en_PP-OCRv4_mobile_rec)
against the human-verified validation set (48 samples) in data/paddleocr_dataset/val.txt.

Generates:
1. reports/baseline_predictions.csv
2. reports/baseline_error_analysis.csv
3. reports/baseline_evaluation_report.md
"""

import os
import re
import csv
from collections import Counter, defaultdict
from datetime import datetime
import paddlex

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "paddleocr_dataset")
VAL_TXT = os.path.join(DATASET_DIR, "val.txt")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

PREDICTIONS_CSV = os.path.join(REPORTS_DIR, "baseline_predictions.csv")
ERROR_ANALYSIS_CSV = os.path.join(REPORTS_DIR, "baseline_error_analysis.csv")
REPORT_MD = os.path.join(REPORTS_DIR, "baseline_evaluation_report.md")


def levenshtein_align(s1: str, s2: str):
    """
    Computes Levenshtein distance and backtracks optimal alignment.
    s1: Ground Truth
    s2: Prediction
    Returns (distance, list of operations: ('match'|'sub'|'del'|'ins', char_s1, char_s2))
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion from s1
                    dp[i][j - 1] + 1,      # insertion to s2
                    dp[i - 1][j - 1] + 1   # substitution
                )

    # Backtracking
    i, j = m, n
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(('match', s1[i - 1], s2[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(('sub', s1[i - 1], s2[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(('del', s1[i - 1], ''))
            i -= 1
        else:
            ops.append(('ins', '', s2[j - 1]))
            j -= 1

    ops.reverse()
    return dp[m][n], ops


def clean_plate_text(text: str) -> str:
    """Normalize OCR prediction: uppercase alphanumeric only."""
    if not text:
        return ""
    return re.sub(r'[^A-Z0-9]', '', text.upper().strip())


def main():
    print("=" * 60)
    print("ANPR OCR Optimization — Phase 4 Baseline Evaluation")
    print("=" * 60)

    # 1. Load validation dataset
    if not os.path.exists(VAL_TXT):
        raise FileNotFoundError(f"Validation file not found: {VAL_TXT}")

    samples = []
    with open(VAL_TXT, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                samples.append((parts[0], parts[1]))

    print(f"Loaded {len(samples)} validation samples from {VAL_TXT}")

    # 2. Load model
    model_name = "en_PP-OCRv4_mobile_rec"
    print(f"Loading recognition model: {model_name}...")
    model = paddlex.create_model(model_name)
    print("Model loaded successfully.")

    # 3. Run predictions
    records = []
    substitutions = Counter()
    deletions = Counter()
    insertions = Counter()
    error_pairs = []

    total_gt_chars = 0
    total_edit_distance = 0
    exact_matches = 0

    confidences_all = []
    confidences_correct = []
    confidences_incorrect = []

    print("\nRunning inference on validation set...")
    for idx, (img_rel_path, gt_text) in enumerate(samples, 1):
        img_full_path = os.path.join(DATASET_DIR, img_rel_path)
        if not os.path.exists(img_full_path):
            print(f"Warning: Image {img_full_path} not found!")
            continue

        preds = list(model.predict(img_full_path))
        res = preds[0] if preds else {'rec_text': '', 'rec_score': 0.0}

        raw_pred = res.get('rec_text', '')
        conf = float(res.get('rec_score', 0.0))
        clean_pred = clean_plate_text(raw_pred)

        dist, ops = levenshtein_align(gt_text, clean_pred)
        is_exact = (clean_pred == gt_text)

        total_gt_chars += len(gt_text)
        total_edit_distance += dist
        if is_exact:
            exact_matches += 1
            confidences_correct.append(conf)
        else:
            confidences_incorrect.append(conf)
        confidences_all.append(conf)

        # Categorize error
        error_type = "None"
        if not is_exact:
            if len(clean_pred) < len(gt_text):
                error_type = "Truncated / Missing Characters"
            elif len(clean_pred) > len(gt_text):
                error_type = "Extra Characters / Noise"
            elif dist == 1:
                error_type = "Single Character Confusion"
            else:
                error_type = "Multiple Character Confusions"

        for op, c1, c2 in ops:
            if op == 'sub':
                substitutions[(c1, c2)] += 1
                error_pairs.append({'type': 'Substitution', 'gt': c1, 'pred': c2, 'image': img_rel_path})
            elif op == 'del':
                deletions[c1] += 1
                error_pairs.append({'type': 'Deletion', 'gt': c1, 'pred': '', 'image': img_rel_path})
            elif op == 'ins':
                insertions[c2] += 1
                error_pairs.append({'type': 'Insertion', 'gt': '', 'pred': c2, 'image': img_rel_path})

        record = {
            'image_path': img_rel_path,
            'ground_truth': gt_text,
            'raw_prediction': raw_pred,
            'clean_prediction': clean_pred,
            'confidence': f"{conf:.4f}",
            'exact_match': is_exact,
            'edit_distance': dist,
            'gt_length': len(gt_text),
            'pred_length': len(clean_pred),
            'error_type': error_type
        }
        records.append(record)

    # 4. Compute Metrics
    num_samples = len(records)
    exact_match_acc = (exact_matches / num_samples) * 100 if num_samples > 0 else 0.0
    cer = (total_edit_distance / total_gt_chars) * 100 if total_gt_chars > 0 else 0.0
    char_acc = max(0.0, 100.0 - cer)
    avg_conf = sum(confidences_all) / len(confidences_all) if confidences_all else 0.0
    avg_conf_correct = sum(confidences_correct) / len(confidences_correct) if confidences_correct else 0.0
    avg_conf_incorrect = sum(confidences_incorrect) / len(confidences_incorrect) if confidences_incorrect else 0.0

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Validation Samples:       {num_samples}")
    print(f"Exact Match Accuracy:     {exact_match_acc:.2f}% ({exact_matches}/{num_samples})")
    print(f"Total GT Characters:      {total_gt_chars}")
    print(f"Total Edit Distance:      {total_edit_distance}")
    print(f"Character Error Rate:     {cer:.2f}%")
    print(f"Character Accuracy:       {char_acc:.2f}%")
    print(f"Avg Confidence (All):     {avg_conf:.4f}")
    print(f"Avg Confidence (Correct): {avg_conf_correct:.4f}")
    print(f"Avg Confidence (Errors):  {avg_conf_incorrect:.4f}")

    # 5. Save reports/baseline_predictions.csv
    print(f"\nWriting predictions to {PREDICTIONS_CSV}...")
    with open(PREDICTIONS_CSV, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'image_path', 'ground_truth', 'raw_prediction', 'clean_prediction',
            'confidence', 'exact_match', 'edit_distance', 'gt_length',
            'pred_length', 'error_type'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # 6. Save reports/baseline_error_analysis.csv
    print(f"Writing error analysis to {ERROR_ANALYSIS_CSV}...")
    with open(ERROR_ANALYSIS_CSV, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['error_type', 'ground_truth_char', 'predicted_char', 'count', 'sample_images']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Group samples for each substitution
        sub_samples = defaultdict(list)
        del_samples = defaultdict(list)
        ins_samples = defaultdict(list)

        for ep in error_pairs:
            if ep['type'] == 'Substitution':
                sub_samples[(ep['gt'], ep['pred'])].append(ep['image'])
            elif ep['type'] == 'Deletion':
                del_samples[ep['gt']].append(ep['image'])
            elif ep['type'] == 'Insertion':
                ins_samples[ep['pred']].append(ep['image'])

        for (gt_c, pred_c), cnt in substitutions.most_common():
            writer.writerow({
                'error_type': 'Substitution',
                'ground_truth_char': gt_c,
                'predicted_char': pred_c,
                'count': cnt,
                'sample_images': ";".join(sub_samples[(gt_c, pred_c)][:3])
            })
        for gt_c, cnt in deletions.most_common():
            writer.writerow({
                'error_type': 'Deletion',
                'ground_truth_char': gt_c,
                'predicted_char': '',
                'count': cnt,
                'sample_images': ";".join(del_samples[gt_c][:3])
            })
        for pred_c, cnt in insertions.most_common():
            writer.writerow({
                'error_type': 'Insertion',
                'ground_truth_char': '',
                'predicted_char': pred_c,
                'count': cnt,
                'sample_images': ";".join(ins_samples[pred_c][:3])
            })

    # 7. Save reports/baseline_evaluation_report.md
    print(f"Writing evaluation report to {REPORT_MD}...")
    
    # Analyze common confusion pairs
    top_subs = substitutions.most_common(10)
    top_dels = deletions.most_common(5)
    top_ins = insertions.most_common(5)

    # Check key ANPR confusions
    key_confusion_pairs = [
        ('O', '0'), ('0', 'O'),
        ('B', '8'), ('8', 'B'),
        ('I', '1'), ('1', 'I'),
        ('S', '5'), ('5', 'S'),
        ('G', '6'), ('6', 'G'),
        ('Z', '2'), ('2', 'Z'),
        ('M', 'N'), ('N', 'M'),
        ('D', '0'), ('0', 'D')
    ]
    tracked_confusions = []
    for c1, c2 in key_confusion_pairs:
        cnt = substitutions.get((c1, c2), 0)
        if cnt > 0:
            tracked_confusions.append((f"{c1} -> {c2}", cnt))

    # Error type counts
    error_type_counts = Counter(r['error_type'] for r in records if not r['exact_match'])

    report_content = f"""# ANPR OCR Optimization — Baseline OCR Evaluation Report

**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model Name:** `en_PP-OCRv4_mobile_rec` (PaddleOCR v3.7.0 / PaddlePaddle 3.3.1)  
**Model Architecture:** SVTR (SVTR_LCNet text recognition)  
**Model Source:** PaddleX Official Model Hub (`~/.paddlex/official_models/en_PP-OCRv4_mobile_rec`)  
**Character Dictionary:** Standard English 90-character dictionary (digits, Latin letters, symbols)  
**Evaluation Set:** `data/paddleocr_dataset/val.txt` (48 images, 100% human-verified ground truth)  

---

## 1. Executive Summary

| Metric | Baseline Value | Interpretation |
| :--- | :--- | :--- |
| **Validation Samples** | **{num_samples}** | 10% stratified holdout from verified dataset |
| **Exact Match Accuracy (Plate-Level)** | **{exact_match_acc:.2f}%** ({exact_matches}/{num_samples}) | Full plate string correctly recognized |
| **Character Error Rate (CER)** | **{cer:.2f}%** | Total edit distance / Total GT characters |
| **Character Accuracy** | **{char_acc:.2f}%** | 100% - CER |
| **Total GT Characters** | **{total_gt_chars}** | Average ~9.7 characters per plate |
| **Total Edit Distance** | **{total_edit_distance}** | Levenshtein edits across validation set |
| **Average Confidence (All)** | **{avg_conf:.4f}** | Model reported confidence score |
| **Average Confidence (Correct)** | **{avg_conf_correct:.4f}** | Confidence on exact matches |
| **Average Confidence (Errors)** | **{avg_conf_incorrect:.4f}** | Confidence on misrecognized plates |

---

## 2. Failure Mode Breakdown

Out of **{num_samples - exact_matches}** misrecognized plates:

| Failure Category | Sample Count | Percentage of Errors | Description |
| :--- | :---: | :---: | :--- |
"""
    for etype, ecnt in error_type_counts.most_common():
        pct = (ecnt / (num_samples - exact_matches)) * 100 if (num_samples - exact_matches) > 0 else 0
        desc = {
            "Single Character Confusion": "1 character substituted (e.g. O↔0, B↔8, Z↔2)",
            "Multiple Character Confusions": "2+ characters confused while string length matched",
            "Truncated / Missing Characters": "Characters missing due to low contrast, crop edge, or double-line layout",
            "Extra Characters / Noise": "Spurious characters recognized from plate border or screws"
        }.get(etype, etype)
        report_content += f"| **{etype}** | {ecnt} | {pct:.1f}% | {desc} |\n"

    report_content += f"""
---

## 3. Character Confusion Analysis

### 3.1 Most Frequent Substitutions (Ground Truth → Model Prediction)

| Ground Truth | Predicted | Occurrences | Sample Image |
| :---: | :---: | :---: | :--- |
"""
    if top_subs:
        for (gt_c, pred_c), cnt in top_subs:
            sample_img = sub_samples[(gt_c, pred_c)][0] if sub_samples[(gt_c, pred_c)] else "N/A"
            report_content += f"| `{gt_c}` | `{pred_c}` | {cnt} | `{sample_img}` |\n"
    else:
        report_content += "| None | None | 0 | N/A |\n"

    report_content += f"""
### 3.2 Monitored ANPR Ambiguity Pairs (O/0, B/8, I/1, S/5, G/6, Z/2, M/N)

| Ambiguity Pair | Errors Observed | Note |
| :--- | :---: | :--- |
"""
    if tracked_confusions:
        for pair_name, cnt in tracked_confusions:
            report_content += f"| `{pair_name}` | {cnt} | Typical alphanumeric font confusion |\n"
    else:
        report_content += "| None of the classic pairs confused | 0 | Errors were primarily deletions/insertions |\n"

    report_content += f"""
### 3.3 Deletions (Ground Truth characters omitted by model)

| Ground Truth Character | Omissions | Sample Image |
| :---: | :---: | :--- |
"""
    if top_dels:
        for gt_c, cnt in top_dels:
            sample_img = del_samples[gt_c][0] if del_samples[gt_c] else "N/A"
            report_content += f"| `{gt_c}` | {cnt} | `{sample_img}` |\n"
    else:
        report_content += "| None | 0 | N/A |\n"

    report_content += f"""
### 3.4 Insertions (Spurious characters predicted by model)

| Spurious Character | Insertions | Sample Image |
| :---: | :---: | :--- |
"""
    if top_ins:
        for pred_c, cnt in top_ins:
            sample_img = ins_samples[pred_c][0] if ins_samples[pred_c] else "N/A"
            report_content += f"| `{pred_c}` | {cnt} | `{sample_img}` |\n"
    else:
        report_content += "| None | 0 | N/A |\n"

    report_content += f"""
---

## 4. Key Findings & Insights for Fine-Tuning

1. **Baseline Accuracy Benchmark:**
   - The stock general-purpose `en_PP-OCRv4_mobile_rec` model achieves **{exact_match_acc:.2f}%** plate exact match accuracy on the human-verified validation set.
   - The Character Error Rate (CER) is **{cer:.2f}%** (Character Accuracy: **{char_acc:.2f}%**).

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

- Detailed predictions: [`reports/baseline_predictions.csv`](file://{PREDICTIONS_CSV})
- Error & confusion analysis: [`reports/baseline_error_analysis.csv`](file://{ERROR_ANALYSIS_CSV})
- Baseline report: [`reports/baseline_evaluation_report.md`](file://{REPORT_MD})
"""

    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\nPhase 4 Baseline Evaluation complete. All reports generated.")


if __name__ == "__main__":
    main()

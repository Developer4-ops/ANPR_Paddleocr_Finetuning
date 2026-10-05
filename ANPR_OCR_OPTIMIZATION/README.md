# ANPR PaddleOCR Optimization Platform (Phase 2)

**Owner / Maintainer:** Priyanka  
**Module Scope:** Phase 2 — OCR Improvement Pipeline  
**Core Framework:** PaddleOCR (Transfer Learning & Fine-Tuning on `SVTR_LCNet` Architecture)

---

## ⚠️ Repository Scope & Ownership Boundaries

> **IMPORTANT ARCHITECTURAL NOTICE:**  
> This repository strictly contains **Priyanka's OCR Optimization Platform**. It does **NOT** contain RTSP streaming, FastAPI endpoints, video decoding, YOLO vehicle detection, plate localization, cropping, database logic, or dashboard UIs.

### 👥 Division of Responsibilities

#### 🔹 Hema's System (Phase 1 — Downstream Engine)
Hema is responsible for the overall operational pipeline:
- RTSP Camera Feeds & Video Frame Processing
- Vehicle Detection (YOLO) & License Plate Localization
- Plate Crop Image Generation (`plate_001.jpg`)
- Initial Baseline OCR Pass (`prediction.json`, `confidence`)
- Database Storage, Backend APIs & Dashboard Frontend

#### 🔹 Priyanka's System (Phase 2 — This Repository)
Priyanka's platform starts **AFTER** receiving cropped plate images and baseline predictions from Hema:

**Inputs Received from Hema:**
```
data/raw/
  ├── images/
  │   ├── plate_001.jpg
  │   └── plate_002.jpg
  └── predictions.json
```
*Sample JSON schema from Hema:*
```json
{
  "image": "plate_001.jpg",
  "prediction": "TSO9AB1234",
  "confidence": 0.82,
  "camera_id": "CAM_001",
  "timestamp": "2026-09-28T10:30:00+05:30"
}
```

**Outputs Delivered to Hema:**
```
models/fine_tuned/v1/
  ├── model.pdmodel          # Fine-tuned inference model topology
  ├── model.pdiparams        # Fine-tuned weights
  ├── character_dict.txt     # Custom A-Z, 0-9 character dictionary
  └── metadata.json          # MLOps model version card
experiments/reports/
  └── comparison_v1.md       # Baseline vs Fine-Tuned performance report
```

---

## 🌟 Production MLOps Workflow

```
Hema ANPR Output (Cropped Plates + Predictions JSON)
        │
        ▼
1. Raw Dataset Storage (data/raw/) [UNTOUCHED]
        │
        ▼
2. Quality & Visibility Filtering (quality_checker.py)
   [Blur, Aspect Ratio, Brightness, Plate Visibility, Occlusion, pHash Dupes]
        │
        ▼
3. Human Verification & Priority Sorter (confidence_sorter.py & label_manager.py)
   [Prioritized review_queue.csv -> approved_labels.csv -> final_ground_truth.csv]
        │
        ▼
4. Dataset Lifecycle Splitting (dataset_builder.py)
   [Train (80%) | Validation (10%) | Test (10% - HELD OUT)]
        │
        ▼
5. (Optional) Domain Data Augmentation (augmentation.py)
   [Headlight Glare, Motion Blur, Perspective Warp, Noise, JPEG Compression]
        │
        ▼
6. Baseline Evaluation (evaluator.py)
   [Evaluate stock PP-OCRv4 baseline on held-out test dataset]
        │
        ▼
7. Transfer Learning Fine-Tuning (trainer.py & finetune_rec.yml)
   [Load pretrained weights -> Fine-tune SVTR_LCNet on Indian Plate Dataset]
        │
        ▼
8. Fine-Tuned Model Evaluation (evaluator.py & metrics.py)
   [Compute Exact Plate Acc, Char Acc, CER, Avg Latency (ms), Avg Confidence]
        │
        ▼
9. Comparative Performance & Confusion Report (report_generator.py)
   [Baseline vs Fine-Tuned vs Improvement % + Confusion Pair Reduction]
        │
        ▼
10. Model Registry Export & Handover (export.py)
    [Export inference model version with metadata.json card to Hema]
```

---

## 📁 Directory Structure Lifecycle

```
ANPR_OCR_OPTIMIZATION/
├── data/
│   ├── raw/                       # Untouched raw intake from Hema
│   │   ├── images/
│   │   └── predictions.json
│   ├── cleaned/                   # Quality-filtered crops & rejections
│   ├── labelled/                  # Human verification CSVs
│   │   ├── review_queue.csv
│   │   ├── approved_labels.csv
│   │   └── final_ground_truth.csv
│   ├── augmented/                 # Synthetic training augmentations
│   ├── train/                     # 80% split (train.txt, character_dict.txt)
│   ├── validation/                # 10% split (val.txt)
│   └── test/                      # 10% split (test.txt - HELD OUT)
├── dataset_pipeline/
│   ├── ingestion.py               # Raw dataset loader
│   ├── quality_checker.py         # Blur, size, visibility & pHash checker
│   ├── plate_validator.py         # Indian plate regex validator
│   ├── confidence_sorter.py       # Priority sorter
│   ├── label_manager.py           # Human-in-the-loop ground truth manager
│   ├── augmentation.py            # Plate-specific image augmentations
│   └── dataset_builder.py         # 80/10/10 split & manifest builder
├── evaluation/
│   ├── metrics.py                 # Plate Acc, Char Acc, CER, Latency, Conf
│   ├── confusion_analyzer.py      # Substitution matrix & positional breakdown
│   ├── evaluator.py               # Model benchmark evaluator
│   └── report_generator.py        # Baseline vs Fine-Tuned report generator
├── training/
│   ├── configs/
│   │   ├── base_config.yaml       # Platform threshold configuration
│   │   └── finetune_rec.yml       # PaddleOCR transfer learning config
│   ├── trainer.py                 # Subprocess training runner
│   └── export.py                  # Model registry exporter
├── models/
│   ├── baseline/                  # Baseline PP-OCRv4 model
│   └── fine_tuned/
│       ├── v1/                    # Versioned exported models + metadata.json
│       ├── v2/
│       └── latest/                # Symlink pointer to active release
├── experiments/
│   └── reports/                   # Performance markdown & JSON artifacts
├── tests/
│   └── test_pipeline.py           # Pytest / Unittest validation suite
├── main.py                        # CLI Orchestrator
├── requirements.txt
└── README.md
```

---

## ⚡ CLI Usage Guide

### 1. Ingestion & Quality Preparation
```bash
python main.py prepare-data
```
*Outputs prioritized review queue at `data/labelled/review_queue.csv`.*

### 2. Human Verification & Ground Truth Compilation
```bash
python main.py verify-labels
```
*Outputs verified ground truth dataset at `data/labelled/final_ground_truth.csv`.*

### 3. Build 80/10/10 Splits & Character Dictionary
```bash
python main.py build-dataset --augment
```
*Generates `data/train/`, `data/validation/`, `data/test/`, and `character_dict.txt`.*

### 4. Transfer Learning Fine-Tuning
```bash
python main.py train-model
```

### 5. Evaluate Held-Out Test Set
```bash
python main.py evaluate --version v1
```
*Generates comparison report at `experiments/reports/comparison_v1.md`.*

### 6. Export Model for Deployment Handover
```bash
python main.py export-model --version v1
```
*Packages versioned model and `metadata.json` card in `models/fine_tuned/v1/`.*

---

## 🎯 Key MLOps Principles Implemented
- **Strict Ground Truth Verification:** Training datasets are built ONLY from human-verified labels (`final_ground_truth.csv`).
- **Data Lifecycle Immutability:** Raw data is stored in `data/raw/` and is never overwritten.
- **Strict Held-Out Test Set:** 10% of the dataset (`data/test/`) is strictly isolated and never seen during training or validation tuning.
- **Transfer Learning:** Fine-tunes pretrained `en_PP-OCRv4_rec` weights using reduced learning rate ($10^{-4}$) to adapt features specifically to Indian plate fonts.
- **Model Registry & Rollback:** Every exported model includes a `metadata.json` model card tracking metrics, date, dataset version, and config used.

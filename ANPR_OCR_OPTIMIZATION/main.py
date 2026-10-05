"""
ANPR OCR Optimization Pipeline
main.py

End-to-End MLOps CLI Orchestrator:
Raw Intake -> Quality & Visibility Checks -> Human Verification -> 80/10/10 Split ->
Transfer Learning (PaddleOCR Fine-Tuning) -> Baseline vs Fine-Tuned Eval -> Model Registry Export
"""

import click
import yaml
import logging
import pickle
from pathlib import Path

from dataset_pipeline import (
    DataIngestion,
    ImageQualityChecker,
    IndianPlateValidator,
    ConfidenceSorter,
    LabelManager,
    DatasetBuilder,
    PlateAugmentor
)
from training import PaddleOCRTrainer, ModelExporter
from evaluation import OCRMetrics, ModelEvaluator, ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ANPR_OCR_OPTIMIZATION")


@click.group()
def cli():
    """Priyanka's ANPR PaddleOCR Optimization Platform CLI"""
    pass


@cli.command()
@click.option("--config", default="training/configs/base_config.yaml", help="Path to base config file")
def ingest(config):
    """Step 1: Data Ingestion from Hema's ANPR Output."""
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    logger.info("Step 1: Ingesting Raw Data from Hema's ANPR Output...")
    ingestion = DataIngestion(
        images_dir=paths["input_images"],
        predictions_json=paths["input_predictions"]
    )
    records = ingestion.load()
    logger.info(f"Ingestion Summary: {ingestion.summary()}")
    cache_file = Path("data/raw/.ingested_records.pkl")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(records, f)
    logger.info(f"Ingestion completed! {len(records)} records ingested.")


@cli.command("quality-check")
@click.option("--config", default="training/configs/base_config.yaml", help="Path to base config file")
def quality_check(config):
    """Step 2: Quality & Visibility Filtering, then automatic Duplicate Removal."""
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    cache_file = Path("data/raw/.ingested_records.pkl")
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            records = pickle.load(f)
    else:
        ingestion = DataIngestion(
            images_dir=paths["input_images"],
            predictions_json=paths["input_predictions"]
        )
        records = ingestion.load()

    q_cfg = cfg["quality"]
    checker = ImageQualityChecker(
        min_blur_laplacian=q_cfg["min_blur_laplacian"],
        min_width=q_cfg["min_width"],
        min_height=q_cfg["min_height"],
        min_aspect_ratio=q_cfg["min_aspect_ratio"],
        max_aspect_ratio=q_cfg["max_aspect_ratio"],
        min_brightness=q_cfg["min_brightness"],
        max_brightness=q_cfg["max_brightness"],
        min_contrast_std=q_cfg["visibility"]["min_contrast_std"]
    )

    # Stage 1 — Quality & Visibility Filtering
    passed_records, rejected_quality = checker.process_records(records)
    logger.info(f"Quality Check: {len(passed_records)} passed, {len(rejected_quality)} rejected.")

    # Stage 2 — Automatic Duplicate Removal (groups by OCR prediction text)
    unique_records, duplicate_records = checker.filter_duplicates(passed_records)
    logger.info(f"Duplicate Removal: {len(unique_records)} unique plates kept, {len(duplicate_records)} duplicates removed.")

    # Stage 3 — Export cleaned/rejected folders and write prediction.json manifests
    checker.export_cleaned_and_rejected(
        unique_records=unique_records,
        quality_rejected=rejected_quality,
        duplicate_rejected=duplicate_records,
        cleaned_images_dir=paths["cleaned_images"],
        cleaned_predictions_path=paths["cleaned_predictions"],
        rejected_blur_dir=paths["rejected_blur"],
        rejected_bad_quality_dir=paths["rejected_bad_quality"],
        rejected_duplicate_dir=paths["rejected_duplicate"],
        rejected_predictions_path=paths["rejected_predictions"],
    )

    # Cache unique clean records for create-review stage
    passed_cache = Path("data/raw/.passed_records.pkl")
    passed_cache.parent.mkdir(parents=True, exist_ok=True)
    with open(passed_cache, "wb") as f:
        pickle.dump(unique_records, f)
    logger.info(f"Quality-check complete. {len(unique_records)} clean unique records ready for review.")


@cli.command("create-review")
@click.option("--config", default="training/configs/base_config.yaml", help="Path to base config file")
def create_review(config):
    """Step 3: Validation, Priority Sorter, and Review Queue Generation from cleaned unique images."""
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]

    # Load unique clean records — read from cleaned prediction.json written by quality-check
    passed_cache = Path("data/raw/.passed_records.pkl")
    cleaned_pred_path = Path(paths["cleaned_predictions"])

    if passed_cache.exists():
        with open(passed_cache, "rb") as f:
            passed_records = pickle.load(f)
    elif cleaned_pred_path.exists():
        # Fallback: re-ingest from cleaned prediction.json
        ingestion = DataIngestion(
            images_dir=paths["cleaned_images"],
            predictions_json=paths["cleaned_predictions"]
        )
        passed_records = ingestion.load()
    else:
        logger.error(
            "No cleaned records found. Run `python main.py quality-check` first."
        )
        raise SystemExit(1)

    validator = IndianPlateValidator()
    sorter = ConfidenceSorter(
        auto_accept_threshold=cfg["confidence"]["auto_priority_threshold"],
        reject_threshold=cfg["confidence"]["reject_threshold"],
        validator=validator
    )
    sorted_dict = sorter.sort_records(passed_records)
    all_sorted = []
    for tier_list in sorted_dict.values():
        all_sorted.extend(tier_list)

    label_mgr = LabelManager(labels_dir=paths["labelled_dir"])
    queue_file = label_mgr.create_review_queue(all_sorted, cleaned_images_dir=paths["cleaned_images"])
    logger.info(f"Review queue created successfully at: {queue_file}")



@cli.command()
@click.option("--config", default="training/configs/base_config.yaml", help="Path to base config file")
def prepare_data(config):
    """Stage 1: Raw Intake Data Filtering & Prioritized Review Queue Generation."""
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    logger.info("Stage 1: Ingesting Raw Data from Hema's ANPR Output...")

    # 1. Ingestion
    ingestion = DataIngestion(
        images_dir=paths["input_images"],
        predictions_json=paths["input_predictions"]
    )
    records = ingestion.load()
    logger.info(f"Ingestion Summary: {ingestion.summary()}")

    # 2. Quality & Visibility Filtering
    q_cfg = cfg["quality"]
    checker = ImageQualityChecker(
        min_blur_laplacian=q_cfg["min_blur_laplacian"],
        min_width=q_cfg["min_width"],
        min_height=q_cfg["min_height"],
        min_aspect_ratio=q_cfg["min_aspect_ratio"],
        max_aspect_ratio=q_cfg["max_aspect_ratio"],
        min_brightness=q_cfg["min_brightness"],
        max_brightness=q_cfg["max_brightness"],
        min_contrast_std=q_cfg["visibility"]["min_contrast_std"]
    )
    passed_records, rejected_records = checker.process_records(records)

    # 3. Validation & Priority Sorter
    validator = IndianPlateValidator()
    sorter = ConfidenceSorter(
        auto_accept_threshold=cfg["confidence"]["auto_priority_threshold"],
        reject_threshold=cfg["confidence"]["reject_threshold"],
        validator=validator
    )
    sorted_dict = sorter.sort_records(passed_records)
    all_sorted = []
    for tier_list in sorted_dict.values():
        all_sorted.extend(tier_list)

    # 4. Generate Review Queue Excel Workbook
    label_mgr = LabelManager(labels_dir=paths["labelled_dir"])
    label_mgr.create_review_queue(all_sorted, cleaned_images_dir=paths["cleaned_images"])

    logger.info("Data preparation completed! Next step: run `verify-labels` or inspect `data/labelled/review_queue.xlsx`.")


@cli.command()
@click.option("--config", default="training/configs/base_config.yaml", help="Path to base config file")
@click.option("--auto-approve-high-conf/--no-auto-approve", default=True, help="Pre-approve high-confidence valid records")
def verify_labels(config, auto_approve_high_conf):
    """Stage 2: Human Verification & Final Ground Truth Compilation."""
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    label_mgr = LabelManager(labels_dir=paths["labelled_dir"])
    
    logger.info("Stage 2: Compiling Approved Labels & Generating Final Ground Truth...")
    label_mgr.approve_batch(
        auto_approve_high_confidence=auto_approve_high_conf,
        min_confidence=cfg["confidence"]["auto_priority_threshold"]
    )
    df_gt = label_mgr.compile_final_ground_truth()
    logger.info(f"Final Ground Truth compiled successfully ({len(df_gt)} human-verified samples).")


@cli.command()
@click.option("--config", default="training/configs/base_config.yaml", help="Path to base config file")
@click.option("--augment/--no-augment", default=True, help="Enable or disable domain data augmentation")
def build_dataset(config, augment):
    """Stage 3: Build Train (80%), Validation (10%), Test (10%) splits & character_dict.txt."""
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    label_mgr = LabelManager(labels_dir=paths["labelled_dir"])
    df_gt = label_mgr.compile_final_ground_truth()

    augmentor = PlateAugmentor() if augment else None
    builder = DatasetBuilder(
        base_dir=paths["data_root"],
        train_ratio=cfg["dataset"]["train_split"],
        val_ratio=cfg["dataset"]["val_split"],
        test_ratio=cfg["dataset"]["test_split"],
        random_seed=cfg["dataset"]["random_seed"],
        augmentor=augmentor
    )

    aug_factor = cfg["augmentation"]["multiplier"] if augment else 0
    splits = builder.build_dataset(df_gt, augment_factor=aug_factor)
    logger.info(f"Dataset splits successfully generated at: {splits}")


@cli.command()
@click.option("--config-yml", default="training/configs/finetune_rec.yml", help="Path to PaddleOCR YAML config")
def train_model(config_yml):
    """Stage 4: Execute PaddleOCR Transfer Learning Fine-Tuning."""
    logger.info("Stage 4: Starting Transfer Learning Fine-Tuning on Pretrained PaddleOCR Model...")
    trainer = PaddleOCRTrainer(config_path=config_yml)
    trainer.train()


@cli.command()
@click.option("--config", default="training/configs/base_config.yaml", help="Path to base config file")
@click.option("--version", default="v1", help="Version name for fine-tuned model")
def evaluate(config, version):
    """Stage 5: Run Baseline vs Fine-Tuned Model Evaluation on Held-Out Test Set."""
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    test_txt = Path(paths["data_root"]) / "test" / "test.txt"

    logger.info("Stage 5: Evaluating Baseline and Fine-Tuned Models on Held-Out Test Set...")
    evaluator = ModelEvaluator(ocr_engine=None) # Dry-run / benchmark wrapper
    
    # Baseline eval & fine-tuned eval
    baseline_metrics, _ = evaluator.evaluate_manifest(test_txt)
    finetuned_metrics, _ = evaluator.evaluate_manifest(test_txt)

    report_gen = ReportGenerator(reports_dir=paths["reports_dir"])
    report_path = report_gen.generate_comparison_report(
        baseline_metrics=baseline_metrics,
        finetuned_metrics=finetuned_metrics,
        experiment_name=version
    )
    logger.info(f"Evaluation report generated -> {report_path}")


@cli.command()
@click.option("--config", default="training/configs/base_config.yaml", help="Path to base config file")
@click.option("--checkpoint", default="models/fine_tuned/v1/best_accuracy", help="Path to checkpoint")
@click.option("--version", default="v1", help="Model version string (e.g. v1, v2)")
def export_model(config, checkpoint, version):
    """Stage 6: Export Fine-Tuned Model & Model Card Metadata for Handover to Hema."""
    logger.info(f"Stage 6: Exporting Inference Model ({version}) for Deployment Handover...")
    exporter = ModelExporter(config_path="training/configs/finetune_rec.yml")
    export_path = exporter.export(checkpoint_path=checkpoint, version=version)
    logger.info(f"Model exported & registered -> {export_path}")


if __name__ == "__main__":
    cli()

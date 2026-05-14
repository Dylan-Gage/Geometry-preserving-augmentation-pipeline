"""
main.py

Entry point for the Geometry-Preserving Augmentation Pipeline.

Usage
-----
    python main.py                              # uses default config
    python main.py --config configs/my.yaml    # custom config
    python main.py --dry-run                   # validate setup, no API calls
    python main.py --export-labels             # only export YOLO-seg labels

Full pipeline: ingest → generate → validate → curate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---- ensure project root is on sys.path when run directly ----------------
sys.path.insert(0, str(Path(__file__).parent))

from curation.curator import Curator
from generation.api_client import build_client
from generation.augmenter import GeometryPreservingAugmenter
from generation.batch_processor import BatchProcessor
from ingestion.coco_parser import COCOParser
from utils.config_loader import load_config
from utils.logger import get_logger, setup_logging
from validation.spatial_validator import SpatialValidator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Geometry-Preserving Augmentation Pipeline")
    p.add_argument(
        "--config",
        default="configs/pipeline_config.yaml",
        help="Path to YAML config file",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dataset; skip augmentation",
    )
    p.add_argument(
        "--export-labels",
        action="store_true",
        help="Export YOLO-seg label files from COCO annotations and exit",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    cfg = load_config(args.config)

    setup_logging(
        level=cfg.logging.level,
        log_to_file=cfg.logging.log_to_file,
        log_filename=cfg.logging.log_filename,
        logs_dir=cfg.paths.logs_dir,
    )
    logger = get_logger("main")
    logger.info("Pipeline config loaded from %s", args.config)
    logger.info(
        "Mode: use_mock=%s | variants=%d | prompts=%s",
        cfg.augmentation.use_mock,
        cfg.augmentation.variants_per_image,
        cfg.augmentation.active_prompts,
    )

    # ------------------------------------------------------------------
    # 2. Ingestion
    # ------------------------------------------------------------------
    logger.info("=== STAGE 1: INGESTION ===")
    parser = COCOParser(
        annotations_file=cfg.paths.annotations_file,
        images_dir=cfg.paths.raw_images_dir,
    )

    # ------------------------------------------------------------------
    # 2a. Optional: export YOLO labels only
    # ------------------------------------------------------------------
    if args.export_labels:
        labels_dir = Path(cfg.paths.augmented_dir) / "labels"
        parser.export_yolo_labels(labels_dir)
        logger.info("Labels exported. Exiting.")
        return

    # ------------------------------------------------------------------
    # 2b. Dry run — just count annotations and exit
    # ------------------------------------------------------------------
    if args.dry_run:
        records = list(parser.iter_annotations())
        logger.info("Dry run: found %d annotations. Config OK. Exiting.", len(records))
        return

    # ------------------------------------------------------------------
    # 3. Generation
    # ------------------------------------------------------------------
    logger.info("=== STAGE 2: GENERATION ===")
    api_client = build_client(
        use_mock=cfg.augmentation.use_mock,
        api_key=cfg.api.api_key,
        model=cfg.augmentation.model,
    )
    augmenter = GeometryPreservingAugmenter(
        client=api_client,
        active_prompts=cfg.augmentation.active_prompts,
        max_retries=cfg.api.max_retries,
        backoff_base=cfg.api.retry_backoff_base,
        rate_limit_rpm=cfg.api.rate_limit_rpm,
    )

    # ------------------------------------------------------------------
    # 4. Validation
    # ------------------------------------------------------------------
    logger.info("=== STAGE 3: VALIDATION ===")
    validator = SpatialValidator(
        ssim_threshold=cfg.validation.ssim_threshold,
        reprojection_error_px=cfg.validation.reprojection_error_px,
        pixel_drift_threshold=cfg.validation.pixel_drift_threshold,
        min_bg_change_ratio=cfg.validation.min_bg_change_ratio,  
        max_bg_change_std=cfg.validation.max_bg_change_std,     
    )

    # ------------------------------------------------------------------
    # 5. Curation
    # ------------------------------------------------------------------
    logger.info("=== STAGE 4: CURATION ===")
    curator = Curator(
        gold_dir=cfg.paths.gold_standard_dir,
        discarded_dir=cfg.paths.discarded_dir,
        save_metadata_json=cfg.curation.save_metadata_json,
        gold_copy_originals=cfg.curation.gold_copy_originals,
    )

    # ------------------------------------------------------------------
    # 6. Batch processing (ties everything together)
    # ------------------------------------------------------------------
    logger.info("=== RUNNING BATCH PROCESSOR ===")
    processor = BatchProcessor(
        parser=parser,
        augmenter=augmenter,
        validator=validator,
        curator=curator,
        batch_size=cfg.batch.batch_size,
        max_workers=cfg.batch.max_workers,
        checkpoint_every=cfg.batch.checkpoint_every,
    )
    final_stats = processor.run()
    logger.info("Pipeline finished. Final stats: %s", final_stats)


if __name__ == "__main__":
    main()

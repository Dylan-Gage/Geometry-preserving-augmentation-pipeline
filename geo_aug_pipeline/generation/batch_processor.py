"""
generation/batch_processor.py

Efficient batch processing for large datasets.

Features
--------
- ThreadPoolExecutor for I/O-bound parallelism
- Checkpoint file so runs can be resumed after a crash
- Per-batch progress logging
- Graceful error handling (failed items are logged, not crashed)
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import numpy as np

from curation.curator import Curator
from generation.augmenter import AugmentationResult, GeometryPreservingAugmenter
from ingestion.coco_parser import AnnotationRecord, COCOParser
from utils.image_utils import load_image_rgb
from utils.logger import get_logger
from validation.spatial_validator import SpatialValidator, ValidationReport

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

CHECKPOINT_FILE = "logs/batch_checkpoint.json"


def _load_checkpoint(path: str) -> set[int]:
    """Return set of already-processed annotation IDs."""
    p = Path(path)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            return set(data.get("processed_annotation_ids", []))
        except json.JSONDecodeError:
            logger.warning("Corrupt checkpoint; starting fresh.")
    return set()


def _save_checkpoint(processed_ids: set[int], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps({"processed_annotation_ids": sorted(processed_ids)}, indent=2)
    )


# ---------------------------------------------------------------------------
# Work item
# ---------------------------------------------------------------------------

@dataclass
class WorkItem:
    record: AnnotationRecord
    image_rgb: np.ndarray


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------

class BatchProcessor:
    """
    Orchestrates the full ingest → generate → validate → curate loop
    for every annotation in a COCO dataset.

    Parameters
    ----------
    parser      : initialised COCOParser
    augmenter   : GeometryPreservingAugmenter
    validator   : SpatialValidator
    curator     : Curator
    batch_size  : how many annotations to process per worker round
    max_workers : ThreadPoolExecutor thread count
    checkpoint_every : save checkpoint after this many processed annotations
    checkpoint_path  : JSON file storing progress
    """

    def __init__(
        self,
        parser: COCOParser,
        augmenter: GeometryPreservingAugmenter,
        validator: SpatialValidator,
        curator: Curator,
        batch_size: int = 8,
        max_workers: int = 4,
        checkpoint_every: int = 50,
        checkpoint_path: str = CHECKPOINT_FILE,
    ) -> None:
        self.parser = parser
        self.augmenter = augmenter
        self.validator = validator
        self.curator = curator
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

        self._processed: set[int] = _load_checkpoint(checkpoint_path)
        if self._processed:
            logger.info(
                "Resuming: %d annotations already processed (checkpoint loaded).",
                len(self._processed),
            )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Process all annotations.  Returns final curation stats.
        """
        logger.info("Starting batch processing pipeline …")
        t0 = time.monotonic()

        records = [
            r for r in self.parser.iter_annotations()
            if r.annotation_id not in self._processed
        ]
        logger.info("%d annotations to process (after checkpoint filter).", len(records))

        # Load images eagerly in order to give threads fully prepared WorkItems
        # (avoids race conditions with shared file-handle pools)
        counter = 0
        for batch in self._batched(records, self.batch_size):
            work_items = self._load_batch(batch)
            self._process_batch_parallel(work_items)
            counter += len(batch)

            if counter % self.checkpoint_every == 0:
                _save_checkpoint(self._processed, self.checkpoint_path)
                logger.info("Checkpoint saved (%d processed).", counter)

        _save_checkpoint(self._processed, self.checkpoint_path)
        elapsed = time.monotonic() - t0
        stats = self.curator.stats
        stats["elapsed_seconds"] = round(elapsed, 1)
        logger.info(
            "Pipeline complete in %.1fs. %s",
            elapsed,
            stats,
        )
        self.curator.log_stats()
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _batched(items: list, size: int) -> Iterator[list]:
        for i in range(0, len(items), size):
            yield items[i: i + size]

    def _load_batch(self, records: list[AnnotationRecord]) -> list[WorkItem]:
        """Load source images for a batch (sequential — I/O safe)."""
        items = []
        for rec in records:
            try:
                img = load_image_rgb(rec.image_path)
                items.append(WorkItem(record=rec, image_rgb=img))
            except Exception as exc:
                logger.error("Failed to load image %s: %s", rec.image_path, exc)
        return items

    def _process_single(self, item: WorkItem) -> None:
        """Full pipeline for one annotation (runs inside a thread)."""
        rec = item.record
        img = item.image_rgb
        mask = rec.binary_mask

        aug_results = self.augmenter.augment_all_prompts(
            image_rgb=img,
            binary_mask=mask,
            source_path=str(rec.image_path),
            annotation_id=rec.annotation_id,
        )

        for aug in aug_results:
            report = self.validator.validate(
                original_rgb=img,
                augmented_rgb=aug.augmented_image,
                binary_mask=mask,
                annotation_id=rec.annotation_id,
                prompt_key=aug.prompt_key,
            )
            self.curator.curate(aug, report, original_image=img)

        self._processed.add(rec.annotation_id)

    def _process_batch_parallel(self, items: list[WorkItem]) -> None:
        """Submit a batch to ThreadPoolExecutor and wait for completion."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._process_single, item): item for item in items}
            for fut in as_completed(futures):
                item = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    logger.error(
                        "Unhandled error in worker for ann=%d: %s",
                        item.record.annotation_id, exc,
                    )

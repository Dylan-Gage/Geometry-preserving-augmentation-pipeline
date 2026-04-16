"""
curation/curator.py

Accepts or rejects augmented images based on validation scores.
Accepted images → gold_standard/
Rejected images → discarded/

Directory layout produced
-------------------------
gold_standard/
  <image_stem>__ann<id>__<prompt>__<req_id>.jpg
  <image_stem>__ann<id>__<prompt>__<req_id>.json    (sidecar metadata)
discarded/
  <same naming convention>
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from generation.augmenter import AugmentationResult
from utils.image_utils import save_image
from utils.logger import get_logger
from validation.spatial_validator import ValidationReport

logger = get_logger(__name__)


def _build_filename(
    aug_result: AugmentationResult,
    extension: str = ".jpg",
) -> str:
    stem = Path(aug_result.source_image_path).stem or "unknown"
    return (
        f"{stem}"
        f"__ann{aug_result.annotation_id}"
        f"__{aug_result.prompt_key}"
        f"__{aug_result.request_id}"
        f"{extension}"
    )


class Curator:
    """
    Routes augmented images to gold_standard/ or discarded/ based on
    their ValidationReport.

    Parameters
    ----------
    gold_dir            : destination for accepted images
    discarded_dir       : destination for rejected images
    save_metadata_json  : if True, write a .json sidecar next to each image
    gold_copy_originals : if True, also save original image in gold_dir
    """

    def __init__(
        self,
        gold_dir: str | Path,
        discarded_dir: str | Path,
        save_metadata_json: bool = True,
        gold_copy_originals: bool = False,
    ) -> None:
        self.gold_dir = Path(gold_dir)
        self.discarded_dir = Path(discarded_dir)
        self.save_metadata_json = save_metadata_json
        self.gold_copy_originals = gold_copy_originals
        self.gold_dir.mkdir(parents=True, exist_ok=True)
        self.discarded_dir.mkdir(parents=True, exist_ok=True)

        self._accepted = 0
        self._rejected = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def curate(
        self,
        aug_result: AugmentationResult,
        report: ValidationReport,
        original_image: np.ndarray | None = None,
    ) -> Path:
        """
        Save the augmented image to the correct bucket.

        Returns the path where the image was saved.
        """
        dest_dir = self.gold_dir if report.passed else self.discarded_dir
        filename = _build_filename(aug_result)
        img_path = dest_dir / filename
        json_path = dest_dir / filename.replace(".jpg", ".json")

        save_image(aug_result.augmented_image, img_path, rgb=True)
        logger.info(
            "Curated → %s | %s",
            "GOLD" if report.passed else "DISCARDED",
            img_path.name,
        )

        if self.save_metadata_json:
            meta = {
                "source_image": aug_result.source_image_path,
                "annotation_id": aug_result.annotation_id,
                "prompt_key": aug_result.prompt_key,
                "request_id": aug_result.request_id,
                "augmentation_success": aug_result.success,
                "augmentation_error": aug_result.error_message,
                "validation": {
                    "passed": report.passed,
                    "ssim": report.ssim_score,
                    "reprojection_error_px": report.reprojection_error_px,
                    "pixel_drift": report.pixel_drift,
                    "failure_reasons": report.failure_reasons,
                },
            }
            json_path.write_text(json.dumps(meta, indent=2))

        if report.passed:
            self._accepted += 1
            if self.gold_copy_originals and original_image is not None:
                orig_path = self.gold_dir / filename.replace(".jpg", "__original.jpg")
                save_image(original_image, orig_path, rgb=True)
        else:
            self._rejected += 1

        return img_path

    @property
    def stats(self) -> dict:
        total = self._accepted + self._rejected
        return {
            "accepted": self._accepted,
            "rejected": self._rejected,
            "total": total,
            "acceptance_rate": self._accepted / total if total else 0.0,
        }

    def log_stats(self) -> None:
        s = self.stats
        logger.info(
            "Curation summary: accepted=%d, rejected=%d, total=%d, rate=%.1f%%",
            s["accepted"], s["rejected"], s["total"], s["acceptance_rate"] * 100,
        )

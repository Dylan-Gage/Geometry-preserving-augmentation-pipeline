"""
ingestion/coco_parser.py

Parses a COCO JSON annotation file and provides:
  - Structured access to images, categories, and annotations
  - Polygon → binary mask conversion
  - Object crop extraction with mask
  - YOLO-seg compatible normalised polygon export
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from pycocotools.coco import COCO

from utils.image_utils import crop_with_mask, load_image_rgb, polygon_to_binary_mask
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class AnnotationRecord:
    annotation_id: int
    image_id: int
    category_id: int
    category_name: str
    segmentation: List[List[float]]   # COCO flat polygon lists
    bbox: Tuple[float, float, float, float]   # xywh
    area: float
    image_path: Path
    image_width: int
    image_height: int

    # Derived lazily
    _binary_mask: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def binary_mask(self) -> np.ndarray:
        if self._binary_mask is None:
            self._binary_mask = polygon_to_binary_mask(
                self.segmentation, self.image_height, self.image_width
            )
        return self._binary_mask

    @property
    def yolo_seg_polygon(self) -> List[List[float]]:
        """
        Returns segmentation normalised to [0,1] for each polygon,
        compatible with YOLOv8/v11-seg label format.
        """
        normalised = []
        for poly in self.segmentation:
            pts = np.array(poly).reshape(-1, 2)
            pts[:, 0] /= self.image_width
            pts[:, 1] /= self.image_height
            normalised.append(pts.flatten().tolist())
        return normalised

    def to_yolo_label_line(self) -> str:
        """
        YOLOv8-seg format:  class_id x1 y1 x2 y2 ... (normalised, space-sep)
        One line per polygon.
        """
        lines = []
        for poly in self.yolo_seg_polygon:
            coords = " ".join(f"{v:.6f}" for v in poly)
            lines.append(f"{self.category_id} {coords}")
        return "\n".join(lines)


@dataclass
class ObjectCrop:
    record: AnnotationRecord
    crop_rgb: np.ndarray
    mask_crop: np.ndarray
    bbox_in_image: Tuple[int, int, int, int]   # x1 y1 x2 y2


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class COCOParser:
    """
    Wraps pycocotools.COCO and exposes domain-friendly iterators.

    Parameters
    ----------
    annotations_file : path to COCO JSON
    images_dir       : root directory where image files live
    category_filter  : if given, only yield annotations of these category names
    """

    def __init__(
        self,
        annotations_file: str | Path,
        images_dir: str | Path,
        category_filter: Optional[List[str]] = None,
    ) -> None:
        self.annotations_file = Path(annotations_file)
        self.images_dir = Path(images_dir)
        self.category_filter = set(category_filter) if category_filter else None

        logger.info("Loading COCO annotations from %s", self.annotations_file)
        self._coco = COCO(str(self.annotations_file))

        # Build category id → name mapping
        self._cat_id_to_name: Dict[int, str] = {
            cat["id"]: cat["name"]
            for cat in self._coco.loadCats(self._coco.getCatIds())
        }
        logger.info(
            "Found %d images, %d categories, %d annotations",
            len(self._coco.imgs),
            len(self._cat_id_to_name),
            len(self._coco.anns),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def iter_annotations(self) -> Iterator[AnnotationRecord]:
        """Yield one AnnotationRecord per annotation (filtered if configured)."""
        for ann_id, ann in self._coco.anns.items():
            cat_name = self._cat_id_to_name.get(ann["category_id"], "unknown")
            if self.category_filter and cat_name not in self.category_filter:
                continue
            if not ann.get("segmentation"):
                logger.debug("Skipping ann %d — no segmentation.", ann_id)
                continue

            img_info = self._coco.imgs[ann["image_id"]]
            img_path = self.images_dir / img_info["file_name"]

            yield AnnotationRecord(
                annotation_id=ann_id,
                image_id=ann["image_id"],
                category_id=ann["category_id"],
                category_name=cat_name,
                segmentation=ann["segmentation"],
                bbox=tuple(ann["bbox"]),
                area=ann.get("area", 0.0),
                image_path=img_path,
                image_width=img_info["width"],
                image_height=img_info["height"],
            )

    def load_object_crop(
        self,
        record: AnnotationRecord,
        padding: int = 10,
    ) -> ObjectCrop:
        """Load the source image and extract the masked crop for a single annotation."""
        img_rgb = load_image_rgb(record.image_path)
        crop, mask_crop, bbox = crop_with_mask(img_rgb, record.binary_mask, padding=padding)
        return ObjectCrop(
            record=record,
            crop_rgb=crop,
            mask_crop=mask_crop,
            bbox_in_image=bbox,
        )

    def get_image_annotations(self, image_id: int) -> List[AnnotationRecord]:
        """Return all (filtered) annotations for a single image id."""
        ann_ids = self._coco.getAnnIds(imgIds=[image_id])
        records = []
        for ann in self._coco.loadAnns(ann_ids):
            cat_name = self._cat_id_to_name.get(ann["category_id"], "unknown")
            if self.category_filter and cat_name not in self.category_filter:
                continue
            img_info = self._coco.imgs[image_id]
            records.append(AnnotationRecord(
                annotation_id=ann["id"],
                image_id=image_id,
                category_id=ann["category_id"],
                category_name=cat_name,
                segmentation=ann["segmentation"],
                bbox=tuple(ann["bbox"]),
                area=ann.get("area", 0.0),
                image_path=self.images_dir / img_info["file_name"],
                image_width=img_info["width"],
                image_height=img_info["height"],
            ))
        return records

    def export_yolo_labels(self, output_dir: str | Path) -> None:
        """
        Write one .txt label file per image in YOLOv8-seg format.
        Also writes a classes.txt mapping index → category name.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Stable category ordering
        cat_ids = sorted(self._cat_id_to_name.keys())
        cat_index = {cid: i for i, cid in enumerate(cat_ids)}

        (output_dir / "classes.txt").write_text(
            "\n".join(self._cat_id_to_name[cid] for cid in cat_ids)
        )

        grouped: Dict[int, List[AnnotationRecord]] = {}
        for rec in self.iter_annotations():
            grouped.setdefault(rec.image_id, []).append(rec)

        for image_id, records in grouped.items():
            img_info = self._coco.imgs[image_id]
            label_name = Path(img_info["file_name"]).stem + ".txt"
            lines = []
            for rec in records:
                for poly in rec.yolo_seg_polygon:
                    coords = " ".join(f"{v:.6f}" for v in poly)
                    lines.append(f"{cat_index[rec.category_id]} {coords}")
            (output_dir / label_name).write_text("\n".join(lines))

        logger.info("Exported YOLO-seg labels to %s", output_dir)

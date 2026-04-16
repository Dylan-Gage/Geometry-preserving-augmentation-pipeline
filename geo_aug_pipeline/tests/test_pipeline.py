"""
tests/test_pipeline.py

Unit tests for core pipeline components.
Run with:  pytest tests/ -v
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generation.api_client import MockEditClient
from generation.augmenter import GeometryPreservingAugmenter
from utils.image_utils import (
    composite_foreground,
    crop_with_mask,
    polygon_to_binary_mask,
)
from validation.spatial_validator import SpatialValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image():
    """200x200 RGB image with a blue rectangle in the centre."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :] = [180, 140, 100]       # brownish background
    img[60:140, 60:140] = [30, 80, 200]  # blue foreground object
    return img


@pytest.fixture
def sample_mask():
    """Binary mask matching the blue rectangle in sample_image."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[60:140, 60:140] = 255
    return mask


# ---------------------------------------------------------------------------
# utils/image_utils
# ---------------------------------------------------------------------------

class TestPolygonToMask:
    def test_square_polygon(self):
        poly = [10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]
        mask = polygon_to_binary_mask([poly], height=100, width=100)
        assert mask.dtype == np.uint8
        assert mask[30, 30] == 255
        assert mask[5, 5] == 0

    def test_empty_polygon_list(self):
        mask = polygon_to_binary_mask([], height=50, width=50)
        assert mask.sum() == 0


class TestCropWithMask:
    def test_crop_shape(self, sample_image, sample_mask):
        crop, mask_crop, bbox = crop_with_mask(sample_image, sample_mask, padding=5)
        x1, y1, x2, y2 = bbox
        assert crop.shape[:2] == mask_crop.shape
        assert x2 - x1 == crop.shape[1]
        assert y2 - y1 == crop.shape[0]

    def test_empty_mask_raises(self, sample_image):
        empty = np.zeros((200, 200), dtype=np.uint8)
        with pytest.raises(ValueError):
            crop_with_mask(sample_image, empty)


class TestCompositeForeground:
    def test_pixels_are_identical_after_composite(self, sample_image, sample_mask):
        # Simulate API returning a heavily modified image
        edited = np.zeros_like(sample_image)
        edited[:, :] = [255, 0, 0]  # all red

        result = composite_foreground(sample_image, edited, sample_mask)
        fg = sample_mask > 0

        # Foreground must match original exactly
        np.testing.assert_array_equal(result[fg], sample_image[fg])
        # Background must match edited
        bg = ~fg
        np.testing.assert_array_equal(result[bg], edited[bg])

    def test_shape_preserved(self, sample_image, sample_mask):
        edited = np.random.randint(0, 256, sample_image.shape, dtype=np.uint8)
        result = composite_foreground(sample_image, edited, sample_mask)
        assert result.shape == sample_image.shape


# ---------------------------------------------------------------------------
# generation/api_client (MockEditClient)
# ---------------------------------------------------------------------------

class TestMockEditClient:
    def test_returns_same_shape(self, sample_image, sample_mask):
        client = MockEditClient()
        result = client.edit_image(sample_image, sample_mask, "rain environment")
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_background_changes(self, sample_image, sample_mask):
        client = MockEditClient()
        result = client.edit_image(sample_image, sample_mask, "rain environment")
        bg = sample_mask == 0
        # Background pixels should differ from original
        diff = np.abs(result[bg].astype(int) - sample_image[bg].astype(int)).mean()
        assert diff > 1.0, "Mock did not change the background"


# ---------------------------------------------------------------------------
# generation/augmenter
# ---------------------------------------------------------------------------

class TestGeometryPreservingAugmenter:
    def test_foreground_pixel_identical(self, sample_image, sample_mask):
        client = MockEditClient()
        augmenter = GeometryPreservingAugmenter(
            client=client,
            active_prompts=["rain_environment"],
            rate_limit_rpm=9999,
        )
        result = augmenter.augment_single(
            sample_image, sample_mask, "rain_environment",
        )
        assert result.success
        fg = sample_mask > 0
        np.testing.assert_array_equal(
            result.augmented_image[fg],
            sample_image[fg],
            err_msg="Foreground pixels must be pixel-identical after composite lock",
        )

    def test_failed_augmentation_returns_original(self, sample_image, sample_mask):
        class BrokenClient(MockEditClient):
            def edit_image(self, *a, **kw):
                raise RuntimeError("Simulated API failure")

        augmenter = GeometryPreservingAugmenter(
            client=BrokenClient(),
            active_prompts=["rain_environment"],
            max_retries=0,
            rate_limit_rpm=9999,
        )
        result = augmenter.augment_single(sample_image, sample_mask, "rain_environment")
        assert not result.success
        assert result.error_message != ""


# ---------------------------------------------------------------------------
# validation/spatial_validator
# ---------------------------------------------------------------------------

class TestSpatialValidator:
    def test_identical_images_pass(self, sample_image, sample_mask):
        validator = SpatialValidator(ssim_threshold=0.80)
        report = validator.validate(sample_image, sample_image.copy(), sample_mask)
        assert report.ssim_score > 0.99
        assert report.pixel_drift < 0.01
        assert report.passed

    def test_corrupted_foreground_fails(self, sample_image, sample_mask):
        corrupted = sample_image.copy()
        corrupted[sample_mask > 0] = 0   # zero out foreground
        validator = SpatialValidator(
            ssim_threshold=0.80,
            pixel_drift_threshold=1.0,
        )
        report = validator.validate(sample_image, corrupted, sample_mask)
        assert not report.passed
        assert len(report.failure_reasons) > 0

    def test_empty_mask_does_not_crash(self, sample_image):
        empty_mask = np.zeros((200, 200), dtype=np.uint8)
        validator = SpatialValidator()
        # Should not raise; returns perfect scores (nothing to check)
        report = validator.validate(sample_image, sample_image.copy(), empty_mask)
        assert report.ssim_score == 1.0


# ---------------------------------------------------------------------------
# curation/curator
# ---------------------------------------------------------------------------

class TestCurator:
    def test_accepted_image_saved_to_gold(self, sample_image, sample_mask, tmp_path):
        from curation.curator import Curator
        from generation.augmenter import AugmentationResult
        from validation.spatial_validator import ValidationReport

        curator = Curator(
            gold_dir=str(tmp_path / "gold"),
            discarded_dir=str(tmp_path / "discarded"),
            save_metadata_json=True,
        )
        aug = AugmentationResult(
            source_image_path="/data/raw/img001.jpg",
            annotation_id=42,
            prompt_key="rain_environment",
            augmented_image=sample_image,
            binary_mask=sample_mask,
            request_id="abc12345",
            success=True,
        )
        report = ValidationReport(
            annotation_id=42,
            prompt_key="rain_environment",
            ssim_score=0.97,
            reprojection_error_px=0.5,
            pixel_drift=0.1,
            passed=True,
            failure_reasons=[],
        )
        saved_path = curator.curate(aug, report)
        assert saved_path.exists()
        assert (tmp_path / "gold").is_dir()
        # Check sidecar
        json_path = saved_path.with_suffix(".json")
        assert json_path.exists()
        meta = json.loads(json_path.read_text())
        assert meta["validation"]["passed"] is True
        assert curator.stats["accepted"] == 1

    def test_rejected_image_saved_to_discarded(self, sample_image, sample_mask, tmp_path):
        from curation.curator import Curator
        from generation.augmenter import AugmentationResult
        from validation.spatial_validator import ValidationReport

        curator = Curator(
            gold_dir=str(tmp_path / "gold"),
            discarded_dir=str(tmp_path / "discarded"),
        )
        aug = AugmentationResult(
            source_image_path="/data/raw/img002.jpg",
            annotation_id=7,
            prompt_key="night_scene",
            augmented_image=sample_image,
            binary_mask=sample_mask,
            request_id="xyz99999",
            success=True,
        )
        report = ValidationReport(
            annotation_id=7,
            prompt_key="night_scene",
            ssim_score=0.40,
            reprojection_error_px=8.5,
            pixel_drift=12.0,
            passed=False,
            failure_reasons=["SSIM too low", "Pixel drift too high"],
        )
        saved_path = curator.curate(aug, report)
        assert (tmp_path / "discarded" / saved_path.name).exists()
        assert curator.stats["rejected"] == 1

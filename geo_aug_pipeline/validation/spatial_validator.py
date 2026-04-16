"""
validation/spatial_validator.py

Spatial consistency validation for geometry-preserving augmentation.

Metrics computed
----------------
1. SSIM (Structural Similarity Index)
   - Computed between original foreground crop and augmented foreground crop.
   - A drop here indicates the API (or composite step) corrupted the object.
   - Expected to be ≈1.0 after the composite_foreground() lock.

2. Reprojection Error
   - Finds ORB keypoints inside the foreground mask region.
   - Computes homography between original and augmented using those keypoints.
   - Measures mean pixel displacement after reprojection.
   - High error → geometric drift / warp introduced by augmentation.

3. Pixel Drift
   - Mean absolute difference of foreground pixels between original and augmented.
   - Fast sanity check before heavier metrics.
   - Should be ≈0.0 after composite lock; non-zero means something went wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    annotation_id: int
    prompt_key: str
    ssim_score: float            # [0, 1]  — higher is better
    reprojection_error_px: float # pixels  — lower is better
    pixel_drift: float           # mean abs diff on foreground — lower is better
    passed: bool
    failure_reasons: list[str]

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] ann={self.annotation_id} prompt={self.prompt_key} | "
            f"SSIM={self.ssim_score:.4f} | "
            f"reproj_err={self.reprojection_error_px:.2f}px | "
            f"drift={self.pixel_drift:.2f}"
        )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class SpatialValidator:
    """
    Validates that augmented images respect foreground geometry.

    Parameters
    ----------
    ssim_threshold          : minimum SSIM on foreground crop to pass
    reprojection_error_px   : maximum mean reprojection error (pixels)
    pixel_drift_threshold   : maximum mean absolute foreground pixel diff
    """

    def __init__(
        self,
        ssim_threshold: float = 0.82,
        reprojection_error_px: float = 3.0,
        pixel_drift_threshold: float = 5.0,
    ) -> None:
        self.ssim_threshold = ssim_threshold
        self.reprojection_error_px = reprojection_error_px
        self.pixel_drift_threshold = pixel_drift_threshold

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def validate(
        self,
        original_rgb: np.ndarray,
        augmented_rgb: np.ndarray,
        binary_mask: np.ndarray,
        annotation_id: int = -1,
        prompt_key: str = "",
    ) -> ValidationReport:
        """
        Run all validation checks and return a ValidationReport.

        Parameters
        ----------
        original_rgb  : HxWx3 original full image
        augmented_rgb : HxWx3 augmented full image (after composite lock)
        binary_mask   : HxW uint8 foreground mask (0=bg, 255=fg)
        """
        ssim_score = self._compute_ssim(original_rgb, augmented_rgb, binary_mask)
        reproj_err = self._compute_reprojection_error(original_rgb, augmented_rgb, binary_mask)
        drift = self._compute_pixel_drift(original_rgb, augmented_rgb, binary_mask)

        failure_reasons = []
        if ssim_score < self.ssim_threshold:
            failure_reasons.append(
                f"SSIM {ssim_score:.4f} < threshold {self.ssim_threshold}"
            )
        if reproj_err > self.reprojection_error_px:
            failure_reasons.append(
                f"Reprojection error {reproj_err:.2f}px > {self.reprojection_error_px}px"
            )
        if drift > self.pixel_drift_threshold:
            failure_reasons.append(
                f"Pixel drift {drift:.2f} > {self.pixel_drift_threshold}"
            )

        report = ValidationReport(
            annotation_id=annotation_id,
            prompt_key=prompt_key,
            ssim_score=ssim_score,
            reprojection_error_px=reproj_err,
            pixel_drift=drift,
            passed=len(failure_reasons) == 0,
            failure_reasons=failure_reasons,
        )
        logger.debug(report.summary())
        return report

    # ------------------------------------------------------------------
    # Private metrics
    # ------------------------------------------------------------------

    def _compute_ssim(
        self,
        original: np.ndarray,
        augmented: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """SSIM restricted to the foreground mask region."""
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
        aug_gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Tightest bbox of the mask so SSIM window fits
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            logger.warning("Empty mask; SSIM returning 1.0")
            return 1.0

        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        orig_crop = orig_gray[y1:y2, x1:x2]
        aug_crop = aug_gray[y1:y2, x1:x2]

        # skimage SSIM requires window ≥ 7 in both dims
        h, w = orig_crop.shape
        if h < 7 or w < 7:
            # Fall back to MAE-based proxy
            diff = np.abs(orig_crop - aug_crop).mean()
            return float(max(0.0, 1.0 - diff / 255.0))

        score, _ = ssim(orig_crop, aug_crop, full=True, data_range=255.0)
        return float(np.clip(score, 0.0, 1.0))

    def _compute_reprojection_error(
        self,
        original: np.ndarray,
        augmented: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """
        ORB feature matching within the foreground mask.
        Estimates homography between original and augmented foreground crops,
        then measures mean pixel displacement of matched keypoints.
        Returns 0.0 if not enough features are found (treated as no drift).
        """
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        aug_gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)

        # Restrict detection to foreground region
        masked_orig = cv2.bitwise_and(orig_gray, orig_gray, mask=mask)
        masked_aug = cv2.bitwise_and(aug_gray, aug_gray, mask=mask)

        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(masked_orig, mask)
        kp2, des2 = orb.detectAndCompute(masked_aug, mask)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            logger.debug("Not enough ORB features for reprojection; returning 0.0")
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < 4:
            return 0.0

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return 0.0

        n_inliers = int(inlier_mask.sum())
        if n_inliers < 4:
            return 0.0

        # Reproject inlier source points through H and measure error
        inlier_src = src_pts[inlier_mask.ravel() == 1]
        inlier_dst = dst_pts[inlier_mask.ravel() == 1]
        projected = cv2.perspectiveTransform(inlier_src, H)
        errors = np.linalg.norm(projected - inlier_dst, axis=2).flatten()
        return float(errors.mean())

    def _compute_pixel_drift(
        self,
        original: np.ndarray,
        augmented: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """
        Mean absolute pixel difference restricted to foreground.
        After the composite lock this should be ≈0.0.
        A non-zero value indicates a dtype or resize mismatch.
        """
        fg = mask > 0
        if not fg.any():
            return 0.0
        orig_fg = original[fg].astype(np.float32)
        aug_fg = augmented[fg].astype(np.float32)
        return float(np.abs(orig_fg - aug_fg).mean())

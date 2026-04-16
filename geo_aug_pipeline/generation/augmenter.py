"""
generation/augmenter.py

Orchestrates the geometry-preserving augmentation loop:

  1. Take a source image + binary mask
  2. Call the API client with a prompt
  3. Composite the original foreground BACK onto the result
     (pixel-identical guarantee, regardless of API behaviour)
  4. Return the augmented full image + the relocated mask

Also contains:
  - RateLimiter   — token-bucket to respect API RPM limits
  - RetryWrapper  — exponential-backoff retry around client.edit_image()
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import List, Optional, Tuple

import numpy as np

from generation.api_client import ImageEditClient, MockEditClient
from prompts.templates import get_prompt
from utils.image_utils import composite_foreground
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Rate limiter (token-bucket, thread-safe)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Enforces a maximum of `max_rpm` calls per 60-second window."""

    def __init__(self, max_rpm: int) -> None:
        self._interval = 60.0 / max(max_rpm, 1)   # seconds per token
        self._lock = Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                logger.debug("Rate limiter sleeping %.2f s", wait)
                time.sleep(wait)
            self._next_allowed = time.monotonic() + self._interval


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def _call_with_retry(
    fn,
    *args,
    max_retries: int = 4,
    backoff_base: float = 2.0,
    rate_limiter: Optional[RateLimiter] = None,
    **kwargs,
):
    """
    Call fn(*args, **kwargs) with exponential-backoff retry.
    Handles:
      - Generic exceptions (network, timeout)
      - 429 / rate-limit errors (detected by exception message)
    """
    last_exc: Exception = RuntimeError("No attempts made.")
    for attempt in range(max_retries + 1):
        try:
            if rate_limiter:
                rate_limiter.acquire()
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            msg = str(exc).lower()
            if "429" in msg or "rate" in msg or "quota" in msg:
                # For rate-limit errors, wait longer
                sleep_time = backoff_base ** (attempt + 2)
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Sleeping %.1f s …",
                    attempt + 1, max_retries, sleep_time,
                )
            else:
                sleep_time = backoff_base ** attempt
                logger.warning(
                    "API error on attempt %d/%d: %s. Retrying in %.1f s …",
                    attempt + 1, max_retries, exc, sleep_time,
                )
            time.sleep(sleep_time)

    raise RuntimeError(f"All {max_retries + 1} attempts failed. Last error: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AugmentationResult:
    source_image_path: str
    annotation_id: int
    prompt_key: str
    augmented_image: np.ndarray          # full image, HxWx3
    binary_mask: np.ndarray              # HxW, unchanged from source
    request_id: str
    success: bool
    error_message: str = ""


# ---------------------------------------------------------------------------
# Augmenter
# ---------------------------------------------------------------------------

class GeometryPreservingAugmenter:
    """
    High-level augmentation engine.

    The foreground composite step is the core safety mechanism:
    after the API returns an edited image, we paste the original foreground
    pixels back. This makes foreground preservation guaranteed by code,
    not just by prompt engineering.
    """

    def __init__(
        self,
        client: ImageEditClient,
        active_prompts: List[str],
        max_retries: int = 4,
        backoff_base: float = 2.0,
        rate_limit_rpm: int = 10,
        object_description: str = "accessibility button or door panel",
    ) -> None:
        self._client = client
        self._active_prompts = active_prompts
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        # Skip rate limiting entirely for mock — it's local, no API quota involved
        self._rate_limiter = None if isinstance(client, MockEditClient) else RateLimiter(rate_limit_rpm)
        self._object_description = object_description

    def augment_single(
        self,
        image_rgb: np.ndarray,
        binary_mask: np.ndarray,
        prompt_key: str,
        source_path: str = "",
        annotation_id: int = -1,
    ) -> AugmentationResult:
        """
        Generate one augmented variant for (image, mask, prompt_key).

        Steps
        -----
        1. Render the full prompt text
        2. Call API with retry/rate-limit wrapping
        3. Composite original foreground back (geometry lock)
        4. Return AugmentationResult
        """
        request_id = str(uuid.uuid4())[:8]
        prompt_text = get_prompt(prompt_key, self._object_description)

        logger.info(
            "Augmenting | ann=%d | prompt=%s | req=%s",
            annotation_id, prompt_key, request_id,
        )

        try:
            edited = _call_with_retry(
                self._client.edit_image,
                image_rgb,
                binary_mask,
                prompt_text,
                request_id=request_id,
                max_retries=self._max_retries,
                backoff_base=self._backoff_base,
                rate_limiter=self._rate_limiter,
            )

            # ---- GEOMETRY LOCK: paste original foreground unconditionally ----
            final = composite_foreground(image_rgb, edited, binary_mask)

            return AugmentationResult(
                source_image_path=source_path,
                annotation_id=annotation_id,
                prompt_key=prompt_key,
                augmented_image=final,
                binary_mask=binary_mask,
                request_id=request_id,
                success=True,
            )

        except Exception as exc:
            logger.error(
                "Augmentation failed | ann=%d | prompt=%s | req=%s | err=%s",
                annotation_id, prompt_key, request_id, exc,
            )
            return AugmentationResult(
                source_image_path=source_path,
                annotation_id=annotation_id,
                prompt_key=prompt_key,
                augmented_image=image_rgb.copy(),   # fallback to original
                binary_mask=binary_mask,
                request_id=request_id,
                success=False,
                error_message=str(exc),
            )

    def augment_all_prompts(
        self,
        image_rgb: np.ndarray,
        binary_mask: np.ndarray,
        source_path: str = "",
        annotation_id: int = -1,
    ) -> List[AugmentationResult]:
        """Run augmentation for every active prompt template."""
        results = []
        for prompt_key in self._active_prompts:
            r = self.augment_single(
                image_rgb=image_rgb,
                binary_mask=binary_mask,
                prompt_key=prompt_key,
                source_path=source_path,
                annotation_id=annotation_id,
            )
            results.append(r)
        return results
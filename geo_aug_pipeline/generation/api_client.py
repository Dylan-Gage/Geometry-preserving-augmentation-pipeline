"""
generation/api_client.py
Abstraction layer for the image-edit API.
Uses the current google-genai SDK (not the deprecated google-generativeai).
"""
from __future__ import annotations

import abc
import io
from typing import Optional

import numpy as np
from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)


class ImageEditClient(abc.ABC):
    @abc.abstractmethod
    def edit_image(self, image_rgb, mask_binary, prompt, *, request_id="") -> np.ndarray:
        pass


class MockEditClient(ImageEditClient):
    _OVERLAY_MAP = {
        "rain":  (0.55, 0.65, 0.75),
        "night": (0.05, 0.05, 0.20),
        "snow":  (0.85, 0.90, 0.95),
        "wall":  (0.60, 0.50, 0.40),
        "foggy": (0.80, 0.80, 0.80),
    }
    _DEFAULT_OVERLAY = (0.70, 0.70, 0.70)

    def edit_image(self, image_rgb, mask_binary, prompt, *, request_id=""):
        logger.debug("[Mock] Generating fake background edit | id=%s", request_id)
        prompt_lower = prompt.lower()
        overlay_rgb = self._DEFAULT_OVERLAY
        for kw, col in self._OVERLAY_MAP.items():
            if kw in prompt_lower:
                overlay_rgb = col
                break
        result = image_rgb.astype(np.float32) / 255.0
        bg_mask = (mask_binary == 0)
        for ch, val in enumerate(overlay_rgb):
            channel = result[:, :, ch]
            channel[bg_mask] = channel[bg_mask] * 0.4 + val * 0.6
            result[:, :, ch] = channel
        noise = np.random.normal(0, 0.02, result.shape).astype(np.float32)
        result[np.stack([bg_mask] * 3, axis=-1)] += noise[np.stack([bg_mask] * 3, axis=-1)]
        return np.clip(result * 255, 0, 255).astype(np.uint8)


class GeminiEditClient(ImageEditClient):
    """Uses the current google-genai SDK. Install: pip install google-genai"""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-preview-image-generation") -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError("Run: pip install google-genai") from exc

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._model_name = model
        logger.info("GeminiEditClient initialised with model=%s", model)

    def edit_image(self, image_rgb, mask_binary, prompt, *, request_id=""):
        logger.info("[Gemini] Requesting edit | id=%s | model=%s", request_id, self._model_name)

        buf = io.BytesIO()
        Image.fromarray(image_rgb).save(buf, format="PNG")
        buf.seek(0)

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=[
                self._types.Part.from_bytes(data=buf.read(), mime_type="image/png"),
                prompt,
            ],
            config=self._types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image"):
                result_rgb = np.array(Image.open(io.BytesIO(part.inline_data.data)).convert("RGB"))
                if result_rgb.shape[:2] != image_rgb.shape[:2]:
                    import cv2
                    result_rgb = cv2.resize(result_rgb, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                logger.info("[Gemini] Edit received | id=%s", request_id)
                return result_rgb

        raise RuntimeError("Gemini returned no image in response.")


def build_client(use_mock: bool, api_key: str = "", model: str = "") -> ImageEditClient:
    if use_mock:
        logger.info("Using MockEditClient (offline mode).")
        return MockEditClient()
    if not api_key:
        raise ValueError("use_mock=False but no API key provided.")
    return GeminiEditClient(api_key=api_key, model=model)
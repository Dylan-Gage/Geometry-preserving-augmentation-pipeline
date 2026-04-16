"""
utils/config_loader.py
Loads and validates the YAML config, exposing a typed dataclass interface.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


# ---------------------------------------------------------------------------
# Dataclasses (typed config sections)
# ---------------------------------------------------------------------------

@dataclass
class PathsConfig:
    raw_images_dir: str
    annotations_file: str
    augmented_dir: str
    gold_standard_dir: str
    discarded_dir: str
    logs_dir: str

    def ensure_dirs(self) -> None:
        for attr in ("augmented_dir", "gold_standard_dir", "discarded_dir", "logs_dir"):
            Path(getattr(self, attr)).mkdir(parents=True, exist_ok=True)


@dataclass
class AugmentationConfig:
    variants_per_image: int
    active_prompts: List[str]
    model: str
    use_mock: bool


@dataclass
class ApiConfig:
    gemini_api_key_env: str
    max_retries: int
    retry_backoff_base: float
    rate_limit_rpm: int
    request_timeout_sec: int

    @property
    def api_key(self) -> str:
        return os.environ.get(self.gemini_api_key_env, "")


@dataclass
class ValidationConfig:
    ssim_threshold: float
    reprojection_error_px: float
    pixel_drift_threshold: float


@dataclass
class CurationConfig:
    save_metadata_json: bool
    gold_copy_originals: bool


@dataclass
class BatchConfig:
    batch_size: int
    max_workers: int
    checkpoint_every: int


@dataclass
class LoggingConfig:
    level: str
    log_to_file: bool
    log_filename: str


@dataclass
class PipelineConfig:
    paths: PathsConfig
    augmentation: AugmentationConfig
    api: ApiConfig
    validation: ValidationConfig
    curation: CurationConfig
    batch: BatchConfig
    logging: LoggingConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "configs/pipeline_config.yaml") -> PipelineConfig:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as fh:
        raw = yaml.safe_load(fh)

    cfg = PipelineConfig(
        paths=PathsConfig(**raw["paths"]),
        augmentation=AugmentationConfig(**raw["augmentation"]),
        api=ApiConfig(**raw["api"]),
        validation=ValidationConfig(**raw["validation"]),
        curation=CurationConfig(**raw["curation"]),
        batch=BatchConfig(**raw["batch"]),
        logging=LoggingConfig(**raw["logging"]),
    )
    cfg.paths.ensure_dirs()
    return cfg

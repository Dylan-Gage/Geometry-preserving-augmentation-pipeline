# Geometry-Preserving Data Augmentation Pipeline

A production-ready modular Python pipeline for augmenting accessibility datasets
(door buttons, elevator panels, etc.) while guaranteeing pixel-identical foreground geometry.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Dry run (validates config + dataset without API calls)
python main.py --dry-run

# 3. Run full pipeline (mock mode — no API key needed)
python main.py

# 4. Run with real Gemini API
export GEMINI_API_KEY=your_key_here
# Edit configs/pipeline_config.yaml: set use_mock: false
python main.py

# 5. Export YOLO-seg labels only
python main.py --export-labels

# 6. Run tests
pytest tests/ -v --tb=short
```

## Directory Layout

```
geo_aug_pipeline/
├── main.py                        ← Pipeline entry point
├── requirements.txt
├── configs/
│   └── pipeline_config.yaml       ← All tunable parameters
├── ingestion/
│   └── coco_parser.py             ← COCO JSON parsing, mask extraction, YOLO export
├── generation/
│   ├── api_client.py              ← Gemini / Mock client abstraction
│   ├── augmenter.py               ← Geometry-lock augmentation loop
│   └── batch_processor.py        ← Parallel batch processing + checkpointing
├── validation/
│   └── spatial_validator.py       ← SSIM, reprojection error, pixel drift
├── curation/
│   └── curator.py                 ← Accept → gold_standard / Reject → discarded
├── prompts/
│   └── templates.py               ← 5 prompt templates with preservation enforcement
├── utils/
│   ├── config_loader.py           ← Typed YAML config dataclasses
│   ├── image_utils.py             ← CV helpers (crop, composite, mask conversion)
│   └── logger.py                  ← Centralised logging setup
├── tests/
│   └── test_pipeline.py           ← Unit tests (pytest)
└── data/
    ├── raw/                       ← Input images + annotations.json
    ├── augmented/                 ← Intermediate augmented outputs
    ├── gold_standard/             ← Accepted (passed validation)
    └── discarded/                 ← Rejected (failed validation)
```

## Geometry Preservation Guarantee

The pipeline enforces foreground preservation at **two independent levels**:

1. **Prompt Engineering** — Every prompt contains explicit preservation clauses,
   negative instructions, and hard constraints telling the model not to alter
   the masked foreground object.

2. **Programmatic Composite Lock** — After receiving any API response,
   `composite_foreground()` pastes original foreground pixels back unconditionally.
   This means even a misbehaving API cannot corrupt the object geometry.

## Key Config Options

| Key | Default | Meaning |
|-----|---------|---------|
| `augmentation.use_mock` | `true` | Use offline mock (no API key needed) |
| `augmentation.variants_per_image` | `3` | Augmented variants per annotation |
| `validation.ssim_threshold` | `0.82` | Minimum SSIM to accept an image |
| `batch.max_workers` | `4` | Parallel threads |
| `api.max_retries` | `4` | Retry attempts per API call |

## Validation Metrics

- **SSIM** — Structural similarity on the foreground crop (should be ≈1.0 after composite lock)
- **Reprojection Error** — ORB keypoint homography residuals inside the foreground mask
- **Pixel Drift** — Mean absolute foreground pixel difference (should be ≈0.0)

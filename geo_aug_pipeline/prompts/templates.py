"""
prompts/templates.py

Indoor-focused prompt templates for accessibility button augmentation.
All prompts include an explicit anti-hallucination clause to prevent
Gemini from adding buttons, panels, or UI elements that don't exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class PromptTemplate:
    key: str
    description: str
    build: Callable[[str], str]


TEMPLATES: Dict[str, PromptTemplate] = {}


def _register(key: str, description: str) -> Callable:
    def decorator(fn: Callable[[str], str]) -> Callable[[str], str]:
        TEMPLATES[key] = PromptTemplate(key=key, description=description, build=fn)
        return fn
    return decorator


# Shared anti-hallucination clause inserted into every prompt
_NO_HALLUCINATION = """
CRITICAL — DO NOT ADD OR REMOVE OBJECTS:
- Do NOT add any new buttons, panels, screens, switches, labels, or hardware
  that do not already exist in the original image.
- Do NOT remove any existing objects from the scene.
- Only change surface appearance (colour, texture, material, lighting).
- The number of physical objects in the scene must remain identical."""


# ------------------------------------------------------------------
# 1. Concrete/brutalist wall
# ------------------------------------------------------------------
@_register("concrete_wall", "Bare concrete or brutalist wall background")
def _concrete(obj: str) -> str:
    return f"""Change the wall and background behind the {obj} to bare concrete or
brutalist unpainted cement. Keep the scene strictly indoors.

PRESERVE (non-negotiable):
- Every pixel of the {obj} exactly as it appears. Shape, colour, texture,
  mounting hardware — all immutable.
- Do not add lighting effects, shadows, or reflections onto the {obj} itself.

CHANGE only the background (outside the {obj}):
- Replace wall material with raw concrete: grey, slightly rough, with
  subtle formwork marks or aggregate texture.
- Adjust ambient indoor lighting to match a concrete environment (cool, diffuse).
- Floor and ceiling may also be concrete if visible.
{_NO_HALLUCINATION}

DO NOT:
- Place concrete texture over the {obj}.
- Change image dimensions or crop.
- Add any outdoor elements (sky, weather, vegetation)."""


# ------------------------------------------------------------------
# 2. Painted drywall / office interior
# ------------------------------------------------------------------
@_register("office_interior", "Clean painted office wall background")
def _office(obj: str) -> str:
    return f"""Change the background behind the {obj} to a clean, modern office
interior with painted drywall. Keep the scene strictly indoors.

PRESERVE (non-negotiable):
- The {obj} pixel-for-pixel. Its geometry, colour, and surface finish must
  not change in any way.

CHANGE only the background (outside the {obj}):
- Replace the wall with smooth painted drywall in a neutral colour
  (white, off-white, light grey, or beige).
- Add subtle indoor fluorescent or LED ambient lighting on the wall.
- Optionally show a carpeted or tiled floor edge and drop ceiling if visible.
{_NO_HALLUCINATION}

DO NOT:
- Add any weather effects, windows showing outdoors, or natural light.
- Alter the {obj}'s appearance in any way.
- Change image dimensions."""


# ------------------------------------------------------------------
# 3. Brick wall interior
# ------------------------------------------------------------------
@_register("brick_wall", "Exposed interior brick wall background")
def _brick(obj: str) -> str:
    return f"""Change the wall behind the {obj} to exposed interior brick.
The scene must remain indoors — this is an interior brick wall, not an
outdoor façade.

PRESERVE (non-negotiable):
- The {obj} completely unchanged — every pixel, edge, and colour.

CHANGE only the background (outside the {obj}):
- Replace the wall surface with realistic exposed red or brown brick,
  with visible mortar joints.
- Warm indoor ambient lighting consistent with an interior brick wall.
- No weather, no sky, no outdoor elements visible.
{_NO_HALLUCINATION}

DO NOT:
- Overlay brick texture onto the {obj}.
- Add any buttons, panels, or fixtures that are not already present.
- Change image dimensions or aspect ratio."""


# ------------------------------------------------------------------
# 4. Tile wall (bathroom / corridor)
# ------------------------------------------------------------------
@_register("tile_wall", "Tiled wall background, indoor corridor or lobby")
def _tile(obj: str) -> str:
    return f"""Change the wall behind the {obj} to ceramic or porcelain tiles,
as found in an indoor corridor, lobby, or accessible bathroom.
The scene must be entirely indoors.

PRESERVE (non-negotiable):
- The {obj} exactly as it appears — no changes to shape, colour,
  mounting, or surface texture.

CHANGE only the background (outside the {obj}):
- Replace the wall with square or rectangular tiles (e.g. white subway
  tile, grey large-format tile, or beige ceramic).
- Grout lines should be visible and consistent.
- Indoor overhead lighting (recessed LED or fluorescent) casting soft
  shadows on the tile, not on the {obj}.
{_NO_HALLUCINATION}

DO NOT:
- Extend tile texture over the {obj}.
- Add any outdoor scenery or weather.
- Introduce new hardware, buttons, or objects not present originally."""


# ------------------------------------------------------------------
# 5. Wooden panelling / warm interior
# ------------------------------------------------------------------
@_register("wood_panel_interior", "Warm wood-panelled indoor wall background")
def _wood(obj: str) -> str:
    return f"""Change the wall behind the {obj} to warm wooden wall panelling,
as found in a hotel lobby, residential building, or upscale indoor corridor.

PRESERVE (non-negotiable):
- The {obj} pixel-identically. Do not alter its colour, shape, edges,
  mounting screws, or any surface detail.

CHANGE only the background (outside the {obj}):
- Replace the wall with horizontal or vertical wood panelling —
  warm oak, walnut, or pine tones with visible grain.
- Warm incandescent or LED ambient lighting on the panelling.
- Scene must be entirely indoors; no sky or weather visible.
{_NO_HALLUCINATION}

DO NOT:
- Apply wood grain texture over the {obj}.
- Add any buttons, signs, or objects not already in the image.
- Change image canvas size or crop."""


# ------------------------------------------------------------------
# 6. Indoor wall texture (original — kept for backwards compatibility)
# ------------------------------------------------------------------
@_register("indoor_wall_texture", "Generic indoor wall texture change")
def _wall_texture(obj: str) -> str:
    return f"""Re-texture the wall and surrounding environment behind the {obj}
to a different indoor material (e.g., brick, concrete, or tile).
The scene must remain indoors — do not add any outdoor elements.

PRESERVE (non-negotiable):
- Every pixel of the {obj}. Its shape, colours, and edges are immutable.
- Existing mounting hardware or fixtures that are part of the {obj}.

CHANGE only what is outside the {obj}:
- Replace the current wall texture with a realistic indoor material.
- Adjust ambient indoor lighting to match the new material.
{_NO_HALLUCINATION}

DO NOT:
- Place wall texture over the {obj}.
- Add weather, sky, rain, snow, or any outdoor environment.
- Introduce new buttons, panels, or objects not already present.
- Change image dimensions."""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_prompt(key: str, object_description: str = "accessibility button") -> str:
    if key not in TEMPLATES:
        raise KeyError(f"Unknown prompt key '{key}'. Available: {list(TEMPLATES)}")
    return TEMPLATES[key].build(object_description)


def list_templates() -> Dict[str, str]:
    return {k: v.description for k, v in TEMPLATES.items()}
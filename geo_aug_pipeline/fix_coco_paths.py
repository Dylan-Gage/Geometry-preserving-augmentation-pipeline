# fix_coco_paths.py
import json
import shutil
import sys
from pathlib import Path

input_json   = sys.argv[1] if len(sys.argv) > 1 else "annotations_raw.json"
images_dir   = sys.argv[2] if len(sys.argv) > 2 else "."
output_json  = sys.argv[3] if len(sys.argv) > 3 else "data/raw/annotations.json"

with open(input_json, "r") as f:
    coco = json.load(f)

images_dir = Path(images_dir)
output_dir = Path("data/raw/images")
output_dir.mkdir(parents=True, exist_ok=True)

copied  = []
skipped = []
missing = []

for img in coco["images"]:
    filename = Path(img["file_name"]).name
    img["file_name"] = filename

    src = images_dir / filename
    dst = output_dir / filename

    if not src.exists():
        missing.append(filename)
    elif src.resolve() == dst.resolve():
        skipped.append(filename)   # already in the right place
    else:
        shutil.copy2(src, dst)
        copied.append(filename)

with open(output_json, "w") as f:
    json.dump(coco, f, indent=2)

print(f"JSON saved  → {output_json}")
print(f"Copied  ({len(copied)}):  {copied}")
print(f"Skipped ({len(skipped)}): already in data/raw/images/")
if missing:
    print(f"Missing ({len(missing)}): {missing}")
    print("  Put the missing .png files in your images folder and re-run.")
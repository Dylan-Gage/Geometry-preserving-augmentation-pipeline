"""
generate_results.py

Reads all .json sidecar files from gold_standard/ and discarded/
and generates:
  - Validation Score Summary Table (printed + saved as CSV)
  - Acceptance Rate by Prompt Table (printed + saved as CSV)
  - SSIM Distribution Histogram (saved as PNG)
  - Pixel Drift Distribution Histogram (saved as PNG)

Run from the geo_aug_pipeline root directory:
    python generate_results.py
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import csv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GOLD_DIR      = Path("data/gold_standard")
DISCARDED_DIR = Path("data/discarded")
OUTPUT_DIR    = Path("results")
SSIM_THRESHOLD = 0.90

OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load all sidecar JSON files
# ---------------------------------------------------------------------------

records = []

for folder in (GOLD_DIR, DISCARDED_DIR):
    for json_path in folder.glob("*.json"):
        try:
            data = json.loads(json_path.read_text())
            v = data.get("validation", {})
            records.append({
                "prompt_key":        data.get("prompt_key", "unknown"),
                "passed":            v.get("passed", False),
                "ssim":              v.get("ssim", None),
                "reprojection_error": v.get("reprojection_error_px", None),
                "pixel_drift":       v.get("pixel_drift", None),
                "failure_reasons":   v.get("failure_reasons", []),
            })
        except Exception as e:
            print(f"Skipping {json_path.name}: {e}")

if not records:
    print("No sidecar JSON files found. Make sure you have run the pipeline first.")
    exit()

print(f"Loaded {len(records)} records ({sum(r['passed'] for r in records)} passed, "
      f"{sum(not r['passed'] for r in records)} failed)\n")

# ---------------------------------------------------------------------------
# Helper: extract valid (non-None) values for a metric
# ---------------------------------------------------------------------------

def get_values(metric_key, passed_only=False, failed_only=False):
    vals = []
    for r in records:
        if passed_only and not r["passed"]:
            continue
        if failed_only and r["passed"]:
            continue
        v = r.get(metric_key)
        if v is not None:
            vals.append(float(v))
    return np.array(vals)

# ---------------------------------------------------------------------------
# 2. Validation Score Summary Table
# ---------------------------------------------------------------------------

print("=" * 65)
print("TABLE 1 — VALIDATION SCORE SUMMARY")
print("=" * 65)

metrics = [
    ("ssim",               "SSIM (foreground)",     SSIM_THRESHOLD, ">="),
    ("reprojection_error", "Reprojection Error (px)", 3.0,           "<="),
    ("pixel_drift",        "Pixel Drift",             5.0,           "<="),
]

summary_rows = []

for key, label, threshold, direction in metrics:
    vals = get_values(key)
    if len(vals) == 0:
        continue
    mean  = np.mean(vals)
    std   = np.std(vals)
    mn    = np.min(vals)
    mx    = np.max(vals)
    if direction == ">=":
        passes = np.sum(vals >= threshold)
    else:
        passes = np.sum(vals <= threshold)
    pass_rate = 100.0 * passes / len(vals)
    summary_rows.append((label, mean, std, mn, mx, pass_rate))
    print(f"  {label:<28} mean={mean:.4f}  std={std:.4f}  "
          f"min={mn:.4f}  max={mx:.4f}  pass={pass_rate:.1f}%")

print()

# Save as CSV
with open(OUTPUT_DIR / "validation_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Metric", "Mean", "Std Dev", "Min", "Max", "Pass Rate (%)"])
    for row in summary_rows:
        w.writerow([row[0], f"{row[1]:.4f}", f"{row[2]:.4f}",
                    f"{row[3]:.4f}", f"{row[4]:.4f}", f"{row[5]:.1f}"])
print("  Saved → results/validation_summary.csv")

# ---------------------------------------------------------------------------
# 3. Acceptance Rate by Prompt
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("TABLE 2 — ACCEPTANCE RATE BY PROMPT VARIANT")
print("=" * 65)

prompt_stats = defaultdict(lambda: {"total": 0, "passed": 0})
for r in records:
    pk = r["prompt_key"]
    prompt_stats[pk]["total"] += 1
    if r["passed"]:
        prompt_stats[pk]["passed"] += 1

prompt_rows = []
for pk, s in sorted(prompt_stats.items()):
    rate = 100.0 * s["passed"] / s["total"] if s["total"] else 0
    prompt_rows.append((pk, s["total"], s["passed"], s["total"] - s["passed"], rate))
    print(f"  {pk:<28} total={s['total']:>4}  "
          f"accepted={s['passed']:>4}  rejected={s['total']-s['passed']:>4}  "
          f"rate={rate:.1f}%")

print()

with open(OUTPUT_DIR / "acceptance_by_prompt.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Prompt", "Total", "Accepted", "Rejected", "Acceptance Rate (%)"])
    for row in prompt_rows:
        w.writerow([row[0], row[1], row[2], row[3], f"{row[4]:.1f}"])
print("  Saved → results/acceptance_by_prompt.csv")

# ---------------------------------------------------------------------------
# 4. SSIM Distribution Histogram
# ---------------------------------------------------------------------------

ssim_all    = get_values("ssim")
ssim_pass   = get_values("ssim", passed_only=True)
ssim_fail   = get_values("ssim", failed_only=True)

fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(0, 1, 41)

if len(ssim_pass) > 0:
    ax.hist(ssim_pass, bins=bins, color="#2ecc71", alpha=0.75,
            label=f"Passed (n={len(ssim_pass)})", edgecolor="white", linewidth=0.4)
if len(ssim_fail) > 0:
    ax.hist(ssim_fail, bins=bins, color="#e74c3c", alpha=0.75,
            label=f"Failed (n={len(ssim_fail)})", edgecolor="white", linewidth=0.4)

ax.axvline(SSIM_THRESHOLD, color="black", linestyle="--", linewidth=1.5,
           label=f"Threshold = {SSIM_THRESHOLD}")

ax.set_xlabel("SSIM Score (foreground crop)", fontsize=12)
ax.set_ylabel("Number of Images", fontsize=12)
ax.set_title("SSIM Distribution — Foreground Preservation Check", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim(0, 1)
ax.grid(axis="y", alpha=0.3)

# Annotate mean
if len(ssim_all) > 0:
    ax.axvline(np.mean(ssim_all), color="#3498db", linestyle=":",
               linewidth=1.5, label=f"Mean = {np.mean(ssim_all):.3f}")
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ssim_distribution.png", dpi=150)
plt.close()
print("\n  Saved → results/ssim_distribution.png")

# ---------------------------------------------------------------------------
# 5. Pixel Drift Distribution Histogram
# ---------------------------------------------------------------------------

drift_all  = get_values("pixel_drift")
drift_pass = get_values("pixel_drift", passed_only=True)
drift_fail = get_values("pixel_drift", failed_only=True)

fig, ax = plt.subplots(figsize=(9, 5))

# Most values cluster near 0; use log scale on y-axis if range is large
max_drift = drift_all.max() if len(drift_all) > 0 else 10
bins_drift = np.linspace(0, max(max_drift * 1.05, 1), 40)

if len(drift_pass) > 0:
    ax.hist(drift_pass, bins=bins_drift, color="#2ecc71", alpha=0.75,
            label=f"Passed (n={len(drift_pass)})", edgecolor="white", linewidth=0.4)
if len(drift_fail) > 0:
    ax.hist(drift_fail, bins=bins_drift, color="#e74c3c", alpha=0.75,
            label=f"Failed (n={len(drift_fail)})", edgecolor="white", linewidth=0.4)

ax.axvline(5.0, color="black", linestyle="--", linewidth=1.5, label="Threshold = 5.0")

ax.set_xlabel("Pixel Drift (mean absolute foreground difference)", fontsize=12)
ax.set_ylabel("Number of Images", fontsize=12)
ax.set_title("Pixel Drift Distribution — Geometry Lock Verification", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

note = "Expected spike at ~0.0 (PNG) or ~0.9 (JPEG) confirms geometry lock is intact"
ax.text(0.98, 0.95, note, transform=ax.transAxes, fontsize=8,
        ha="right", va="top", color="grey", style="italic")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pixel_drift_distribution.png", dpi=150)
plt.close()
print("  Saved → results/pixel_drift_distribution.png")

# ---------------------------------------------------------------------------
# 6. Acceptance Rate by Prompt — Bar Chart
# ---------------------------------------------------------------------------

if prompt_rows:
    labels = [r[0].replace("_", "\n") for r in prompt_rows]
    rates  = [r[4] for r in prompt_rows]
    colors = ["#2ecc71" if r >= 80 else "#e67e22" if r >= 50 else "#e74c3c"
              for r in rates]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, rates, color=colors, edgecolor="white", linewidth=0.5, width=0.6)

    ax.axhline(100, color="#2ecc71", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("Acceptance Rate (%)", fontsize=12)
    ax.set_title("Acceptance Rate by Prompt Variant", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "acceptance_by_prompt.png", dpi=150)
    plt.close()
    print("  Saved → results/acceptance_by_prompt.png")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("All results saved to results/")
print("  validation_summary.csv")
print("  acceptance_by_prompt.csv")
print("  ssim_distribution.png")
print("  pixel_drift_distribution.png")
print("  acceptance_by_prompt.png")
print("=" * 65)

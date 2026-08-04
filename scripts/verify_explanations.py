"""Verify that all SHAP explanation artifacts exist and are valid."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.explainability import (
    build_shap_explainer,
    get_processed_feature_names,
    load_background_data,
    load_model_pipeline,
)

EXPLANATIONS_DIR = BASE_DIR / "reports" / "explanations"

REQUIRED_ARTIFACTS = {
    "shap_summary_plot.png": "Global SHAP summary (beeswarm) plot",
    "shap_global_importance.png": "Global feature importance bar plot",
    "shap_dependence_plot.png": "SHAP dependence plot",
    "shap_waterfall_plot.png": "Local SHAP waterfall plot",
    "shap_force_plot.png": "Local SHAP force plot",
    "shap_contributions.csv": "Per-instance contribution table",
    "shap_local_contributions.csv": "Local contribution table",
}

print("=" * 70)
print("SHAP EXPLANATION ARTIFACT VERIFICATION")
print("=" * 70)

# 1. Check artifact files
print("\n[1] Artifact Files")
all_ok = True
for filename, description in REQUIRED_ARTIFACTS.items():
    path = EXPLANATIONS_DIR / filename
    exists = path.exists()
    size_kb = path.stat().st_size / 1024 if exists else 0
    status = "OK" if exists else "MISSING"
    if not exists:
        all_ok = False
    print(f"  [{status}] {filename} ({size_kb:.1f} KB) - {description}")

# 2. Validate model pipeline loading
print("\n[2] Model Pipeline")
try:
    pipeline = load_model_pipeline(BASE_DIR / "models" / "best_model.joblib")
    feature_names = get_processed_feature_names(pipeline)
    print(f"  [OK] Loaded pipeline with {len(feature_names)} processed features")
except Exception as exc:
    print(f"  [FAIL] {exc}")
    all_ok = False

# 3. Verify background data loading
print("\n[3] Background Data")
try:
    background = load_background_data(pipeline, sample_size=50)
    print(f"  [OK] Background data shape: {background.shape}")
except Exception as exc:
    print(f"  [FAIL] {exc}")
    all_ok = False

# 4. Verify SHAP explainer
print("\n[4] SHAP Explainer")
try:
    explainer = build_shap_explainer(pipeline)
    print(f"  [OK] Explainer type: {type(explainer).__name__}")
except Exception as exc:
    print(f"  [FAIL] {exc}")
    all_ok = False

# 5. Validate CSV contents
print("\n[5] CSV Contents")
try:
    contrib_df = pd.read_csv(EXPLANATIONS_DIR / "shap_contributions.csv")
    print(f"  [OK] Global contributions: {len(contrib_df)} rows x {len(contrib_df.columns)} cols")
    print(f"       Columns: {list(contrib_df.columns)}")
except Exception as exc:
    print(f"  [FAIL] {exc}")
    all_ok = False

try:
    local_df = pd.read_csv(EXPLANATIONS_DIR / "shap_local_contributions.csv")
    print(f"  [OK] Local contributions: {len(local_df)} rows x {len(local_df.columns)} cols")
    print(f"       Columns: {list(local_df.columns)}")
except Exception as exc:
    print(f"  [FAIL] {exc}")
    all_ok = False

print("\n" + "=" * 70)
if all_ok:
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED")
print("=" * 70)
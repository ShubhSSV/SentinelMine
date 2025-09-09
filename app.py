# app.py
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import os
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# Config
# -------------------------------
DATA_CSV = Path("landslide_dataset.csv")  # must be in repo root
TARGET_COL = "Landslide"                  # dataset target name
MODEL_DIR = Path("model_out")
MODEL_PIPELINE_PATH = MODEL_DIR / "model_pipeline.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

# Blend factor between real model and demo-derived score.
# 0.0 -> only demo_score (fully synthetic)
# 1.0 -> only model_proba (fully model-driven)
BLEND_ALPHA = float(os.environ.get("BLEND_ALPHA", 0.6))  # default 0.6

# Alert thresholds
THRESHOLD_GREEN = 0.30
THRESHOLD_YELLOW = 0.70

# -------------------------------
# If no model exists, train it
# -------------------------------
if not MODEL_PIPELINE_PATH.exists():
    if DATA_CSV.exists():
        # import train.main and run training (train.py should define main(csv_path, target, model_out_dir))
        from train import main
        print("⚡ No trained model found, training now...")
        main(str(DATA_CSV), TARGET_COL, model_out_dir=str(MODEL_DIR))
    else:
        print("❗ No model and no dataset found. Please add landslide_dataset.csv and redeploy.")
        # continue - the app will still run but prediction will use demo-only method

# -------------------------------
# Try loading model pipeline; fallback to retrain if load fails
# -------------------------------
pipeline = None
if MODEL_PIPELINE_PATH.exists():
    try:
        pipeline = joblib.load(MODEL_PIPELINE_PATH)
        print("✅ Loaded model pipeline.")
    except Exception as e:
        print("⚠️ Could not load model pickle, retraining due to:", e)
        if DATA_CSV.exists():
            from train import main
            main(str(DATA_CSV), TARGET_COL, model_out_dir=str(MODEL_DIR))
            pipeline = joblib.load(MODEL_PIPELINE_PATH)
            print("✅ Retrained and loaded new pipeline.")
        else:
            print("❗ Retrain failed because dataset missing. Pipeline unavailable; demo-only mode.")

# -------------------------------
# Load metadata (if available)
# -------------------------------
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    numeric_feats = metadata.get("numeric_features", [])
    categorical_feats = metadata.get("categorical_features", [])
    feature_values = metadata.get("feature_example_values", {})
else:
    numeric_feats = []
    categorical_feats = []
    feature_values = {}

# -------------------------------
# Auto-compute demo weights from dataset correlations
# -------------------------------
feature_weights = {}   # normalized absolute-correlation weights (sum to 1)
feature_sign = {}      # +1 or -1 indicating direction
minmax_map = {}        # {feature: {'min':..., 'max':...}}

if DATA_CSV.exists():
    try:
        df_raw = pd.read_csv(DATA_CSV)
        if TARGET_COL in df_raw.columns:
            # Convert target to numeric
            df_raw[TARGET_COL] = pd.to_numeric(df_raw[TARGET_COL], errors='coerce')
            # Only consider numeric features present in df_raw
            corr_map = {}
            for f in numeric_feats:
                if f in df_raw.columns:
                    s = pd.to_numeric(df_raw[f], errors='coerce')
                    minmax_map[f] = {
                        "min": float(np.nanmin(s)) if s.notna().any() else 0.0,
                        "max": float(np.nanmax(s)) if s.notna().any() else 1.0
                    }
                    try:
                        corr = float(s.corr(df_raw[TARGET_COL]))
                    except Exception:
                        corr = 0.0
                    if np.isnan(corr):
                        corr = 0.0
                    corr_map[f] = corr
                else:
                    # Feature missing from csv -> fallback defaults
                    corr_map[f] = 0.0
                    minmax_map[f] = {"min": 0.0, "max": 1.0}

            # If all correlations zero, fall back to uniform weights
            abs_corrs = {f: abs(v) for f, v in corr_map.items()}
            total = sum(abs_corrs.values())
            if total <= 0:
                # uniform
                n = len(numeric_feats) if numeric_feats else 1
                for f in numeric_feats:
                    feature_weights[f] = 1.0 / n
                    feature_sign[f] = 1
            else:
                for f, v in abs_corrs.items():
                    feature_weights[f] = v / total
                    feature_sign[f] = 1 if corr_map[f] >= 0 else -1
            print("✅ Derived demo weights from correlations:", feature_weights)
        else:
            # Target not present - fallback to uniform
            n = len(numeric_feats) if numeric_feats else 1
            for f in numeric_feats:
                feature_weights[f] = 1.0 / n
                feature_sign[f] = 1
                minmax_map[f] = {"min": 0.0, "max": 1.0}
            print("⚠️ Target column not in CSV. Using uniform demo weights.")
    except Exception as ex:
        print("⚠️ Could not compute correlations from dataset:", ex)
        # fallback uniform
        n = len(numeric_feats) if numeric_feats else 1
        for f in numeric_feats:
            feature_weights[f] = 1.0 / n
            feature_sign[f] = 1
            minmax_map[f] = {"min": 0.0, "max": 1.0}
else:
    # No CSV: fallback uniform weights and defaults
    n = len(numeric_feats) if numeric_feats else 1
    for f in numeric_feats:
        feature_weights[f] = 1.0 / n
        feature_sign[f] = 1
        minmax_map[f] = {"min": 0.0, "max": 1.0}
    print("⚠️ Dataset not found. Using uniform demo weights and default ranges.")

# For numeric features not present in minmax_map, set default
for f in numeric_feats:
    if f not in minmax_map:
        minmax_map[f] = {"min": 0.0, "max": 1.0}
    if f not in feature_weights:
        feature_weights[f] = 0.0
    if f not in feature_sign:
        feature_sign[f] = 1

# -------------------------------
# Flask App
# -------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    # Build descriptors for UI (used by index.html)
    feature_descriptors = []
    for f in numeric_feats:
        mm = minmax_map.get(f, {"min": 0.0, "max": 1.0, "mean": 0.0})
        feature_descriptors.append({
            "name": f,
            "type": "numeric",
            "min_r": round(mm.get("min", 0.0), 6),
            "max_r": round(mm.get("max", 1.0), 6),
            "mean_r": round((mm.get("min", 0.0) + mm.get("max", 1.0)) / 2.0, 6)
        })
    for f in categorical_feats:
        vals = feature_values.get(f, [])
        feature_descriptors.append({
            "name": f,
            "type": "categorical",
            "options": vals if isinstance(vals, list) else list(vals)
        })
    return render_template("index.html", features=feature_descriptors)


def compute_demo_score(input_values: dict) -> float:
    """
    Compute a demo score in [0,1] using normalized feature values and correlation-derived weights.
    If a feature has negative correlation, its effect is inverted (more value => less risk).
    """
    score = 0.0
    for f, w in feature_weights.items():
        # Skip zero-weight features
        if w <= 0:
            continue
        raw_val = input_values.get(f, 0.0)
        try:
            val = float(raw_val)
        except Exception:
            val = 0.0
        mm = minmax_map.get(f, {"min": 0.0, "max": 1.0})
        minv = mm["min"]
        maxv = mm["max"]
        # normalize safely
        if maxv > minv:
            val_norm = (val - minv) / (maxv - minv)
        else:
            val_norm = 0.0
        val_norm = float(max(0.0, min(1.0, val_norm)))
        if feature_sign.get(f, 1) < 0:
            # negative correlation => high value reduces risk
            contrib = (1.0 - val_norm) * w
        else:
            contrib = val_norm * w
        score += contrib
    # score should already be in [0,1] if weights sum to 1
    score = float(max(0.0, min(1.0, score)))
    return score


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    # Build a single-row DataFrame for the model (if available)
    df_row = pd.DataFrame([data])

    # Model probability (if pipeline present)
    model_proba = None
    if pipeline is not None:
        try:
            if hasattr(pipeline, "predict_proba"):
                model_proba = float(pipeline.predict_proba(df_row)[:, 1][0])
            else:
                model_proba = float(pipeline.predict(df_row)[0])
        except Exception as e:
            # If model prediction fails for any reason, fall back to None
            print("⚠️ Model prediction failed:", e)
            model_proba = None

    # Demo-derived probability from correlations and normalized slider values
    demo_proba = compute_demo_score(data)

    # Blend model and demo probabilities
    if model_proba is None:
        final_proba = demo_proba
    else:
        alpha = BLEND_ALPHA
        final_proba = float(max(0.0, min(1.0, alpha * model_proba + (1.0 - alpha) * demo_proba)))

    # Decide alert label
    if final_proba < THRESHOLD_GREEN:
        alert = "GREEN"
        message = "No immediate risk"
    elif final_proba < THRESHOLD_YELLOW:
        alert = "YELLOW"
        message = "Potential risk — exercise caution"
    else:
        alert = "RED"
        message = "High risk — evacuate immediately"

    # Return rich debug info too (helpful for tuning)
    debug = {
        "model_proba": None if model_proba is None else float(model_proba),
        "demo_proba": float(demo_proba),
        "final_proba": float(final_proba),
        "weights": feature_weights,
    }

    return jsonify({
        "probability": float(final_proba),
        "alert": alert,
        "message": message,
        "debug": debug
    })


# -------------------------------
# Run (Render-compatible)
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Flask dev server is OK for demo; do not use in production
    app.run(host="0.0.0.0", port=port, debug=False)



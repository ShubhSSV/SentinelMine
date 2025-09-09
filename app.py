#JAI SHREE GANESH
# app.py (updated: better demo weighting, slider scaling, damped weights, adaptive blending)
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

# Blend factor between real model and demo-derived score (base).
BLEND_ALPHA = float(os.environ.get("BLEND_ALPHA", 0.6))  # default 0.6

# Minimum model weight when model is uncertain
BLEND_ALPHA_MIN = float(os.environ.get("BLEND_ALPHA_MIN", 0.20))

# damping & variance influence hyperparameters (tunable)
WEIGHT_GAMMA = float(os.environ.get("WEIGHT_GAMMA", 0.6))   # gamma < 1 reduces dominance
VAR_BETA = float(os.environ.get("VAR_BETA", 0.35))         # influence of normalized variance

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
# Auto-compute demo weights from dataset correlations + variance (improved)
# -------------------------------
feature_weights = {}   # normalized weights
feature_sign = {}      # +1 or -1 indicating direction
minmax_map = {}        # {feature: {'min':..., 'max':..., 'mean':...}}
_feature_var = {}      # raw variance map (for internal use)

if DATA_CSV.exists():
    try:
        df_raw = pd.read_csv(DATA_CSV)
        if TARGET_COL in df_raw.columns:
            df_raw[TARGET_COL] = pd.to_numeric(df_raw[TARGET_COL], errors='coerce')

            corr_map = {}
            var_map = {}
            # compute stats for each numeric feature
            for f in numeric_feats:
                if f in df_raw.columns:
                    s = pd.to_numeric(df_raw[f], errors='coerce')
                    # store min/max/mean
                    try:
                        mn = float(np.nanmin(s)) if s.notna().any() else 0.0
                        mx = float(np.nanmax(s)) if s.notna().any() else 1.0
                        meanv = float(np.nanmean(s)) if s.notna().any() else 0.0
                    except Exception:
                        mn, mx, meanv = 0.0, 1.0, 0.0
                    minmax_map[f] = {"min": mn, "max": mx, "mean": meanv}
                    # variance (for importance modulation)
                    try:
                        v = float(np.nanvar(s)) if s.notna().any() else 0.0
                    except Exception:
                        v = 0.0
                    var_map[f] = v
                    _feature_var[f] = v
                    # correlation with target
                    try:
                        corr = float(s.corr(df_raw[TARGET_COL]))
                    except Exception:
                        corr = 0.0
                    if np.isnan(corr):
                        corr = 0.0
                    corr_map[f] = corr
                else:
                    # defaults
                    minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}
                    corr_map[f] = 0.0
                    var_map[f] = 0.0
                    _feature_var[f] = 0.0

            # normalize variance to 0..1
            max_var = max(var_map.values()) if len(var_map) > 0 else 0.0
            var_norm = {f: (var_map[f] / max_var) if max_var > 0 else 0.0 for f in var_map}

            # combined importance: |corr| * (1 + VAR_BETA * var_norm)
            combined = {}
            for f in numeric_feats:
                combined_val = abs(corr_map.get(f, 0.0)) * (1.0 + VAR_BETA * var_norm.get(f, 0.0))
                combined[f] = combined_val

            # apply gamma damping to reduce dominance of top features
            # then normalize the weights
            gamma = WEIGHT_GAMMA if WEIGHT_GAMMA > 0 else 1.0
            damped = {f: (combined[f] ** gamma) for f in combined}
            total = sum(damped.values())
            if total <= 0:
                n = len(numeric_feats) if numeric_feats else 1
                for f in numeric_feats:
                    feature_weights[f] = 1.0 / n
                    feature_sign[f] = 1
            else:
                for f, v in damped.items():
                    feature_weights[f] = v / total
                    feature_sign[f] = 1 if corr_map.get(f, 0.0) >= 0 else -1

            print("✅ Derived improved demo weights:", feature_weights)
        else:
            # fallback uniform
            n = len(numeric_feats) if numeric_feats else 1
            for f in numeric_feats:
                feature_weights[f] = 1.0 / n
                feature_sign[f] = 1
                minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}
            print("⚠️ Target column not in CSV. Using uniform demo weights.")
    except Exception as ex:
        print("⚠️ Could not compute correlations from dataset:", ex)
        # fallback uniform
        n = len(numeric_feats) if numeric_feats else 1
        for f in numeric_feats:
            feature_weights[f] = 1.0 / n
            feature_sign[f] = 1
            minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}
else:
    # No CSV: fallback uniform weights and defaults
    n = len(numeric_feats) if numeric_feats else 1
    for f in numeric_feats:
        feature_weights[f] = 1.0 / n
        feature_sign[f] = 1
        minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}
    print("⚠️ Dataset not found. Using uniform demo weights and default ranges.")

# ensure defaults exist
for f in numeric_feats:
    if f not in minmax_map:
        minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}
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


def _interpret_slider_value(raw_val, f):
    """
    Interpret incoming slider value consistently:
    - If frontend uses 0..100 while real min/max != 0..100, treat as percentage of real range.
    - Otherwise assume value is already in real units.
    """
    try:
        val = float(raw_val)
    except Exception:
        return 0.0
    mm = minmax_map.get(f, {"min": 0.0, "max": 1.0})
    minv, maxv = mm["min"], mm["max"]
    # If slider 0..100 but actual range not 0..100, treat as percent
    if 0.0 <= val <= 100.0 and (maxv - minv) != 100.0:
        return minv + (val / 100.0) * (maxv - minv)
    return val


def compute_demo_score(input_values: dict) -> float:
    """
    Compute a demo score in [0,1] using normalized feature values and improved weights.
    Uses interpreted slider values, feature_sign, and the damped normalized weights.
    """
    score = 0.0
    for f, w in feature_weights.items():
        if w <= 0:
            continue
        raw_val = input_values.get(f, 0.0)
        val_actual = _interpret_slider_value(raw_val, f)
        mm = minmax_map.get(f, {"min": 0.0, "max": 1.0})
        minv = mm["min"]
        maxv = mm["max"]
        if maxv > minv:
            val_norm = (val_actual - minv) / (maxv - minv)
        else:
            val_norm = 0.0
        val_norm = float(max(0.0, min(1.0, val_norm)))
        if feature_sign.get(f, 1) < 0:
            contrib = (1.0 - val_norm) * w
        else:
            contrib = val_norm * w
        score += contrib
    score = float(max(0.0, min(1.0, score)))
    return score


def _adaptive_alpha(model_proba, base_alpha):
    """
    Adaptive alpha: reduce model weight when model is uncertain (near 0.5).
    - model_proba in [0,1]
    - base_alpha the configured BLEND_ALPHA
    returns effective alpha in [BLEND_ALPHA_MIN, base_alpha]
    """
    if model_proba is None:
        return 0.0
    conf = abs(model_proba - 0.5) * 2.0  # 0 when p=0.5, 1 when p=0 or 1
    # scale alpha by confidence
    alpha_eff = max(BLEND_ALPHA_MIN, base_alpha * conf)
    return float(max(0.0, min(1.0, alpha_eff)))


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    # Build a single-row DataFrame for the model (if available)
    # Note: df_row uses raw incoming values — the model pipeline expects the same units used during training.
    df_row = pd.DataFrame([{
        f: _interpret_slider_value(data.get(f, 0.0), f) for f in numeric_feats
    }])

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

    # Blend model and demo probabilities with adaptive alpha
    if model_proba is None:
        final_proba = demo_proba
    else:
        alpha_eff = _adaptive_alpha(model_proba, BLEND_ALPHA)
        final_proba = float(max(0.0, min(1.0, alpha_eff * model_proba + (1.0 - alpha_eff) * demo_proba)))

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
        "feature_var": _feature_var
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

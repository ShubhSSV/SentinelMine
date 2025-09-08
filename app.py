#JAI SHREE GANESH
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
# Config (unchanged core behavior)
# -------------------------------
DATA_CSV = Path("landslide_dataset.csv")  # must be in repo root for retrain
TARGET_COL = "Landslide"
MODEL_DIR = Path("model_out")
MODEL_PIPELINE_PATH = MODEL_DIR / "model_pipeline.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

# Blend factor (environment override possible)
BLEND_ALPHA = float(os.environ.get("BLEND_ALPHA", 0.6))

# Alert thresholds
THRESHOLD_GREEN = 0.30
THRESHOLD_YELLOW = 0.70

# Default zones (can be changed)
ZONES = ["A1", "A2", "B1", "B2", "C1"]

# -------------------------------
# Train / Load model (existing logic preserved)
# -------------------------------
if not MODEL_PIPELINE_PATH.exists():
    if DATA_CSV.exists():
        from train import main
        print("⚡ No trained model found, training now...")
        main(str(DATA_CSV), TARGET_COL, model_out_dir=str(MODEL_DIR))
    else:
        print("❗ No model and no dataset found. Running in demo-only mode.")

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
# Load metadata (for UI generation)
# -------------------------------
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    numeric_feats = metadata.get("numeric_features", [])
    categorical_feats = metadata.get("categorical_features", [])
    feature_values = metadata.get("feature_example_values", {})
else:
    # keep defaults for demo if metadata missing
    numeric_feats = ["Rainfall_mm", "Slope_Angle", "Soil_Saturation",
                     "Vegetation_Cover", "Earthquake_Activity", "Proximity_to_Water"]
    categorical_feats = []
    feature_values = {}

# -------------------------------
# Demo weighting derived from CSV (unchanged)
# -------------------------------
feature_weights = {}
feature_sign = {}
minmax_map = {}

if DATA_CSV.exists():
    try:
        df_raw = pd.read_csv(DATA_CSV)
        if TARGET_COL in df_raw.columns:
            df_raw[TARGET_COL] = pd.to_numeric(df_raw[TARGET_COL], errors='coerce')
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
                    corr_map[f] = 0.0
                    minmax_map[f] = {"min": 0.0, "max": 1.0}
            abs_corrs = {f: abs(v) for f, v in corr_map.items()}
            total = sum(abs_corrs.values())
            if total <= 0:
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
            n = len(numeric_feats) if numeric_feats else 1
            for f in numeric_feats:
                feature_weights[f] = 1.0 / n
                feature_sign[f] = 1
                minmax_map[f] = {"min": 0.0, "max": 1.0}
            print("⚠️ Target column not in CSV. Using uniform demo weights.")
    except Exception as ex:
        print("⚠️ Could not compute correlations:", ex)
        n = len(numeric_feats) if numeric_feats else 1
        for f in numeric_feats:
            feature_weights[f] = 1.0 / n
            feature_sign[f] = 1
            minmax_map[f] = {"min": 0.0, "max": 1.0}
else:
    n = len(numeric_feats) if numeric_feats else 1
    for f in numeric_feats:
        feature_weights[f] = 1.0 / n
        feature_sign[f] = 1
        minmax_map[f] = {"min": 0.0, "max": 1.0}
    print("⚠️ Dataset not found. Using uniform demo weights and default ranges.")

for f in numeric_feats:
    if f not in minmax_map:
        minmax_map[f] = {"min": 0.0, "max": 1.0}
    if f not in feature_weights:
        feature_weights[f] = 0.0
    if f not in feature_sign:
        feature_sign[f] = 1

# -------------------------------
# Flask App (same app name)
# -------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    # Build feature descriptors for UI (keeps previous behavior)
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

    # pass zones to template (new but non-destructive)
    return render_template("index.html", features=feature_descriptors, zones=ZONES)


def compute_demo_score(input_values: dict) -> float:
    score = 0.0
    for f, w in feature_weights.items():
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
        if maxv > minv:
            val_norm = (val - minv) / (maxv - minv)
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


def _predict_single_row(input_dict: dict):
    """
    Keep existing single-payload behaviour. Returns same structure as before.
    """
    df_row = pd.DataFrame([input_dict])
    model_proba = None
    if pipeline is not None:
        try:
            if hasattr(pipeline, "predict_proba"):
                model_proba = float(pipeline.predict_proba(df_row)[:, 1][0])
            else:
                model_proba = float(pipeline.predict(df_row)[0])
        except Exception as e:
            print("⚠️ Model prediction failed:", e)
            model_proba = None

    demo_proba = compute_demo_score(input_dict)

    if model_proba is None:
        final_proba = demo_proba
    else:
        alpha = BLEND_ALPHA
        final_proba = float(max(0.0, min(1.0, alpha * model_proba + (1.0 - alpha) * demo_proba)))

    if final_proba < THRESHOLD_GREEN:
        alert = "GREEN"; message = "No immediate risk"
    elif final_proba < THRESHOLD_YELLOW:
        alert = "YELLOW"; message = "Potential risk — exercise caution"
    else:
        alert = "RED"; message = "High risk — evacuate immediately"

    debug = {
        "model_proba": None if model_proba is None else float(model_proba),
        "demo_proba": float(demo_proba),
        "final_proba": float(final_proba),
        "weights": feature_weights,
    }

    return {
        "probability": float(final_proba),
        "alert": alert,
        "message": message,
        "debug": debug
    }


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}

    # Detect zone-mode: if any value of data is a dict -> zone payload
    is_zone_mode = False
    # data might be {} if client sends empty; handle that as single-row empty (old behaviour)
    if isinstance(data, dict) and len(data) > 0:
        # find first value
        first_val = next(iter(data.values()))
        if isinstance(first_val, dict):
            is_zone_mode = True

    if not is_zone_mode:
        # preserve old behavior (global single-sensor input)
        result = _predict_single_row(data)
        return jsonify(result)

    # -------------------------------
    # Zone mode: iterate zones and compute per-zone results
    # -------------------------------
    results = {}
    max_proba = -1.0
    max_zone = None
    for zone, readings in data.items():
        if not isinstance(readings, dict):
            # skip invalid zone payloads gracefully
            continue

        # compute model probability for this zone (if pipeline available)
        model_proba = None
        df_row = pd.DataFrame([readings])
        if pipeline is not None:
            try:
                if hasattr(pipeline, "predict_proba"):
                    model_proba = float(pipeline.predict_proba(df_row)[:, 1][0])
                else:
                    model_proba = float(pipeline.predict(df_row)[0])
            except Exception as e:
                # pipeline may fail because missing columns — gracefully fallback
                print(f"⚠️ Model prediction for zone {zone} failed:", e)
                model_proba = None

        demo_proba = compute_demo_score(readings)

        if model_proba is None:
            final_proba = demo_proba
        else:
            alpha = BLEND_ALPHA
            final_proba = float(max(0.0, min(1.0, alpha * model_proba + (1.0 - alpha) * demo_proba)))

        if final_proba < THRESHOLD_GREEN:
            alert = "GREEN"; message = f"Zone {zone}: सुरक्षित क्षेत्र (Safe Zone)"
        elif final_proba < THRESHOLD_YELLOW:
            alert = "YELLOW"; message = f"Zone {zone}: सावधान रहें (Potential Risk)"
        else:
            alert = "RED"; message = f"Zone {zone}: तुरंत बाहर निकलें! (Evacuate Immediately!)"

        results[zone] = {
            "probability": float(final_proba),
            "alert": alert,
            "message": message,
            "sms_preview": f"Mine Alert: {alert} in Zone {zone}. {message}",
            "debug": {"model_proba": None if model_proba is None else float(model_proba),
                      "demo_proba": float(demo_proba)}
        }

        if final_proba > max_proba:
            max_proba = final_proba
            max_zone = zone

    # also include overall worst-zone summary
    summary = {"worst_zone": max_zone, "worst_prob": max_proba}
    return jsonify({"zones": results, "summary": summary})


# -------------------------------
# Run (Render-compatible)
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
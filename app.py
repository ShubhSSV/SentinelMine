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
# Config
# -------------------------------
DATA_CSV = Path("landslide_dataset.csv")  # global model dataset (if present)
ZONE_CSV = Path("synthetic_mine_sensors_zones_6000.csv")  # zone dataset (if present)
TARGET_COL = "Landslide"
MODEL_DIR = Path("model_out")
MODEL_PIPELINE_PATH = MODEL_DIR / "model_pipeline.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

ZONE_MODEL_DIR = Path("model_out_zone")
ZONE_MODEL_PATH = ZONE_MODEL_DIR / "model_pipeline_zone.pkl"
ZONE_METADATA_PATH = ZONE_MODEL_DIR / "metadata_zone.json"

# Blend factor between model and demo score (0.0 -> demo only, 1.0 -> model only)
BLEND_ALPHA = float(os.environ.get("BLEND_ALPHA", 0.6))

THRESHOLD_GREEN = 0.30
THRESHOLD_YELLOW = 0.70

# Default zone list if CSV not present
DEFAULT_ZONES = ["A1", "A2", "B1", "B2", "C1", "C2"]

# -------------------------------
# If no global model exists, attempt to train (keeps original behaviour)
# -------------------------------
if not MODEL_PIPELINE_PATH.exists():
    if DATA_CSV.exists():
        try:
            from train import main as train_main
            print("⚡ No trained global model found, training now...")
            train_main(str(DATA_CSV), TARGET_COL, model_out_dir=str(MODEL_DIR))
        except Exception as e:
            print("⚠️ Failed to auto-train global model:", e)
    else:
        print("❗ No global model and no dataset found. Running in demo-only mode for global predictions.")

# -------------------------------
# Load global model pipeline (if exists)
# -------------------------------
pipeline = None
if MODEL_PIPELINE_PATH.exists():
    try:
        pipeline = joblib.load(MODEL_PIPELINE_PATH)
        print("✅ Loaded global model pipeline.")
    except Exception as e:
        print("⚠️ Could not load global model pipeline:", e)
        pipeline = None

# -------------------------------
# Load zone model if present
# -------------------------------
zone_pipeline = None
if ZONE_MODEL_PATH.exists():
    try:
        zone_pipeline = joblib.load(ZONE_MODEL_PATH)
        print("✅ Loaded zone model pipeline.")
    except Exception as e:
        print("⚠️ Could not load zone model pipeline:", e)
        zone_pipeline = None

# Optionally: auto-train zone model on first deploy if CSV exists and zone model missing.
# Uncomment if you want automatic training on startup (careful: increases startup time).
if not ZONE_MODEL_PATH.exists() and ZONE_CSV.exists():
    try:
        print("⚡ Zone model missing and zone CSV present — training zone model now (this may take time)...")
        # local import: expects train_zone.py in repo root
        from train_zone import main as train_zone_main
        train_zone_main(str(ZONE_CSV), None, out_dir=str(ZONE_MODEL_DIR))
        if ZONE_MODEL_PATH.exists():
            zone_pipeline = joblib.load(ZONE_MODEL_PATH)
            print("✅ Zone model trained and loaded.")
    except Exception as e:
        print("⚠️ Auto-training zone model failed:", e)

# -------------------------------
# Load metadata for global UI (if present)
# -------------------------------
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    numeric_feats = metadata.get("numeric_features", [])
    categorical_feats = metadata.get("categorical_features", [])
    feature_values = metadata.get("feature_example_values", {})
else:
    # sensible defaults if metadata missing
    numeric_feats = ["Rainfall_mm", "Slope_Angle", "Soil_Saturation",
                     "Vegetation_Cover", "Earthquake_Activity", "Proximity_to_Water"]
    categorical_feats = []
    feature_values = {}

# -------------------------------
# Improved weight derivation (correlation + variance) and minmax_map
# -------------------------------
feature_weights = {}
feature_sign = {}
minmax_map = {}

def safe_stats_series(s):
    s = pd.to_numeric(s, errors='coerce')
    return {
        "min": float(np.nanmin(s)) if s.notna().any() else 0.0,
        "max": float(np.nanmax(s)) if s.notna().any() else 1.0,
        "var": float(np.nanvar(s)) if s.notna().any() else 0.0,
        "mean": float(np.nanmean(s)) if s.notna().any() else 0.0,
    }

if DATA_CSV.exists():
    try:
        df_raw = pd.read_csv(DATA_CSV)
        if TARGET_COL in df_raw.columns:
            df_raw[TARGET_COL] = pd.to_numeric(df_raw[TARGET_COL], errors='coerce')
            corr_map = {}
            var_map = {}
            for f in numeric_feats:
                if f in df_raw.columns:
                    s = pd.to_numeric(df_raw[f], errors='coerce')
                    stats = safe_stats_series(s)
                    minmax_map[f] = {"min": stats["min"], "max": stats["max"], "mean": stats["mean"]}
                    var_map[f] = stats["var"]
                    try:
                        corr = float(s.corr(df_raw[TARGET_COL]))
                    except Exception:
                        corr = 0.0
                    if np.isnan(corr):
                        corr = 0.0
                    corr_map[f] = corr
                else:
                    corr_map[f] = 0.0
                    var_map[f] = 0.0
                    minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}

            abs_corrs = {f: abs(v) for f, v in corr_map.items()}
            total_var = sum(var_map.values())

            combined = {}
            for f in numeric_feats:
                corr_term = abs_corrs.get(f, 0.0)
                var_term = (var_map.get(f, 0.0) / total_var) if total_var > 0 else 0.0
                if corr_term > 0 and var_term > 0:
                    combined_val = 0.7 * corr_term + 0.3 * var_term
                else:
                    combined_val = corr_term + 0.01 * var_term
                combined[f] = combined_val

            tot = sum(combined.values())
            if tot <= 0:
                n = len(numeric_feats) if numeric_feats else 1
                for f in numeric_feats:
                    feature_weights[f] = 1.0 / n
                    feature_sign[f] = 1 if corr_map.get(f, 0.0) >= 0 else -1
            else:
                for f, v in combined.items():
                    feature_weights[f] = v / tot
                    feature_sign[f] = 1 if corr_map.get(f, 0.0) >= 0 else -1

            print("✅ Derived improved demo weights:", feature_weights)
        else:
            n = len(numeric_feats) if numeric_feats else 1
            for f in numeric_feats:
                feature_weights[f] = 1.0 / n
                feature_sign[f] = 1
                minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}
            print("⚠️ Target column not in CSV. Using uniform demo weights.")
    except Exception as ex:
        print("⚠️ Could not compute improved correlations/variance:", ex)
        n = len(numeric_feats) if numeric_feats else 1
        for f in numeric_feats:
            feature_weights[f] = 1.0 / n
            feature_sign[f] = 1
            minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}
else:
    n = len(numeric_feats) if numeric_feats else 1
    for f in numeric_feats:
        feature_weights[f] = 1.0 / n
        feature_sign[f] = 1
        minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}
    print("⚠️ Dataset not found. Using uniform demo weights and default ranges.")

for f in numeric_feats:
    if f not in minmax_map:
        minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}
    if f not in feature_weights:
        feature_weights[f] = 0.0
    if f not in feature_sign:
        feature_sign[f] = 1

# -------------------------------
# Demo scoring (unified)
# -------------------------------
def compute_demo_score(input_values: dict) -> float:
    """
    Compute demo score in [0,1] using normalized values and improved weights.
    Interprets frontend slider 0-100 as percent of min->max if needed.
    """
    score = 0.0
    for f, w in feature_weights.items():
        if w <= 0:
            continue
        raw_val = input_values.get(f, None)
        if raw_val is None:
            val = 0.0
        else:
            try:
                val = float(raw_val)
            except Exception:
                val = 0.0
        mm = minmax_map.get(f, {"min": 0.0, "max": 1.0})
        minv, maxv = mm["min"], mm["max"]
        # Treat slider 0..100 as percentage if appropriate
        if 0 <= val <= 100 and (maxv - minv) != 100:
            val_actual = minv + (val / 100.0) * (maxv - minv)
        else:
            val_actual = val
        val_norm = 0.0
        if maxv > minv:
            val_norm = (val_actual - minv) / (maxv - minv)
        val_norm = float(max(0.0, min(1.0, val_norm)))
        if feature_sign.get(f, 1) < 0:
            contrib = (1.0 - val_norm) * w
        else:
            contrib = val_norm * w
        score += contrib
    score = float(max(0.0, min(1.0, score)))
    return score

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

@app.route("/zones")
def zones():
    """Return available zones and feature list for frontend to build zone UI dynamically."""
    zones = []
    if ZONE_CSV.exists():
        try:
            df = pd.read_csv(ZONE_CSV)
            if "Zone" in df.columns:
                zones = sorted(df["Zone"].dropna().unique().tolist())
        except Exception as e:
            print("⚠️ Could not read zone CSV to enumerate zones:", e)
    if not zones:
        zones = DEFAULT_ZONES
    return jsonify({"zones": zones, "features": [f for f in numeric_feats]})

# -------------------------------
# Existing prediction behavior preserved + zone-mode handling
# -------------------------------
def _predict_single_row(input_dict: dict):
    df_row = pd.DataFrame([input_dict])
    model_proba = None
    if pipeline is not None:
        try:
            if hasattr(pipeline, "predict_proba"):
                model_proba = float(pipeline.predict_proba(df_row)[:, 1][0])
            else:
                model_proba = float(pipeline.predict(df_row)[0])
        except Exception as e:
            print("⚠️ Global model prediction failed:", e)
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
        "weights": feature_weights
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

    # detect zone-mode (payload with dict values per zone)
    is_zone_mode = False
    if isinstance(data, dict) and len(data) > 0:
        first_val = next(iter(data.values()))
        if isinstance(first_val, dict):
            is_zone_mode = True

    if not is_zone_mode:
        result = _predict_single_row(data)
        return jsonify(result)

    # zone-mode
    results = {}
    max_proba = -1.0
    max_zone = None
    for zone, readings in data.items():
        if not isinstance(readings, dict):
            continue

        # Try zone model first (if available)
        model_proba = None
        if zone_pipeline is not None:
            try:
                df_row = pd.DataFrame([readings])
                if hasattr(zone_pipeline, "predict_proba"):
                    model_proba = float(zone_pipeline.predict_proba(df_row)[:, 1][0])
                else:
                    model_proba = float(zone_pipeline.predict(df_row)[0])
            except Exception as e:
                print(f"⚠️ Zone model prediction failed for {zone}:", e)
                model_proba = None

        # fallback to global model if zone model not available
        if model_proba is None and pipeline is not None:
            try:
                df_row = pd.DataFrame([readings])
                if hasattr(pipeline, "predict_proba"):
                    model_proba = float(pipeline.predict_proba(df_row)[:, 1][0])
                else:
                    model_proba = float(pipeline.predict(df_row)[0])
            except Exception:
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

    summary = {"worst_zone": max_zone, "worst_prob": max_proba}
    return jsonify({"zones": results, "summary": summary})

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


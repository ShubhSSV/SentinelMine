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
DATA_CSV = Path("landslide_dataset.csv")
ZONE_CSV = Path("synthetic_mine_sensors_zones_6000.csv")
TARGET_COL = "Landslide"
MODEL_DIR = Path("model_out")
MODEL_PIPELINE_PATH = MODEL_DIR / "model_pipeline.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

ZONE_MODEL_DIR = Path("model_out_zone")
ZONE_MODEL_PATH = ZONE_MODEL_DIR / "model_pipeline_zone.pkl"
ZONE_METADATA_PATH = ZONE_MODEL_DIR / "metadata_zone.json"

GLOBAL_BLEND_ALPHA = float(os.environ.get("BLEND_ALPHA_GLOBAL", 0.6))
ZONE_BLEND_ALPHA = float(os.environ.get("BLEND_ALPHA_ZONE", 0.85))

THRESHOLD_GREEN = 0.30
THRESHOLD_YELLOW = 0.70

DEFAULT_ZONES = ["A1", "A2", "B1", "B2", "C1", "C2"]

# -------------------------------
# Auto-train global model if missing (keeps behavior)
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
# Load pipelines
# -------------------------------
pipeline = None
if MODEL_PIPELINE_PATH.exists():
    try:
        pipeline = joblib.load(MODEL_PIPELINE_PATH)
        print("✅ Loaded global model pipeline.")
    except Exception as e:
        print("⚠️ Could not load global model pipeline:", e)
        pipeline = None

zone_pipeline = None
if ZONE_MODEL_PATH.exists():
    try:
        zone_pipeline = joblib.load(ZONE_MODEL_PATH)
        print("✅ Loaded zone model pipeline.")
    except Exception as e:
        print("⚠️ Could not load zone model pipeline:", e)
        zone_pipeline = None

# optionally auto-train zone model
if not ZONE_MODEL_PATH.exists() and ZONE_CSV.exists():
    try:
        print("⚡ Zone model missing and zone CSV present — training zone model now...")
        from train_zone import main as train_zone_main
        train_zone_main(str(ZONE_CSV), None, out_dir=str(ZONE_MODEL_DIR))
        if ZONE_MODEL_PATH.exists():
            zone_pipeline = joblib.load(ZONE_MODEL_PATH)
            print("✅ Zone model trained and loaded.")
    except Exception as e:
        print("⚠️ Auto-training zone model failed:", e)

# -------------------------------
# Load metadata for UI
# -------------------------------
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    numeric_feats = metadata.get("numeric_features", [])
    categorical_feats = metadata.get("categorical_features", [])
    feature_values = metadata.get("feature_example_values", {})
else:
    numeric_feats = ["Rainfall_mm", "Slope_Angle", "Soil_Saturation",
                     "Vegetation_Cover", "Earthquake_Activity", "Proximity_to_Water"]
    categorical_feats = []
    feature_values = {}

# -------------------------------
# Restore original global weight derivation
# -------------------------------
global_feature_weights = {}
global_feature_sign = {}
global_minmax_map = {}

if DATA_CSV.exists():
    try:
        df_raw = pd.read_csv(DATA_CSV)
        if TARGET_COL in df_raw.columns:
            df_raw[TARGET_COL] = pd.to_numeric(df_raw[TARGET_COL], errors='coerce')
            corr_map = {}
            for f in numeric_feats:
                if f in df_raw.columns:
                    s = pd.to_numeric(df_raw[f], errors='coerce')
                    try:
                        minv = float(np.nanmin(s))
                        maxv = float(np.nanmax(s))
                        meanv = float(np.nanmean(s))
                    except Exception:
                        minv, maxv, meanv = 0.0, 1.0, 0.0
                    global_minmax_map[f] = {"min": minv, "max": maxv, "mean": meanv}
                    try:
                        corr = float(s.corr(df_raw[TARGET_COL]))
                    except Exception:
                        corr = 0.0
                    if np.isnan(corr):
                        corr = 0.0
                    corr_map[f] = corr
                else:
                    corr_map[f] = 0.0
                    global_minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}
            abs_corrs = {f: abs(v) for f, v in corr_map.items()}
            total = sum(abs_corrs.values())
            if total <= 0:
                n = len(numeric_feats) if numeric_feats else 1
                for f in numeric_feats:
                    global_feature_weights[f] = 1.0 / n
                    global_feature_sign[f] = 1
            else:
                for f, v in abs_corrs.items():
                    global_feature_weights[f] = v / total
                    global_feature_sign[f] = 1 if corr_map[f] >= 0 else -1
            print("✅ Restored global weights (original method).")
        else:
            n = len(numeric_feats) if numeric_feats else 1
            for f in numeric_feats:
                global_feature_weights[f] = 1.0 / n
                global_feature_sign[f] = 1
                global_minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}
            print("⚠️ Target missing; global weights uniform.")
    except Exception as ex:
        print("⚠️ Could not compute global correlations:", ex)
        n = len(numeric_feats) if numeric_feats else 1
        for f in numeric_feats:
            global_feature_weights[f] = 1.0 / n
            global_feature_sign[f] = 1
            global_minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}
else:
    n = len(numeric_feats) if numeric_feats else 1
    for f in numeric_feats:
        global_feature_weights[f] = 1.0 / n
        global_feature_sign[f] = 1
        global_minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}
    print("⚠️ No DATA_CSV: global weights uniform fallback.")

for f in numeric_feats:
    if f not in global_minmax_map:
        global_minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}
    if f not in global_feature_weights:
        global_feature_weights[f] = 0.0
    if f not in global_feature_sign:
        global_feature_sign[f] = 1

# -------------------------------
# Zone: improved weights (corr+var) and minmax
# -------------------------------
zone_feature_weights = {}
zone_feature_sign = {}
zone_minmax_map = {}

def safe_stats_series(s):
    s = pd.to_numeric(s, errors='coerce')
    return {
        "min": float(np.nanmin(s)) if s.notna().any() else 0.0,
        "max": float(np.nanmax(s)) if s.notna().any() else 1.0,
        "var": float(np.nanvar(s)) if s.notna().any() else 0.0,
        "mean": float(np.nanmean(s)) if s.notna().any() else 0.0,
    }

if ZONE_CSV.exists():
    try:
        df_z = pd.read_csv(ZONE_CSV)
        corr_map = {}
        var_map = {}
        # detect potential target in zone CSV
        possible_targets = [c for c in df_z.columns if c.lower() in ("landslide","risk","label","event_label","zone_risk")]
        for f in numeric_feats:
            if f in df_z.columns:
                s = pd.to_numeric(df_z[f], errors='coerce')
                stats = safe_stats_series(s)
                zone_minmax_map[f] = {"min": stats["min"], "max": stats["max"], "mean": stats["mean"]}
                var_map[f] = stats["var"]
                corr = 0.0
                if possible_targets:
                    try:
                        tc = pd.to_numeric(df_z[possible_targets[0]], errors='coerce')
                        corr = float(s.corr(tc))
                        if np.isnan(corr):
                            corr = 0.0
                    except Exception:
                        corr = 0.0
                corr_map[f] = corr
            else:
                zone_minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}
                corr_map[f] = 0.0
                var_map[f] = 0.0
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
                zone_feature_weights[f] = 1.0 / n
                zone_feature_sign[f] = 1
        else:
            for f, v in combined.items():
                zone_feature_weights[f] = v / tot
                zone_feature_sign[f] = 1 if corr_map.get(f, 0.0) >= 0 else -1
        # boost earthquake influence floor
        if "Earthquake_Activity" in zone_feature_weights:
            zone_feature_weights["Earthquake_Activity"] = max(zone_feature_weights.get("Earthquake_Activity", 0.0), 0.12)
            s = sum(zone_feature_weights.values())
            if s > 0:
                for k in zone_feature_weights:
                    zone_feature_weights[k] /= s
        print("✅ Derived zone weights.")
    except Exception as ex:
        print("⚠️ Could not compute zone stats:", ex)
        n = len(numeric_feats) if numeric_feats else 1
        for f in numeric_feats:
            zone_feature_weights[f] = 1.0 / n
            zone_feature_sign[f] = 1
            zone_minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}
else:
    n = len(numeric_feats) if numeric_feats else 1
    for f in numeric_feats:
        zone_feature_weights[f] = 1.0 / n
        zone_feature_sign[f] = 1
        zone_minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}
    print("⚠️ No zone CSV: using uniform zone weights.")

for f in numeric_feats:
    if f not in zone_minmax_map:
        zone_minmax_map[f] = {"min": 0.0, "max": 1.0, "mean": 0.0}
    if f not in zone_feature_weights:
        zone_feature_weights[f] = 0.0
    if f not in zone_feature_sign:
        zone_feature_sign[f] = 1

# -------------------------------
# Demo scoring functions
# -------------------------------
def compute_demo_score_global(input_values: dict) -> float:
    score = 0.0
    for f, w in global_feature_weights.items():
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
        mm = global_minmax_map.get(f, {"min": 0.0, "max": 1.0})
        minv, maxv = mm["min"], mm["max"]
        if 0 <= val <= 100 and (maxv - minv) != 100:
            val_actual = minv + (val / 100.0) * (maxv - minv)
        else:
            val_actual = val
        val_norm = 0.0
        if maxv > minv:
            val_norm = (val_actual - minv) / (maxv - minv)
        val_norm = float(max(0.0, min(1.0, val_norm)))
        if global_feature_sign.get(f, 1) < 0:
            contrib = (1.0 - val_norm) * w
        else:
            contrib = val_norm * w
        score += contrib
    return float(max(0.0, min(1.0, score)))

def compute_demo_score_zone(input_values: dict) -> float:
    score = 0.0
    for f, w in zone_feature_weights.items():
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
        mm = zone_minmax_map.get(f, {"min": 0.0, "max": 1.0})
        minv, maxv = mm["min"], mm["max"]
        if 0 <= val <= 100 and (maxv - minv) != 100:
            val_actual = minv + (val / 100.0) * (maxv - minv)
        else:
            val_actual = val
        val_norm = 0.0
        if maxv > minv:
            val_norm = (val_actual - minv) / (maxv - minv)
        val_norm = float(max(0.0, min(1.0, val_norm)))
        if zone_feature_sign.get(f, 1) < 0:
            contrib = (1.0 - val_norm) * w
        else:
            contrib = val_norm * w
        score += contrib
    return float(max(0.0, min(1.0, score)))

# -------------------------------
# Extreme-scaling helper
# -------------------------------
def scale_to_endpoints(value, low_val, high_val, out_low=0.01, out_high=0.99):
    # Map value from [low_val, high_val] -> [out_low, out_high] linearly
    if high_val <= low_val:
        return max(out_low, min(out_high, value))
    t = (value - low_val) / (high_val - low_val)
    return float(max(out_low, min(out_high, out_low + t * (out_high - out_low))))

def compute_extremes_for_row(row_builder, model_predict_fn, demo_score_fn, alpha):
    """
    row_builder(sign) -> dict mapping features to either ideal/worst values
      sign: 'ideal' or 'worst'
    model_predict_fn(row_dict) -> model_proba (or None)
    demo_score_fn(row_dict) -> demo_proba
    returns tuple (combined_proba_for_extreme)
    """
    row = row_builder()
    demo = demo_score_fn(row)
    model_p = None
    if model_predict_fn is not None:
        try:
            model_p = model_predict_fn(row)
        except Exception:
            model_p = None
    if model_p is None:
        return demo
    return float(max(0.0, min(1.0, alpha * model_p + (1.0 - alpha) * demo)))

def build_extreme_row(sign, minmax_map, feature_sign_map):
    """
    sign == 'ideal' -> choose per-feature value that reduces risk
    sign == 'worst' -> choose per-feature value that increases risk
    For features with positive direction (feature_sign=+1), ideal -> min, worst -> max
    For features with negative direction, ideal -> max, worst -> min
    """
    row = {}
    for f in numeric_feats:
        mm = minmax_map.get(f, {"min": 0.0, "max": 1.0})
        if feature_sign_map.get(f, 1) >= 0:
            if sign == 'ideal':
                row[f] = mm["min"]
            else:
                row[f] = mm["max"]
        else:
            # negative sign: higher value reduces risk
            if sign == 'ideal':
                row[f] = mm["max"]
            else:
                row[f] = mm["min"]
    return row

# -------------------------------
# Flask app
# -------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def index():
    feature_descriptors = []
    for f in numeric_feats:
        mm = global_minmax_map.get(f, {"min": 0.0, "max": 1.0, "mean": 0.0})
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
# Prediction: global and zone with endpoint-scaling to 1%..99%
# -------------------------------
def model_predict_global_row(row):
    if pipeline is None:
        return None
    try:
        df_row = pd.DataFrame([row])
        if hasattr(pipeline, "predict_proba"):
            return float(pipeline.predict_proba(df_row)[:, 1][0])
        else:
            return float(pipeline.predict(df_row)[0])
    except Exception:
        return None

def model_predict_zone_row(row):
    if zone_pipeline is not None:
        try:
            df_row = pd.DataFrame([row])
            if hasattr(zone_pipeline, "predict_proba"):
                return float(zone_pipeline.predict_proba(df_row)[:, 1][0])
            else:
                return float(zone_pipeline.predict(df_row)[0])
        except Exception:
            return None
    # fallback to global model
    return model_predict_global_row(row)

def _predict_single_row(input_dict: dict):
    # compute raw model & demo probabilities
    model_proba = None
    if pipeline is not None:
        try:
            model_proba = model_predict_global_row(input_dict)
        except Exception:
            model_proba = None
    demo_proba = compute_demo_score_global(input_dict)

    # blended
    if model_proba is None:
        blended = demo_proba
    else:
        blended = float(max(0.0, min(1.0, GLOBAL_BLEND_ALPHA * model_proba + (1.0 - GLOBAL_BLEND_ALPHA) * demo_proba)))

    # compute extremes (ideal & worst) using same logic
    def row_builder_ideal():
        return build_extreme_row('ideal', global_minmax_map, global_feature_sign)
    def row_builder_worst():
        return build_extreme_row('worst', global_minmax_map, global_feature_sign)

    ideal_combined = compute_extremes_for_row(row_builder_ideal, model_predict_global_row if pipeline is not None else None, compute_demo_score_global, GLOBAL_BLEND_ALPHA)
    worst_combined = compute_extremes_for_row(row_builder_worst, model_predict_global_row if pipeline is not None else None, compute_demo_score_global, GLOBAL_BLEND_ALPHA)

    # scale blended to [0.01,0.99] using these extremes
    scaled = scale_to_endpoints(blended, ideal_combined, worst_combined, out_low=0.01, out_high=0.99)

    final_proba = float(max(0.01, min(0.99, scaled)))

    if final_proba < THRESHOLD_GREEN:
        alert = "GREEN"; message = "No immediate risk"
    elif final_proba < THRESHOLD_YELLOW:
        alert = "YELLOW"; message = "Potential risk — exercise caution"
    else:
        alert = "RED"; message = "High risk — evacuate immediately"

    debug = {
        "model_proba": None if model_proba is None else float(model_proba),
        "demo_proba": float(demo_proba),
        "blended_raw": float(blended),
        "ideal_extreme": float(ideal_combined),
        "worst_extreme": float(worst_combined),
        "final_proba": float(final_proba)
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

    # detect zone-mode
    is_zone_mode = False
    if isinstance(data, dict) and len(data) > 0:
        first_val = next(iter(data.values()))
        if isinstance(first_val, dict):
            is_zone_mode = True

    if not is_zone_mode:
        return jsonify(_predict_single_row(data))

    # zone-mode
    results = {}
    max_proba = -1.0
    max_zone = None

    # precompute zone extremes using zone maps (if zone model exists prefer it)
    def zone_model_fn(row):
        return model_predict_zone_row(row)

    for zone, readings in data.items():
        if not isinstance(readings, dict):
            continue

        # model prob (prefer zone_pipeline)
        model_p = None
        try:
            model_p = model_predict_zone_row(readings)
        except Exception:
            model_p = None

        demo_p = compute_demo_score_zone(readings)

        if model_p is None:
            blended = demo_p
        else:
            blended = float(max(0.0, min(1.0, ZONE_BLEND_ALPHA * model_p + (1.0 - ZONE_BLEND_ALPHA) * demo_p)))

        # build extremes for zone using zone_minmax_map & zone_feature_sign
        ideal_row = build_extreme_row('ideal', zone_minmax_map, zone_feature_sign)
        worst_row = build_extreme_row('worst', zone_minmax_map, zone_feature_sign)

        ideal_combined = compute_extremes_for_row(lambda: ideal_row, zone_model_fn if zone_pipeline is not None or pipeline is not None else None, compute_demo_score_zone, ZONE_BLEND_ALPHA)
        worst_combined = compute_extremes_for_row(lambda: worst_row, zone_model_fn if zone_pipeline is not None or pipeline is not None else None, compute_demo_score_zone, ZONE_BLEND_ALPHA)

        scaled = scale_to_endpoints(blended, ideal_combined, worst_combined, out_low=0.01, out_high=0.99)
        final_proba = float(max(0.01, min(0.99, scaled)))

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
            "debug": {"model_proba": None if model_p is None else float(model_p),
                      "demo_proba": float(demo_p),
                      "blended_raw": float(blended),
                      "ideal_extreme": float(ideal_combined),
                      "worst_extreme": float(worst_combined)}
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
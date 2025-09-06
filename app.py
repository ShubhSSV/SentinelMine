#JAI SHREE GANESH
import json
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
import os

# -------------------------------
# Auto-train if model missing
# -------------------------------
MODEL_DIR = Path("model_out")
MODEL_PIPELINE_PATH = MODEL_DIR / "model_pipeline.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

if not MODEL_PIPELINE_PATH.exists():
    from train import main
    print("⚡ No trained model found, training now...")
    main("landslide_dataset.csv", "Landslide", model_out_dir="model_out")

# -------------------------------
# Load Model + Metadata with Fallback
# -------------------------------
try:
    pipeline = joblib.load(MODEL_PIPELINE_PATH)
except Exception as e:
    print("⚠️ Could not load model pickle, retraining due to:", e)
    from train import main
    main("landslide_dataset.csv", "Landslide", model_out_dir="model_out")
    pipeline = joblib.load(MODEL_PIPELINE_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

numeric_feats = metadata.get("numeric_features", [])
categorical_feats = metadata.get("categorical_features", [])
feature_values = metadata.get("feature_example_values", {})

THRESHOLD_GREEN = 0.30
THRESHOLD_YELLOW = 0.70

# -------------------------------
# Flask App
# -------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def index():
    feature_descriptors = []
    for f in numeric_feats:
        vals = feature_values.get(f, {})
        feature_descriptors.append({
            "name": f,
            "type": "numeric",
            "min_r": round(vals.get("min", 0.0), 3),
            "max_r": round(vals.get("max", 100.0), 3),
            "mean_r": round(vals.get("mean", 0.0), 3),
        })
    for f in categorical_feats:
        vals = feature_values.get(f, [])
        feature_descriptors.append({
            "name": f,
            "type": "categorical",
            "options": vals if isinstance(vals, list) else list(vals),
        })
    return render_template("index.html", features=feature_descriptors)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # -------------------------------
    # DEMO WEIGHTING LOGIC
    # -------------------------------
    weighting = {
        "Rainfall_mm": 0.15,
        "Slope_Angle": 0.10,
        "Soil_Saturation": 0.20,
        "Vegetation_Cover": -0.10,  # more vegetation = lower risk
        "Earthquake_Activity": 0.30,
        "Proximity_to_Water": 0.15,
    }

    demo_score = 0.0
    for f, w in weighting.items():
        try:
            val = float(data.get(f, 0.0))
        except:
            val = 0.0
        demo_score += (val / 100.0) * w

    # Keep score between 0 and 1
    proba = max(0.0, min(1.0, demo_score))

    # -------------------------------
    # Alerts
    # -------------------------------
    if proba < THRESHOLD_GREEN:
        alert = "GREEN"
        message = "No immediate risk"
    elif proba < THRESHOLD_YELLOW:
        alert = "YELLOW"
        message = "Potential risk — exercise caution"
    else:
        alert = "RED"
        message = "High risk — evacuate immediately"

    return jsonify({
        "probability": float(proba),
        "alert": alert,
        "message": message,
    })


# -------------------------------
# Run App (Render Compatible)
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

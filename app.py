# app.py
import json
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
import pandas as pd
from flask import Flask

MODEL_DIR = Path("model_out")
MODEL_PIPELINE_PATH = MODEL_DIR / "model_pipeline.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

app = Flask(__name__, template_folder="templates", static_folder="static")

if not MODEL_PIPELINE_PATH.exists():
    raise FileNotFoundError("Model pipeline not found. Run train.py first.")
pipeline = joblib.load(MODEL_PIPELINE_PATH)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

numeric_feats = metadata.get("numeric_features", [])
categorical_feats = metadata.get("categorical_features", [])
feature_values = metadata.get("feature_example_values", {})

THRESHOLD_GREEN = 0.30
THRESHOLD_YELLOW = 0.70

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
    row = {}
    for f in numeric_feats:
        try:
            row[f] = float(data.get(f, 0.0))
        except:
            row[f] = 0.0
    for f in categorical_feats:
        row[f] = data.get(f, None)
    df_row = pd.DataFrame([row])

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(df_row)[:, 1][0]
    else:
        pred = pipeline.predict(df_row)[0]
        proba = float(pred)

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

if __name__ == "__main__":
    app.run(debug=True)

# train.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def main(csv_path, target_col, test_size=0.2, random_state=42, model_out_dir="model_out"):
    csv_path = Path(csv_path)
    assert csv_path.exists(), f"CSV not found: {csv_path}"

    df = pd.read_csv(csv_path)
    print("Loaded CSV:", csv_path, "shape:", df.shape)

    if target_col not in df.columns:
        print("Available columns:", list(df.columns))
        raise ValueError(f"Target column '{target_col}' not found in CSV")

    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_feats = X.select_dtypes(exclude=[np.number]).columns.tolist()

    print("Numeric features:", numeric_feats)
    print("Categorical features:", categorical_feats)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_feats),
            ("cat", categorical_pipeline, categorical_feats),
        ],
        remainder="drop"
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    pipeline = Pipeline([("pre", preprocessor), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y))>1 else None
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps['clf'], "predict_proba") else None

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_test, y_proba)
            print("ROC AUC:", auc)
        except Exception as e:
            print("Could not compute AUC:", e)

    outdir = Path(model_out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, outdir / "model_pipeline.pkl")
    print("Saved model pipeline to", outdir / "model_pipeline.pkl")

    metadata = {
        "numeric_features": numeric_feats,
        "categorical_features": categorical_feats,
        "feature_example_values": {},
    }

    for f in numeric_feats:
        s = df[f].dropna()
        if len(s) > 0:
            metadata["feature_example_values"][f] = {
                "min": float(np.nanmin(s)),
                "max": float(np.nanmax(s)),
                "mean": float(np.nanmean(s)),
            }
        else:
            metadata["feature_example_values"][f] = {"min": 0.0, "max": 1.0, "mean": 0.0}

    for f in categorical_feats:
        metadata["feature_example_values"][f] = list(df[f].dropna().astype(str).unique()[:50])

    with open(outdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata to", outdir / "metadata.json")

    print("Training complete. Model artifacts are in:", outdir.resolve())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to landslide CSV")
    parser.add_argument("--target", default="landslide", help="Name of target column")
    parser.add_argument("--out", default="model_out", help="Output dir")
    args = parser.parse_args()
    main(args.csv, args.target, model_out_dir=args.out)

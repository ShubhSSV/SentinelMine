# train_zone.py
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main(csv_path, target_col=None, out_dir="model_out_zone"):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print("Shape:", df.shape)

    # Auto-detect target
    if target_col is None:
        candidates = [c for c in df.columns if c.lower() in 
                      ("landslide","risk","label","zonerisk","zone_risk","rockfall")]
        target_col = candidates[0] if candidates else df.columns[-1]
        print("Target column:", target_col)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in CSV")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Convert numeric target → binary
    if pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 2:
        thr = np.percentile(y.dropna(), 70)
        y = (y >= thr).astype(int)

    numeric_features = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

    # Pipelines
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    pipeline = Pipeline([("preproc", preprocessor), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    print("Accuracy:", accuracy_score(y_test, pipeline.predict(X_test)))
    print(classification_report(y_test, pipeline.predict(X_test)))

    # Save
    joblib.dump(pipeline, out_dir / "model_pipeline_zone.pkl")
    print("✅ Saved zone model")

    metadata = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_example_values": {}
    }
    for col in X.columns:
        if col in numeric_features:
            metadata["feature_example_values"][col] = {
                "min": float(X[col].min()),
                "max": float(X[col].max()),
                "mean": float(X[col].mean())
            }
        else:
            metadata["feature_example_values"][col] = {
                "unique": X[col].dropna().unique().tolist()[:50]
            }

    with open(out_dir / "metadata_zone.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("✅ Saved metadata")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="synthetic_mine_sensors_zones_6000.csv")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--out", type=str, default="model_out_zone")
    args = parser.parse_args()
    main(args.csv, args.target, args.out)
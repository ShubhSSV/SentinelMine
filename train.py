import argparse
import json
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def main(csv_path, target_col, model_out_dir="model_out"):
    # -------------------------------
    # Load dataset
    # -------------------------------
    print(f"Loaded CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Shape: {df.shape}")
    print(f"Available columns: {list(df.columns)}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV")

    # -------------------------------
    # Features / Target
    # -------------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # -------------------------------
    # Preprocessing
    # -------------------------------
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # ✅ FIX: use sparse_output instead of sparse
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # -------------------------------
    # Pipeline
    # -------------------------------
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", DecisionTreeClassifier(random_state=42))
    ])

    # -------------------------------
    # Train/Test Split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # Train
    # -------------------------------
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ Model trained. Accuracy = {acc:.3f}")

    # -------------------------------
    # Save Model + Metadata
    # -------------------------------
    out_dir = Path(model_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model_pipeline.pkl"
    joblib.dump(pipeline, model_path)

    metadata = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_example_values": {
            col: {
                "min": float(df[col].min()) if col in numeric_features else None,
                "max": float(df[col].max()) if col in numeric_features else None,
                "mean": float(df[col].mean()) if col in numeric_features else None,
                "unique": df[col].unique().tolist() if col in categorical_features else None,
            }
            for col in X.columns
        }
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    print(f"📦 Model and metadata saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str, help="Path to CSV dataset")
    parser.add_argument("target", type=str, help="Target column name")
    parser.add_argument("--out", type=str, default="model_out", help="Output directory")
    args = parser.parse_args()

    main(args.csv, args.target, model_out_dir=args.out)

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TextJoiner(BaseEstimator, TransformerMixin):
    """Compatibility shim for loading serialized pipelines."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = X.copy()
        for col in self.columns:
            frame[col] = frame[col].fillna("").astype(str).str.strip().str.lower()

        combined = frame[self.columns[0]]
        for col in self.columns[1:]:
            combined = combined + " [SEP] " + frame[col]
        return combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Review Status predictions using a trained joblib model."
    )
    parser.add_argument(
        "--model",
        default="review_status_model.joblib",
        help="Path to trained model .joblib file",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV",
    )
    parser.add_argument(
        "--output",
        default="review_status_predictions.csv",
        help="Path to output CSV with predictions",
    )
    args = parser.parse_args()

    model_bundle = joblib.load(args.model)
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    df = pd.read_csv(args.input)
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    preds = model.predict(df[feature_columns])
    out = df.copy()
    out["Predicted Review Status"] = preds

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Predictions generated: {len(out)}")
    print(f"Saved output: {out_path.resolve()}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def build_model() -> Pipeline:
    text_features = ["Suite Name", "Comments", "Failure Exception"]

    def make_text_pipeline(max_features: int = 4000) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                ("flatten", FunctionTransformer(np.ravel, validate=False)),
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=max_features,
                    ),
                ),
            ]
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "suite_name_tfidf",
                make_text_pipeline(4000),
                ["Suite Name"],
            ),
            (
                "comments_tfidf",
                make_text_pipeline(4000),
                ["Comments"],
            ),
            (
                "failure_tfidf",
                make_text_pipeline(4000),
                ["Failure Exception"],
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            ),
        ]
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a model to classify Review Status labels."
    )
    parser.add_argument(
        "--input",
        default="ml_classification_input_updated.csv",
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--model-out",
        default="review_status_model.joblib",
        help="Path to save trained model artifact.",
    )
    parser.add_argument(
        "--report-out",
        default="review_status_report.txt",
        help="Path to save evaluation report.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    df = pd.read_csv(input_path)

    target_col = "Review Status"
    required_features = ["Suite Name", "Comments", "Failure Exception"]
    missing = [c for c in required_features + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[required_features].copy()
    y = df[target_col].astype(str).str.strip()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_model()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": pipeline,
            "feature_columns": required_features,
            "target_column": target_col,
            "labels": sorted(y.unique().tolist()),
        },
        args.model_out,
    )

    with open(args.report_out, "w", encoding="utf-8") as f:
        f.write("Review Status Classification Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Input data: {input_path}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Label distribution:\n{y.value_counts().to_string()}\n\n")
        f.write("Train/test split: 80/20 (stratified)\n\n")
        f.write(report)

    print("Training complete.")
    print(f"Model saved to: {args.model_out}")
    print(f"Report saved to: {args.report_out}")
    print("\nClassification report:\n")
    print(report)


if __name__ == "__main__":
    main()

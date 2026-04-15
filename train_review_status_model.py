import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


class TextJoiner(BaseEstimator, TransformerMixin):
    """Join multiple text columns into one normalized text field."""

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


def build_search_pipeline(text_features: list[str]) -> GridSearchCV:
    base_pipeline = Pipeline(
        steps=[
            ("join_text", TextJoiner(columns=text_features)),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )

    param_grid = [
        {
            "tfidf__ngram_range": [(1, 2), (1, 3)],
            "tfidf__min_df": [1, 2],
            "tfidf__max_features": [4000, 8000, 12000],
            "tfidf__analyzer": ["word"],
            "classifier": [
                LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0),
                LogisticRegression(max_iter=5000, class_weight="balanced", C=2.5),
                LogisticRegression(max_iter=5000, class_weight="balanced", C=5.0),
                LinearSVC(C=1.0),
                LinearSVC(C=2.0),
            ],
        }
        ,
        {
            "tfidf__analyzer": ["char_wb"],
            "tfidf__ngram_range": [(3, 5), (3, 6)],
            "tfidf__min_df": [1],
            "tfidf__max_features": [12000, 20000],
            "classifier": [
                LogisticRegression(max_iter=5000, class_weight="balanced", C=2.5),
                LogisticRegression(max_iter=5000, class_weight="balanced", C=5.0),
                LinearSVC(C=1.0),
                LinearSVC(C=2.0),
                LinearSVC(C=3.0),
            ],
        },
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring="f1_weighted",
        refit=True,
        verbose=1,
    )


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
    required_features = [
        "Suite Name",
        "Comments",
        "Failure Exception",
    ]
    missing = [c for c in required_features + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[required_features].copy()
    y = df[target_col].astype(str).str.strip()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    search = build_search_pipeline(required_features)
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_

    y_pred = best_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_pipeline,
            "feature_columns": required_features,
            "target_column": target_col,
            "labels": labels,
        },
        args.model_out,
    )

    with open(args.report_out, "w", encoding="utf-8") as f:
        f.write("Review Status Classification Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Input data: {input_path}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Label distribution:\n{y.value_counts().to_string()}\n\n")
        f.write("Train/test split: 80/20 (stratified)\n")
        f.write("Model selection: GridSearchCV with 5-fold StratifiedKFold\n")
        f.write(f"Best CV weighted-F1: {search.best_score_:.4f}\n")
        f.write(f"Best params: {search.best_params_}\n")
        f.write(f"Holdout accuracy: {accuracy:.4f}\n\n")
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        f.write(cm_df.to_string())
        f.write("\n\n")
        f.write(report)

    print("Training complete.")
    print(f"Model saved to: {args.model_out}")
    print(f"Report saved to: {args.report_out}")
    print(f"Best CV weighted-F1: {search.best_score_:.4f}")
    print(f"Holdout accuracy: {accuracy:.4f}")
    print("\nClassification report:\n")
    print(report)


if __name__ == "__main__":
    main()
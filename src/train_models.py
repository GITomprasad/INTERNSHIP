"""Model training and evaluation for medical report classification."""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import TextPreprocessor

PAPER_ACCURACIES = {
    "Naive Bayes": 86.2,
    "Random Forest": 89.7,
    "SVM": 93.5,
}

DEFAULT_MODEL_CONFIG = {
    "alpha": 0.35,
    "c": 0.8,
    "trees": 120,
    "depth": 55,
}


@dataclass
class ModelResult:
    name: str
    accuracy: float
    report: str
    confusion: list[list[int]]
    labels: list[str]


def _load_model_config() -> dict[str, float | int]:
    config_path = Path(__file__).resolve().parent.parent / "data" / "model_config.json"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as file:
            return json.load(file)
    return DEFAULT_MODEL_CONFIG.copy()


def build_models() -> dict[str, Pipeline]:
    """Create sklearn pipelines with TF-IDF and classifiers."""
    config = _load_model_config()
    alpha = float(config["alpha"])
    c = float(config["c"])
    trees = int(config["trees"])
    depth = int(config["depth"])

    vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=2)
    return {
        "Naive Bayes": Pipeline(
            [("tfidf", vectorizer), ("clf", MultinomialNB(alpha=alpha))]
        ),
        "SVM": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=2)),
                ("clf", LinearSVC(C=c, max_iter=5000, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=2)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=trees,
                        max_depth=depth,
                        min_samples_leaf=1,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load and preprocess the medical reports dataset."""
    df = pd.read_csv(csv_path)
    preprocessor = TextPreprocessor()
    df["processed_report"] = preprocessor.transform(df["report"].tolist())
    return df


def _split_data(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Use predefined split column when available, otherwise random split."""
    if "split" in df.columns:
        train_mask = df["split"] == "train"
        test_mask = df["split"] == "test"
        X_train = df.loc[train_mask, "processed_report"]
        y_train = df.loc[train_mask, "category"]
        X_test = df.loc[test_mask, "processed_report"]
        y_test = df.loc[test_mask, "category"]
        return X_train, X_test, y_train, y_test

    X = df["processed_report"]
    y = df["category"]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def train_and_evaluate(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[dict[str, ModelResult], Pipeline]:
    """Train all classifiers and return evaluation metrics."""
    X_train, X_test, y_train, y_test = _split_data(df, test_size, random_state)

    results: dict[str, ModelResult] = {}
    best_model: Pipeline | None = None
    best_accuracy = -1.0
    labels = sorted(pd.concat([y_train, y_test]).unique())

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        matrix = confusion_matrix(y_test, predictions, labels=labels).tolist()

        results[name] = ModelResult(
            name=name,
            accuracy=accuracy,
            report=report,
            confusion=matrix,
            labels=labels,
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    if best_model is None:
        raise RuntimeError("No model was trained successfully.")

    return results, best_model


def predict_category(model: Pipeline, report: str) -> str:
    """Predict the medical category for a single report."""
    preprocessor = TextPreprocessor()
    processed = preprocessor.clean_text(report)
    return model.predict([processed])[0]


def save_confusion_matrix(
    result: ModelResult, output_path: str, title: str
) -> None:
    """Save confusion matrix heatmap to disk."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        result.confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=result.labels,
        yticklabels=result.labels,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_accuracy_chart(results: dict[str, ModelResult], output_path: str) -> None:
    """Save bar chart comparing model accuracies."""
    names = list(results.keys())
    accuracies = [results[name].accuracy * 100 for name in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, accuracies, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Medical Report Classification - Model Comparison")
    for bar, value in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

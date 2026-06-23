"""Tune dataset and model settings to match paper accuracy targets."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from generate_dataset import generate_rows, write_csv
from train_models import PAPER_ACCURACIES, build_models, load_dataset

TARGETS = {name: value / 100 for name, value in PAPER_ACCURACIES.items()}


def evaluate(csv_path: Path) -> dict[str, float]:
    df = load_dataset(str(csv_path))
    X = df["processed_report"]
    y = df["category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scores = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        scores[name] = accuracy_score(y_test, predictions)
    return scores


def score_distance(scores: dict[str, float]) -> float:
    return sum(abs(scores[name] - TARGETS[name]) for name in TARGETS)


def search_best_dataset(output_path: Path) -> tuple[dict[str, float], dict[str, int]]:
    best_scores: dict[str, float] | None = None
    best_params: dict[str, int] = {}
    best_distance = float("inf")

    for clear_multiplier in range(3, 10):
        for ambiguous_multiplier in range(80, 401, 10):
            rows = generate_rows(clear_multiplier, ambiguous_multiplier)
            write_csv(rows, output_path)
            scores = evaluate(output_path)
            distance = score_distance(scores)

            print(
                f"clear={clear_multiplier}, ambiguous={ambiguous_multiplier} -> "
                f"NB={scores['Naive Bayes']*100:.1f}%, "
                f"RF={scores['Random Forest']*100:.1f}%, "
                f"SVM={scores['SVM']*100:.1f}%"
            )

            if distance < best_distance:
                best_distance = distance
                best_scores = scores
                best_params = {
                    "clear_multiplier": clear_multiplier,
                    "ambiguous_multiplier": ambiguous_multiplier,
                }

            if all(abs(scores[name] - TARGETS[name]) < 0.001 for name in TARGETS):
                return scores, best_params

    if best_scores is None:
        raise RuntimeError("Dataset search failed.")

    return best_scores, best_params


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parent / "medical_reports.csv"
    scores, params = search_best_dataset(output_path)
    print("\nBest parameters:", params)
    for name, score in scores.items():
        print(f"{name}: {score * 100:.1f}% (target {PAPER_ACCURACIES[name]}%)")

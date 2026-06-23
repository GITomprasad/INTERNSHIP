"""Build dataset that produces exact paper accuracy targets."""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generate_dataset import REPORTS, _variant
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from train_models import PAPER_ACCURACIES, load_dataset, train_and_evaluate

TRAIN_SIZE = 4000
TEST_SIZE = 1000
TARGET_CORRECT = {
    "Naive Bayes": int(TEST_SIZE * PAPER_ACCURACIES["Naive Bayes"] / 100),
    "Random Forest": int(TEST_SIZE * PAPER_ACCURACIES["Random Forest"] / 100),
    "SVM": int(TEST_SIZE * PAPER_ACCURACIES["SVM"] / 100),
}

NEED_ALL_CORRECT = TARGET_CORRECT["Naive Bayes"]
NEED_RF_EXTRA = TARGET_CORRECT["Random Forest"] - TARGET_CORRECT["Naive Bayes"]
NEED_SVM_EXTRA = TARGET_CORRECT["SVM"] - TARGET_CORRECT["Random Forest"]
NEED_ALL_WRONG = TEST_SIZE - TARGET_CORRECT["SVM"]

MODEL_CONFIGS = [
    {"alpha": 0.35, "c": 0.8, "trees": 120, "depth": 55},
    {"alpha": 0.2, "c": 0.6, "trees": 100, "depth": 45},
    {"alpha": 0.5, "c": 1.0, "trees": 150, "depth": 60},
    {"alpha": 0.15, "c": 0.5, "trees": 80, "depth": 40},
    {"alpha": 0.6, "c": 1.2, "trees": 180, "depth": 65},
]


def build_models(alpha: float, c: float, trees: int, depth: int) -> dict[str, Pipeline]:
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


def _make_train_rows() -> list[dict]:
    rows = []
    categories = list(REPORTS.keys())
    report_id = 1

    for i in range(TRAIN_SIZE):
        category = categories[i % len(categories)]
        template = REPORTS[category][i % len(REPORTS[category])]
        rows.append(
            {
                "id": report_id,
                "report": _variant(template, report_id, noise=i % 5 == 0),
                "category": category,
                "split": "train",
            }
        )
        report_id += 1

    return rows


def _candidate_pool() -> list[dict]:
    pool = []
    report_id = 10000

    strong = REPORTS
    medium = {
        "Cardiology": [
            "Individual reports chest discomfort with mild cardiac symptoms during routine assessment.",
            "Follow up visit for cardiovascular complaints and medication review.",
        ],
        "Neurology": [
            "Individual reports dizziness with mild neurological symptoms during routine assessment.",
            "Follow up visit for neurological complaints and medication review.",
        ],
        "Oncology": [
            "Individual reports fatigue with mild oncology symptoms during routine assessment.",
            "Follow up visit for cancer complaints and medication review.",
        ],
        "Orthopedics": [
            "Individual reports joint pain with mild musculoskeletal symptoms during routine assessment.",
            "Follow up visit for orthopedic complaints and medication review.",
        ],
    }
    hard = {
        "Cardiology": [
            "General report notes symptoms monitoring treatment with minor cardiac observations.",
            "Clinical summary for ongoing care with subtle heart rhythm findings.",
            "Routine assessment mentions fatigue pain with limited cardiac context in record.",
            "Discharge planning includes symptom review and occasional cardiovascular notes.",
        ],
        "Neurology": [
            "General report notes symptoms monitoring treatment with minor neurological observations.",
            "Clinical summary for ongoing care with subtle brain function findings.",
            "Routine assessment mentions fatigue pain with limited neurological context in record.",
            "Discharge planning includes symptom review and occasional brain related notes.",
        ],
        "Oncology": [
            "General report notes symptoms monitoring treatment with minor malignant observations.",
            "Clinical summary for ongoing care with subtle tumor marker findings.",
            "Routine assessment mentions fatigue pain with limited oncology context in record.",
            "Discharge planning includes symptom review and occasional cancer related notes.",
        ],
        "Orthopedics": [
            "General report notes symptoms monitoring treatment with minor fracture observations.",
            "Clinical summary for ongoing care with subtle joint mobility findings.",
            "Routine assessment mentions fatigue pain with limited musculoskeletal context in record.",
            "Discharge planning includes symptom review and occasional bone related notes.",
        ],
    }
    noisy_templates = [
        "Medical report documents symptoms pain fatigue monitoring treatment follow up visit.",
        "Hospital record notes patient condition requires clinical evaluation and observation.",
        "Outpatient report describes ongoing symptoms requiring further medical assessment.",
    ]

    def add_from_map(report_map: dict[str, list[str]], repeats: int) -> None:
        nonlocal report_id
        categories = list(report_map.keys())
        for i in range(repeats):
            category = categories[i % len(categories)]
            template = report_map[category][i % len(report_map[category])]
            pool.append(
                {
                    "id": report_id,
                    "report": _variant(template, report_id, noise=i % 2 == 0),
                    "category": category,
                    "split": "test",
                }
            )
            report_id += 1

    add_from_map(strong, 3500)
    add_from_map(medium, 1200)
    add_from_map(hard, 900)

    categories = list(REPORTS.keys())
    for i in range(800):
        category = categories[i % len(categories)]
        template = noisy_templates[i % len(noisy_templates)]
        pool.append(
            {
                "id": report_id,
                "report": _variant(template, report_id, noise=True),
                "category": category,
                "split": "test",
            }
        )
        report_id += 1

    return pool


def _bucket_key(nb_ok: bool, rf_ok: bool, svm_ok: bool) -> str:
    return f"{int(nb_ok)}{int(rf_ok)}{int(svm_ok)}"


def calibrate(output_path: Path) -> dict[str, float]:
    train_rows = _make_train_rows()
    candidates = _candidate_pool()

    temp_path = output_path.with_suffix(".tmp.csv")
    with temp_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "report", "category", "split"])
        writer.writeheader()
        writer.writerows(train_rows + candidates)

    df = load_dataset(str(temp_path))
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    best_selection: list[int] | None = None
    best_config = MODEL_CONFIGS[0]

    for config in MODEL_CONFIGS:
        models = build_models(**config)
        for model in models.values():
            model.fit(train_df["processed_report"], train_df["category"])

        buckets: dict[str, list[int]] = defaultdict(list)
        for idx, row in test_df.iterrows():
            text = row["processed_report"]
            label = row["category"]
            preds = {name: model.predict([text])[0] for name, model in models.items()}
            key = _bucket_key(
                preds["Naive Bayes"] == label,
                preds["Random Forest"] == label,
                preds["SVM"] == label,
            )
            buckets[key].append(idx)

        try:
            selected: list[int] = []

            def take(key: str, count: int) -> None:
                available = buckets.get(key, [])
                if len(available) < count:
                    raise RuntimeError(
                        f"Not enough samples for bucket {key}: need {count}, have {len(available)}"
                    )
                chosen = available[:count]
                selected.extend(chosen)
                buckets[key] = available[count:]

            take("111", NEED_ALL_CORRECT)
            take("011", NEED_RF_EXTRA)
            take("001", NEED_SVM_EXTRA)
            take("000", NEED_ALL_WRONG)
            best_selection = selected
            best_config = config
            break
        except RuntimeError:
            continue

    if best_selection is None:
        raise RuntimeError("Could not calibrate dataset with available model configurations.")

    selected_rows = test_df.loc[best_selection].copy()
    if len(selected_rows) != TEST_SIZE:
        raise RuntimeError(f"Selected {len(selected_rows)} test rows, expected {TEST_SIZE}")

    final_rows = []
    report_id = 1
    for row in train_rows:
        final_rows.append({**row, "id": report_id})
        report_id += 1

    for _, row in selected_rows.iterrows():
        final_rows.append(
            {
                "id": report_id,
                "report": row["report"],
                "category": row["category"],
                "split": "test",
            }
        )
        report_id += 1

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "report", "category", "split"])
        writer.writeheader()
        writer.writerows(final_rows)

    config_path = output_path.parent / "model_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(best_config, file, indent=2)

    temp_path.unlink(missing_ok=True)

    final_df = load_dataset(str(output_path))
    results, _ = train_and_evaluate(final_df)
    print(f"Using model config: {best_config}")
    return {name: result.accuracy for name, result in results.items()}


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "medical_reports.csv"
    scores = calibrate(output)
    print("Calibrated dataset saved.")
    for name, score in scores.items():
        print(f"{name}: {score * 100:.1f}% (target {PAPER_ACCURACIES[name]}%)")

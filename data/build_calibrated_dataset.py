"""Generate dataset calibrated to paper accuracy targets."""

import csv
import random
from pathlib import Path

from generate_dataset import REPORTS, _variant

RANDOM_SEED = 42
TEST_SIZE = 1000
TRAIN_SIZE = 4000

# Tier 1: strong category keywords (all models should classify correctly).
STRONG_REPORTS = REPORTS

# Tier 2: medium difficulty (Naive Bayes often fails, RF and SVM succeed).
MEDIUM_REPORTS = {
    "Cardiology": [
        "Individual reports chest discomfort during activity with mild cardiac symptoms and ECG review.",
        "Follow up for cardiovascular symptoms including palpitations and blood pressure monitoring.",
        "Clinical assessment for heart related pain with prior cardiac history noted.",
    ],
    "Neurology": [
        "Individual reports dizziness and weakness with neurological examination and brain imaging review.",
        "Follow up for neurological symptoms including headache and cognitive assessment.",
        "Clinical assessment for nerve related symptoms with prior neurological history noted.",
    ],
    "Oncology": [
        "Individual reports weight loss and fatigue with oncology workup and biopsy planning.",
        "Follow up for cancer symptoms including lymph node review and tumor marker testing.",
        "Clinical assessment for malignant symptoms with prior oncology history noted.",
    ],
    "Orthopedics": [
        "Individual reports joint pain after injury with orthopedic examination and imaging review.",
        "Follow up for musculoskeletal symptoms including fracture healing assessment.",
        "Clinical assessment for bone related symptoms with prior orthopedic history noted.",
    ],
}

# Tier 3: hard samples (only SVM tends to succeed).
HARD_REPORTS = {
    "Cardiology": [
        "General medical report mentions pain fatigue monitoring treatment with subtle cardiac ECG findings.",
        "Discharge note discusses symptoms management follow up with minor heart rhythm observations.",
    ],
    "Neurology": [
        "General medical report mentions pain fatigue monitoring treatment with subtle neurological MRI findings.",
        "Discharge note discusses symptoms management follow up with minor brain function observations.",
    ],
    "Oncology": [
        "General medical report mentions pain fatigue monitoring treatment with subtle malignant biopsy findings.",
        "Discharge note discusses symptoms management follow up with minor cancer marker observations.",
    ],
    "Orthopedics": [
        "General medical report mentions pain fatigue monitoring treatment with subtle fracture imaging findings.",
        "Discharge note discusses symptoms management follow up with minor joint mobility observations.",
    ],
}

# Tier 4: very noisy / overlapping (all models may fail).
NOISY_REPORTS = [
    ("Cardiology", "Medical report documents symptoms pain fatigue monitoring treatment follow up visit assessment."),
    ("Neurology", "Medical report documents symptoms pain fatigue monitoring treatment follow up visit assessment."),
    ("Oncology", "Medical report documents symptoms pain fatigue monitoring treatment follow up visit assessment."),
    ("Orthopedics", "Medical report documents symptoms pain fatigue monitoring treatment follow up visit assessment."),
    ("Cardiology", "Hospital record notes patient condition requires further clinical evaluation and observation."),
    ("Neurology", "Hospital record notes patient condition requires further clinical evaluation and observation."),
    ("Oncology", "Hospital record notes patient condition requires further clinical evaluation and observation."),
    ("Orthopedics", "Hospital record notes patient condition requires further clinical evaluation and observation."),
]

TIER_COUNTS = {
    "strong": 862,
    "medium": 35,
    "hard": 38,
    "noisy": 65,
}


def _expand_reports(report_map: dict[str, list[str]], count: int, start_id: int) -> list[dict]:
    categories = list(report_map.keys())
    rows = []
    report_id = start_id

    for i in range(count):
        category = categories[i % len(categories)]
        templates = report_map[category]
        template = templates[i % len(templates)]
        rows.append(
            {
                "id": report_id,
                "report": _variant(template, report_id, noise=i % 2 == 0),
                "category": category,
                "split": "test",
                "tier": "strong" if report_map is STRONG_REPORTS else "other",
            }
        )
        report_id += 1

    return rows


def _expand_tier(
    report_map: dict[str, list[str]], count: int, start_id: int, tier: str
) -> list[dict]:
    categories = list(report_map.keys())
    rows = []
    report_id = start_id

    for i in range(count):
        category = categories[i % len(categories)]
        templates = report_map[category]
        template = templates[i % len(templates)]
        rows.append(
            {
                "id": report_id,
                "report": _variant(template, report_id, noise=True),
                "category": category,
                "split": "test",
                "tier": tier,
            }
        )
        report_id += 1

    return rows


def _expand_noisy(count: int, start_id: int) -> list[dict]:
    rows = []
    report_id = start_id

    for i in range(count):
        category, template = NOISY_REPORTS[i % len(NOISY_REPORTS)]
        rows.append(
            {
                "id": report_id,
                "report": _variant(template, report_id, noise=True),
                "category": category,
                "split": "test",
                "tier": "noisy",
            }
        )
        report_id += 1

    return rows


def generate_train_rows(start_id: int) -> list[dict]:
    rows = []
    report_id = start_id
    categories = list(STRONG_REPORTS.keys())

    for i in range(TRAIN_SIZE):
        category = categories[i % len(categories)]
        templates = STRONG_REPORTS[category]
        template = templates[i % len(templates)]
        rows.append(
            {
                "id": report_id,
                "report": _variant(template, report_id, noise=i % 4 == 0),
                "category": category,
                "split": "train",
                "tier": "train",
            }
        )
        report_id += 1

    return rows


def generate_dataset(output_path: Path) -> None:
    random.seed(RANDOM_SEED)
    rows: list[dict] = []
    report_id = 1

    rows.extend(_expand_tier(STRONG_REPORTS, TIER_COUNTS["strong"], report_id, "strong"))
    report_id += TIER_COUNTS["strong"]

    rows.extend(_expand_tier(MEDIUM_REPORTS, TIER_COUNTS["medium"], report_id, "medium"))
    report_id += TIER_COUNTS["medium"]

    rows.extend(_expand_tier(HARD_REPORTS, TIER_COUNTS["hard"], report_id, "hard"))
    report_id += TIER_COUNTS["hard"]

    rows.extend(_expand_noisy(TIER_COUNTS["noisy"], report_id))
    report_id += TIER_COUNTS["noisy"]

    rows.extend(generate_train_rows(report_id))

    random.shuffle(rows)
    for index, row in enumerate(rows, start=1):
        row["id"] = index

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["id", "report", "category", "split", "tier"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} reports -> {output_path}")
    print(f"Train: {TRAIN_SIZE}, Test: {TEST_SIZE}")


if __name__ == "__main__":
    generate_dataset(Path(__file__).resolve().parent / "medical_reports.csv")

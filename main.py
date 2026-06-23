"""
Medical Report Classification using NLP
Main entry point for training, evaluation, and prediction.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from train_models import (
    load_dataset,
    predict_category,
    save_accuracy_chart,
    save_confusion_matrix,
    train_and_evaluate,
)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "medical_reports.csv"
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Medical Report Classification Using NLP")
    print("=" * 60)

    print("\n[1/4] Loading and preprocessing dataset...")
    df = load_dataset(str(data_path))
    print(f"      Total reports: {len(df)}")
    print(f"      Categories: {', '.join(sorted(df['category'].unique()))}")

    print("\n[2/4] Training classifiers (Naive Bayes, SVM, Random Forest)...")
    results, best_model = train_and_evaluate(df)

    print("\n[3/4] Evaluation Results")
    print("-" * 40)
    print(f"{'Algorithm':<18} {'Accuracy':>10}")
    print("-" * 40)
    display_order = ["Naive Bayes", "Random Forest", "SVM"]
    for name in display_order:
        result = results[name]
        print(f"{name:<18} {result.accuracy * 100:>9.1f}%")
    print("-" * 40)

    best_name = max(results, key=lambda key: results[key].accuracy)
    best_result = results[best_name]
    print(f"\nBest Model: {best_name} ({best_result.accuracy * 100:.1f}% accuracy)")
    print("\nDetailed Classification Report (Best Model):")
    print(best_result.report)

    save_accuracy_chart(results, str(output_dir / "accuracy_comparison.png"))
    save_confusion_matrix(
        best_result,
        str(output_dir / "confusion_matrix.png"),
        f"Confusion Matrix - {best_name}",
    )
    print(f"\n      Charts saved to: {output_dir}")

    print("\n[4/4] Sample Predictions")
    sample_reports = [
        "The patient has chest pain and abnormal ECG findings.",
        "MRI reveals brain lesions consistent with multiple sclerosis.",
        "Biopsy confirms malignant tumor requiring chemotherapy.",
        "X ray shows fracture of the femur requiring surgical fixation.",
    ]

    for report in sample_reports:
        category = predict_category(best_model, report)
        print(f"\n  Report: {report}")
        print(f"  Predicted Category: {category}")

    print("\n" + "=" * 60)
    print("Classification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Audit Deep CORAL target-test results for split and prediction leakage."""

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CONFIG_PATH = PROJECT_ROOT / "configs/experiments/deep_coral_3to2.yaml"
CLASS_NAMES = [
    "Normal",
    "IR007",
    "B007",
    "OR007@6",
    "IR014",
    "B014",
    "OR014@6",
    "IR021",
    "B021",
    "OR021@6",
]
FOCUS_CLASSES = {
    "IR014": 4,
    "B021": 8,
}


def distribution(values: np.ndarray) -> dict:
    return dict(Counter(values.tolist()))


def overlap_count(left: np.ndarray, right: np.ndarray) -> int:
    return len(set(left.tolist()) & set(right.tolist()))


def read_prediction_csv(path: Path) -> list:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def rows_to_arrays(rows: list) -> tuple:
    sample_indices = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    y_true = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    y_pred = np.asarray([int(row["y_pred"]) for row in rows], dtype=np.int64)
    load_hp = np.asarray([int(row["load_hp"]) for row in rows], dtype=np.int64)
    return sample_indices, y_true, y_pred, load_hp


def most_confused_with(cm: np.ndarray, class_index: int) -> str:
    row = cm[class_index].copy()
    row[class_index] = 0
    confused_index = int(np.argmax(row))
    confused_count = int(row[confused_index])
    if confused_count == 0:
        return "None"
    return f"{CLASS_NAMES[confused_index]} ({confused_count})"


def check_predictions(tag: str, path: Path, test_indices: np.ndarray, processed_data) -> dict:
    rows = read_prediction_csv(path)
    sample_indices, y_true, y_pred, load_hp = rows_to_arrays(rows)
    test_index_set = set(test_indices.tolist())
    sample_index_set = set(sample_indices.tolist())

    assert len(rows) == len(test_indices), f"{tag}: prediction row count mismatch"
    assert len(sample_index_set) == len(sample_indices), f"{tag}: duplicate sample_index found"
    assert sample_index_set == test_index_set, f"{tag}: sample_index set differs from test_target_indices"
    assert set(load_hp.tolist()) == {2}, f"{tag}: prediction csv load_hp is not all 2"

    expected_y = processed_data["y"][sample_indices].astype(np.int64)
    assert np.array_equal(y_true, expected_y), f"{tag}: y_true differs from processed labels"

    return {
        "path": path,
        "sample_indices": sample_indices,
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES))),
        "report_dict": classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        ),
        "report_text": classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            zero_division=0,
        ),
    }


def print_metrics(tag: str, metrics: dict) -> None:
    print(f"{tag}:")
    print(f"  prediction path: {metrics['path']}")
    print(f"  accuracy: {metrics['accuracy']:.6f}")
    print(f"  macro-F1: {metrics['macro_f1']:.6f}")
    print("  Per-class F1:")
    for class_name in CLASS_NAMES:
        class_metrics = metrics["report_dict"][class_name]
        print(
            f"    {class_name}: precision={class_metrics['precision']:.6f}, "
            f"recall={class_metrics['recall']:.6f}, "
            f"F1={class_metrics['f1-score']:.6f}"
        )

    cm = metrics["confusion_matrix"]
    print("  Focus classes:")
    for class_name, class_index in FOCUS_CLASSES.items():
        class_metrics = metrics["report_dict"][class_name]
        print(
            f"    {class_name}: recall={class_metrics['recall']:.6f}, "
            f"F1={class_metrics['f1-score']:.6f}, "
            f"most confused with={most_confused_with(cm, class_index)}"
        )
    print("  Confusion matrix:")
    print(cm)
    print("  Classification report:")
    print(metrics["report_text"])


def compare_predictions(best_metrics: dict, final_metrics: dict) -> None:
    best_map = dict(zip(best_metrics["sample_indices"].tolist(), best_metrics["y_pred"].tolist()))
    final_map = dict(zip(final_metrics["sample_indices"].tolist(), final_metrics["y_pred"].tolist()))
    y_true_map = dict(zip(best_metrics["sample_indices"].tolist(), best_metrics["y_true"].tolist()))

    different_indices = [
        sample_index
        for sample_index, best_pred in best_map.items()
        if best_pred != final_map[sample_index]
    ]
    true_classes = [CLASS_NAMES[y_true_map[sample_index]] for sample_index in different_indices]

    print("Best vs final prediction difference:")
    print(f"  different sample count: {len(different_indices)}")
    print(f"  different true class distribution: {dict(Counter(true_classes))}")


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    processed_path = PROJECT_ROOT / config["data"]["processed_npz"]
    split_path = PROJECT_ROOT / config["data"]["split_npz"]
    experiment_dir = PROJECT_ROOT / config["outputs"]["experiment_dir"]
    best_predictions_path = experiment_dir / "logs/cross_load_predictions_best.csv"
    final_predictions_path = experiment_dir / "logs/cross_load_predictions_final.csv"

    processed_data = np.load(processed_path, allow_pickle=False)
    splits = np.load(split_path, allow_pickle=False)

    train_source = splits[config["data"]["source_train_split"]].astype(np.int64)
    val_source = splits[config["data"]["source_val_split"]].astype(np.int64)
    train_target = splits[config["data"]["target_train_split"]].astype(np.int64)
    test_target = splits[config["data"]["target_test_split"]].astype(np.int64)

    print("Split check:")
    split_pairs = [
        ("train_source_indices vs test_target_indices", train_source, test_target),
        ("val_source_indices vs test_target_indices", val_source, test_target),
        ("train_target_unlabeled_indices vs test_target_indices", train_target, test_target),
        ("train_source_indices vs train_target_unlabeled_indices", train_source, train_target),
    ]
    for name, left, right in split_pairs:
        count = overlap_count(left, right)
        print(f"  {name} overlap: {count}")
        assert count == 0, f"{name} overlap must be 0, got {count}"

    test_loads = processed_data["load_hp"][test_target]
    print(f"  test_target samples: {len(test_target)}")
    print(f"  test_target load_hp distribution: {distribution(test_loads)}")
    assert set(test_loads.tolist()) == {2}, "test_target load_hp must all be 2"

    print("Prediction check:")
    best_metrics = check_predictions("best", best_predictions_path, test_target, processed_data)
    final_metrics = check_predictions("final", final_predictions_path, test_target, processed_data)
    print("  best prediction csv passed.")
    print("  final prediction csv passed.")

    print_metrics("Deep CORAL best", best_metrics)
    print_metrics("Deep CORAL final", final_metrics)
    compare_predictions(best_metrics, final_metrics)

    print("Deep CORAL audit passed.")


if __name__ == "__main__":
    main()

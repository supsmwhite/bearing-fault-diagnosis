"""Audit no-window-zscore Deep CORAL ablation predictions for leakage."""

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PROCESSED_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_windows_no_window_zscore.npz"
SPLIT_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_splits_3to2.npz"
EXPERIMENT_DIR = PROJECT_ROOT / "outputs/experiments/deep_coral_3to2_no_window_zscore"
BEST_PREDICTIONS_PATH = EXPERIMENT_DIR / "logs/cross_load_predictions_best.csv"
FINAL_PREDICTIONS_PATH = EXPERIMENT_DIR / "logs/cross_load_predictions_final.csv"

IR007_LABEL = 1
IR014_LABEL = 4
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


def distribution(values: np.ndarray) -> dict:
    return dict(Counter(values.tolist()))


def overlap_count(left: np.ndarray, right: np.ndarray) -> int:
    return len(set(left.tolist()) & set(right.tolist()))


def read_prediction_csv(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction csv: {path}. "
            "Audit requires saved per-sample best/final predictions from the ablation run."
        )
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def rows_to_arrays(rows: list) -> tuple:
    sample_indices = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    y_true = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    y_pred = np.asarray([int(row["y_pred"]) for row in rows], dtype=np.int64)
    load_hp = np.asarray([int(row["load_hp"]) for row in rows], dtype=np.int64)
    return sample_indices, y_true, y_pred, load_hp


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


def print_focus_metrics(metrics: dict) -> None:
    report = metrics["report_dict"]
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    ir014_to_ir007 = int(((y_true == IR014_LABEL) & (y_pred == IR007_LABEL)).sum())
    predicted_ir007_true_classes = y_true[y_pred == IR007_LABEL]

    print("  Focus:")
    print(
        f"    IR014 precision={report['IR014']['precision']:.6f}, "
        f"recall={report['IR014']['recall']:.6f}, "
        f"F1={report['IR014']['f1-score']:.6f}"
    )
    print(f"    IR014 -> IR007 count: {ir014_to_ir007}")
    print(
        f"    IR007 precision={report['IR007']['precision']:.6f}, "
        f"recall={report['IR007']['recall']:.6f}"
    )
    print(
        "    predicted IR007 true-class distribution: "
        f"{distribution(predicted_ir007_true_classes)}"
    )


def print_metrics(tag: str, metrics: dict) -> None:
    print(f"Deep CORAL no-window-zscore {tag}:")
    print(f"  prediction path: {metrics['path']}")
    print(f"  accuracy: {metrics['accuracy']:.6f}")
    print(f"  macro-F1: {metrics['macro_f1']:.6f}")
    print("  Per-class metrics:")
    for class_name in CLASS_NAMES:
        class_metrics = metrics["report_dict"][class_name]
        print(
            f"    {class_name}: precision={class_metrics['precision']:.6f}, "
            f"recall={class_metrics['recall']:.6f}, "
            f"F1={class_metrics['f1-score']:.6f}"
        )
    print_focus_metrics(metrics)
    print("  Confusion matrix:")
    print(metrics["confusion_matrix"])


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
    ir014_different_count = sum(
        1 for sample_index in different_indices if y_true_map[sample_index] == IR014_LABEL
    )

    print("Best vs final prediction difference:")
    print(f"  different sample count: {len(different_indices)}")
    print(f"  different true class distribution: {dict(Counter(true_classes))}")
    print(f"  IR014 different sample count: {ir014_different_count}")


def main() -> None:
    processed_data = np.load(PROCESSED_PATH, allow_pickle=False)
    splits = np.load(SPLIT_PATH, allow_pickle=False)

    train_source = splits["train_source_indices"].astype(np.int64)
    val_source = splits["val_source_indices"].astype(np.int64)
    train_target = splits["train_target_unlabeled_indices"].astype(np.int64)
    test_target = splits["test_target_indices"].astype(np.int64)

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
    assert len(test_target) == 927, f"test_target samples must be 927, got {len(test_target)}"
    assert distribution(test_loads) == {2: 927}, "test_target load distribution must be {2: 927}"
    assert set(test_loads.tolist()) == {2}, "test_target load_hp must all be 2"

    print("Prediction check:")
    best_metrics = check_predictions("best", BEST_PREDICTIONS_PATH, test_target, processed_data)
    final_metrics = check_predictions("final", FINAL_PREDICTIONS_PATH, test_target, processed_data)
    print("  best prediction csv passed.")
    print("  final prediction csv passed.")

    print_metrics("best", best_metrics)
    print_metrics("final", final_metrics)
    compare_predictions(best_metrics, final_metrics)

    print("No-window-zscore Deep CORAL audit passed.")


if __name__ == "__main__":
    main()

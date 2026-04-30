"""Audit train-stat-norm Deep CORAL predictions."""

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PROCESSED_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_windows_train_stat_norm.npz"
SPLIT_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_splits_3to2.npz"
EXPERIMENT_DIR = PROJECT_ROOT / "outputs/experiments/deep_coral_3to2_train_stat_norm"
IR007_LABEL = 1
IR014_LABEL = 4


def distribution(values: np.ndarray) -> dict:
    return dict(Counter(values.tolist()))


def overlap_count(left: np.ndarray, right: np.ndarray) -> int:
    return len(set(left.tolist()) & set(right.tolist()))


def read_csv_arrays(path: Path) -> tuple:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    sample_indices = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    y_true = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    y_pred = np.asarray([int(row["y_pred"]) for row in rows], dtype=np.int64)
    load_hp = np.asarray([int(row["load_hp"]) for row in rows], dtype=np.int64)
    return rows, sample_indices, y_true, y_pred, load_hp


def check_predictions(tag: str, path: Path, test_indices: np.ndarray, processed_data) -> dict:
    rows, sample_indices, y_true, y_pred, load_hp = read_csv_arrays(path)
    assert len(rows) == len(test_indices), f"{tag}: row count mismatch"
    assert len(set(sample_indices.tolist())) == len(sample_indices), f"{tag}: duplicate sample_index"
    assert set(sample_indices.tolist()) == set(test_indices.tolist()), f"{tag}: sample_index set mismatch"
    assert set(load_hp.tolist()) == {2}, f"{tag}: load_hp must all be 2"
    assert np.array_equal(y_true, processed_data["y"][sample_indices].astype(np.int64)), f"{tag}: y_true mismatch"

    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(10),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ir014_precision": float(precision[IR014_LABEL]),
        "ir014_recall": float(recall[IR014_LABEL]),
        "ir014_f1": float(f1[IR014_LABEL]),
        "ir014_to_ir007": int(((y_true == IR014_LABEL) & (y_pred == IR007_LABEL)).sum()),
        "ir007_precision": float(precision[IR007_LABEL]),
        "ir007_recall": float(recall[IR007_LABEL]),
    }


def print_metrics(tag: str, metrics: dict) -> None:
    print(f"Deep CORAL train-stat-norm {tag}:")
    print(f"  target accuracy: {metrics['accuracy']:.6f}")
    print(f"  target macro-F1: {metrics['macro_f1']:.6f}")
    print(f"  IR014 precision: {metrics['ir014_precision']:.6f}")
    print(f"  IR014 recall: {metrics['ir014_recall']:.6f}")
    print(f"  IR014 F1: {metrics['ir014_f1']:.6f}")
    print(f"  IR014 -> IR007 count: {metrics['ir014_to_ir007']}")
    print(f"  IR007 precision: {metrics['ir007_precision']:.6f}")
    print(f"  IR007 recall: {metrics['ir007_recall']:.6f}")


def main() -> None:
    processed_data = np.load(PROCESSED_PATH, allow_pickle=False)
    splits = np.load(SPLIT_PATH, allow_pickle=False)
    train_source = splits["train_source_indices"].astype(np.int64)
    val_source = splits["val_source_indices"].astype(np.int64)
    train_target = splits["train_target_unlabeled_indices"].astype(np.int64)
    test_target = splits["test_target_indices"].astype(np.int64)

    print("Split check:")
    for name, left, right in [
        ("train_source_indices vs test_target_indices", train_source, test_target),
        ("val_source_indices vs test_target_indices", val_source, test_target),
        ("train_target_unlabeled_indices vs test_target_indices", train_target, test_target),
        ("train_source_indices vs train_target_unlabeled_indices", train_source, train_target),
    ]:
        count = overlap_count(left, right)
        print(f"  {name} overlap: {count}")
        assert count == 0, f"{name} overlap must be 0"

    test_loads = processed_data["load_hp"][test_target]
    print(f"  test_target samples: {len(test_target)}")
    print(f"  test_target load_hp distribution: {distribution(test_loads)}")
    assert len(test_target) == 927
    assert distribution(test_loads) == {2: 927}

    print("Prediction check:")
    best = check_predictions("best", EXPERIMENT_DIR / "logs/cross_load_predictions_best.csv", test_target, processed_data)
    final = check_predictions("final", EXPERIMENT_DIR / "logs/cross_load_predictions_final.csv", test_target, processed_data)
    print("  best prediction csv passed.")
    print("  final prediction csv passed.")
    print_metrics("best", best)
    print_metrics("final", final)
    print("Train-stat-norm Deep CORAL audit passed.")


if __name__ == "__main__":
    main()

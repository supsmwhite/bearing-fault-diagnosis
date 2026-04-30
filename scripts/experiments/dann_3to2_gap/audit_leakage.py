"""Audit possible leakage for the unusually high DANN gap-split final checkpoint result."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CONFIG_PATH = PROJECT_ROOT / "configs/experiments/dann_3to2_gap.yaml"
TRAIN_SCRIPT_PATH = PROJECT_ROOT / "scripts/experiments/dann_3to2_gap/train.py"
TRAIN_CORE_PATH = PROJECT_ROOT / "src/train/train_dann.py"
AUDIT_PATH = PROJECT_ROOT / "outputs/experiments/dann_3to2_gap/logs/leakage_audit.txt"


def append(lines: list, text: str = "") -> None:
    print(text)
    lines.append(text)


def distribution(values: np.ndarray) -> dict:
    return dict(Counter(values.tolist()))


def summarize_split(lines: list, name: str, indices: np.ndarray, load_hp: np.ndarray, y: np.ndarray) -> None:
    append(lines, f"{name}: {len(indices)}")
    append(lines, f"  load distribution: {distribution(load_hp[indices])}")
    append(lines, f"  label distribution: {distribution(y[indices])}")


def overlap_count(left: np.ndarray, right: np.ndarray) -> int:
    return len(set(left.tolist()) & set(right.tolist()))


def label_name_map(df: pd.DataFrame) -> dict:
    return {
        int(row["y_true"]): str(row["class_name"])
        for _, row in df[["y_true", "class_name"]].drop_duplicates().iterrows()
    }


def analyze_predictions(lines: list, tag: str, path: Path, test_indices: np.ndarray) -> bool:
    df = pd.read_csv(path)
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    names = label_name_map(df)
    sample_indices = set(df["sample_index"].astype(int).tolist())
    test_index_set = set(test_indices.tolist())
    all_in_test = sample_indices <= test_index_set

    append(lines, f"{tag} predictions:")
    append(lines, f"  rows: {len(df)}")
    append(lines, f"  accuracy: {accuracy_score(y_true, y_pred):.6f}")
    append(lines, f"  macro-F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.6f}")
    append(lines, f"  y_true label distribution: {distribution(df['y_true'].to_numpy())}")
    append(lines, f"  y_pred label distribution: {distribution(df['y_pred'].to_numpy())}")
    wrong_df = df[df["y_true"] != df["y_pred"]]
    append(lines, f"  wrong sample count: {len(wrong_df)}")
    append(lines, f"  sample_index all from test_target_indices: {all_in_test}")

    for label in sorted(df["y_true"].unique().tolist()):
        class_df = df[df["y_true"] == label]
        correct = int((class_df["y_true"] == class_df["y_pred"]).sum())
        wrong = int(len(class_df) - correct)
        append(lines, f"  {names.get(int(label), str(label))}: correct={correct}, wrong={wrong}")

    append(lines, "  first 20 wrong samples:")
    if wrong_df.empty:
        append(lines, "    None")
    else:
        for _, row in wrong_df.head(20).iterrows():
            append(
                lines,
                "    "
                f"sample_index={int(row['sample_index'])}, "
                f"y_true={int(row['y_true'])}, "
                f"y_pred={int(row['y_pred'])}, "
                f"class_name={row['class_name']}, "
                f"file_name={row['file_name']}, "
                f"start_index={int(row['start_index'])}",
            )
    append(lines)
    return all_in_test


def find_suspicious_lines(path: Path) -> list:
    suspicious_patterns = [
        "class_criterion(target",
        "target_class_logits",
    ]
    lines = path.read_text(encoding="utf-8").splitlines()
    findings = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if any(pattern in stripped for pattern in suspicious_patterns):
            findings.append((line_number, line))
        if "target_y" in stripped and "class" in stripped.lower():
            findings.append((line_number, line))
    return findings


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    processed_path = PROJECT_ROOT / config["data"]["processed_npz"]
    split_path = PROJECT_ROOT / config["data"]["split_npz"]
    experiment_dir = PROJECT_ROOT / config["outputs"]["experiment_dir"]
    best_predictions = experiment_dir / "logs/cross_load_predictions_best.csv"
    final_predictions = experiment_dir / "logs/cross_load_predictions_final.csv"

    data = np.load(processed_path, allow_pickle=False)
    splits = np.load(split_path, allow_pickle=False)
    y = data["y"]
    load_hp = data["load_hp"]

    train_source = splits["train_source_indices"].astype(np.int64)
    val_source = splits["val_source_indices"].astype(np.int64)
    train_target = splits["train_target_unlabeled_indices"].astype(np.int64)
    test_target = splits["test_target_indices"].astype(np.int64)
    discarded_gap = splits["discarded_gap_indices"].astype(np.int64)

    lines = []
    append(lines, "DANN gap leakage audit")
    append(lines)
    append(lines, "Split checks:")
    summarize_split(lines, "train_source_indices", train_source, load_hp, y)
    summarize_split(lines, "val_source_indices", val_source, load_hp, y)
    summarize_split(lines, "train_target_unlabeled_indices", train_target, load_hp, y)
    summarize_split(lines, "test_target_indices", test_target, load_hp, y)
    append(lines, f"discarded_gap_indices: {len(discarded_gap)}")

    overlap_train_target_test = overlap_count(train_target, test_target)
    overlap_train_source_test = overlap_count(train_source, test_target)
    overlap_val_source_test = overlap_count(val_source, test_target)
    overlap_gap_test = overlap_count(discarded_gap, test_target)

    append(lines, f"train_target_unlabeled_indices vs test_target_indices overlap: {overlap_train_target_test}")
    append(lines, f"train_source_indices vs test_target_indices overlap: {overlap_train_source_test}")
    append(lines, f"val_source_indices vs test_target_indices overlap: {overlap_val_source_test}")
    append(lines, f"discarded_gap_indices vs test_target_indices overlap: {overlap_gap_test}")
    append(lines)

    best_all_in_test = analyze_predictions(lines, "best", best_predictions, test_target)
    final_all_in_test = analyze_predictions(lines, "final", final_predictions, test_target)

    append(lines, "Training script text audit:")
    findings = []
    for path in [TRAIN_SCRIPT_PATH, TRAIN_CORE_PATH]:
        path_findings = find_suspicious_lines(path)
        if path_findings:
            append(lines, f"  Suspicious lines in {path}:")
            for line_number, line in path_findings:
                append(lines, f"    {line_number}: {line}")
            findings.extend(path_findings)
        else:
            append(lines, f"  No suspicious class-loss target usage found in {path}.")
    append(lines)

    test_only_load_2 = set(load_hp[test_target].tolist()) == {2}
    no_split_overlap = all(
        value == 0
        for value in [
            overlap_train_target_test,
            overlap_train_source_test,
            overlap_val_source_test,
            overlap_gap_test,
        ]
    )

    if test_only_load_2 and best_all_in_test and final_all_in_test and no_split_overlap:
        append(lines, "No split overlap detected.")
    else:
        append(lines, "Potential leakage detected.")

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    append([], f"Audit path: {AUDIT_PATH}")


if __name__ == "__main__":
    main()


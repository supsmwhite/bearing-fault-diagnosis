"""Analyze DANN cross-load prediction errors for best and final checkpoints."""

from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


EXPERIMENT_DIR = Path("outputs/experiments/dann_3to2")


def build_label_name_map(df: pd.DataFrame) -> dict:
    mapping = {}
    for _, row in df[["y_true", "class_name"]].drop_duplicates().iterrows():
        mapping[int(row["y_true"])] = str(row["class_name"])
    return mapping


def get_most_confused_with(df: pd.DataFrame, true_label: int, label_name_map: dict) -> str:
    wrong_preds = df[(df["y_true"] == true_label) & (df["y_pred"] != true_label)]
    if wrong_preds.empty:
        return ""
    confused_label, _ = Counter(wrong_preds["y_pred"].astype(int).tolist()).most_common(1)[0]
    return label_name_map.get(confused_label, str(confused_label))


def analyze_tag(tag: str) -> None:
    predictions_path = EXPERIMENT_DIR / f"logs/cross_load_predictions_{tag}.csv"
    analysis_path = EXPERIMENT_DIR / f"logs/cross_load_error_analysis_{tag}.txt"
    per_class_path = EXPERIMENT_DIR / f"logs/cross_load_per_class_metrics_{tag}.csv"

    df = pd.read_csv(predictions_path)
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    labels = sorted(df["y_true"].unique().tolist())
    label_name_map = build_label_name_map(df)

    overall_accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    rows = []
    for index, label in enumerate(labels):
        class_rows = df[df["y_true"] == label]
        correct = int((class_rows["y_true"] == class_rows["y_pred"]).sum())
        wrong = int(len(class_rows) - correct)
        rows.append(
            {
                "class_name": label_name_map.get(label, str(label)),
                "label": int(label),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "correct": correct,
                "wrong": wrong,
                "most_confused_with": get_most_confused_with(df, label, label_name_map),
            }
        )

    pd.DataFrame(rows).to_csv(per_class_path, index=False, encoding="utf-8")

    lines = [
        f"DANN {tag} cross-load error analysis",
        "",
        f"overall accuracy: {overall_accuracy:.6f}",
        f"macro-F1: {macro_f1:.6f}",
        "",
        "per-class metrics:",
    ]
    for row in rows:
        lines.append(
            "  "
            f"{row['class_name']} (label={row['label']}): "
            f"precision={row['precision']:.6f}, "
            f"recall={row['recall']:.6f}, "
            f"f1={row['f1']:.6f}, "
            f"support={row['support']}, "
            f"correct={row['correct']}, "
            f"wrong={row['wrong']}, "
            f"most_confused_with={row['most_confused_with'] or 'None'}"
        )
    analysis_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"DANN {tag}:")
    print(f"  overall accuracy: {overall_accuracy:.6f}")
    print(f"  macro-F1: {macro_f1:.6f}")
    print(f"  analysis path: {analysis_path}")
    print(f"  per-class metrics path: {per_class_path}")


def main() -> None:
    analyze_tag("best")
    analyze_tag("final")


if __name__ == "__main__":
    main()


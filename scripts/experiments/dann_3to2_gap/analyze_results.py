"""Analyze DANN gap-split cross-load prediction errors for best and final checkpoints."""

from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


CONFIG_PATH = Path("configs/experiments/dann_3to2_gap.yaml")


def analyze_tag(experiment_dir: Path, tag: str) -> None:
    predictions_path = experiment_dir / f"logs/cross_load_predictions_{tag}.csv"
    analysis_path = experiment_dir / f"logs/cross_load_error_analysis_{tag}.txt"
    per_class_path = experiment_dir / f"logs/cross_load_per_class_metrics_{tag}.csv"
    df = pd.read_csv(predictions_path)
    labels = sorted(df["y_true"].unique().tolist())
    label_names = {int(row["y_true"]): str(row["class_name"]) for _, row in df[["y_true", "class_name"]].drop_duplicates().iterrows()}
    precision, recall, f1, support = precision_recall_fscore_support(df["y_true"], df["y_pred"], labels=labels, zero_division=0)
    rows = []
    for i, label in enumerate(labels):
        class_rows = df[df["y_true"] == label]
        wrong_rows = class_rows[class_rows["y_pred"] != label]
        most_confused = ""
        if not wrong_rows.empty:
            confused_label, _ = Counter(wrong_rows["y_pred"].astype(int).tolist()).most_common(1)[0]
            most_confused = label_names.get(confused_label, str(confused_label))
        rows.append(
            {
                "class_name": label_names.get(label, str(label)),
                "label": int(label),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "correct": int((class_rows["y_true"] == class_rows["y_pred"]).sum()),
                "wrong": int(len(wrong_rows)),
                "most_confused_with": most_confused,
            }
        )
    pd.DataFrame(rows).to_csv(per_class_path, index=False, encoding="utf-8")
    text = [
        f"DANN gap {tag} cross-load error analysis",
        "",
        f"overall accuracy: {accuracy_score(df['y_true'], df['y_pred']):.6f}",
        f"macro-F1: {f1_score(df['y_true'], df['y_pred'], average='macro', zero_division=0):.6f}",
        "",
        "per-class metrics:",
    ]
    for row in rows:
        text.append(
            f"  {row['class_name']} (label={row['label']}): "
            f"precision={row['precision']:.6f}, recall={row['recall']:.6f}, "
            f"f1={row['f1']:.6f}, support={row['support']}, "
            f"correct={row['correct']}, wrong={row['wrong']}, "
            f"most_confused_with={row['most_confused_with'] or 'None'}"
        )
    analysis_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"DANN gap {tag} analysis path: {analysis_path}")
    print(f"DANN gap {tag} per-class metrics path: {per_class_path}")


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    experiment_dir = Path(config["outputs"]["experiment_dir"])
    analyze_tag(experiment_dir, "best")
    analyze_tag(experiment_dir, "final")


if __name__ == "__main__":
    main()


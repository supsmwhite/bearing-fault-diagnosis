"""Run Deep CORAL normalization ablation for IR014 confusion."""

import copy
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CWRUWindowDataset  # noqa: E402
from src.models.deep_coral import DeepCORAL1D  # noqa: E402
from src.train.train_deep_coral import evaluate_source_deep_coral  # noqa: E402
from src.train.train_deep_coral import train_one_epoch_deep_coral  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


PROCESSED_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_windows_no_window_zscore.npz"
SPLIT_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_splits_3to2.npz"
BASELINE_PREDICTIONS_PATH = PROJECT_ROOT / "outputs/experiments/deep_coral_3to2/logs/cross_load_predictions_best.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/deep_coral_3to2_no_window_zscore"
REPORT_PATH = OUTPUT_DIR / "logs/ir014_ablation_summary.txt"

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
CORAL_LOSS_WEIGHT = 1.0
SEED = 42
NUM_CLASSES = 10
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


def compute_class_weights(y: np.ndarray, indices: np.ndarray, num_classes: int) -> torch.Tensor:
    labels = y[indices]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    if np.any(counts == 0):
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Missing classes in train_source labels: {missing}")
    weights = counts.sum() / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32)


def read_baseline_predictions(path: Path) -> tuple:
    y_true = []
    y_pred = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            y_true.append(int(row["y_true"]))
            y_pred.append(int(row["y_pred"]))
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


@torch.no_grad()
def predict(model, loader, device) -> tuple:
    model.eval()
    y_true_batches = []
    y_pred_batches = []
    for x, y in loader:
        x = x.to(device)
        class_logits, _features = model(x)
        y_pred_batches.append(torch.argmax(class_logits, dim=1).cpu().numpy())
        y_true_batches.append(y.cpu().numpy())
    return (
        np.concatenate(y_true_batches, axis=0),
        np.concatenate(y_pred_batches, axis=0),
    )


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        zero_division=0,
    )
    ir014_to_ir007 = int(((y_true == IR014_LABEL) & (y_pred == IR007_LABEL)).sum())
    return {
        "target_accuracy": float(accuracy_score(y_true, y_pred)),
        "target_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ir014_precision": float(precision[IR014_LABEL]),
        "ir014_recall": float(recall[IR014_LABEL]),
        "ir014_f1": float(f1[IR014_LABEL]),
        "ir014_support": int(support[IR014_LABEL]),
        "ir014_to_ir007_count": ir014_to_ir007,
        "ir007_precision": float(precision[IR007_LABEL]),
        "ir007_recall": float(recall[IR007_LABEL]),
        "ir007_support": int(support[IR007_LABEL]),
    }


def add_summary(lines: list, title: str, summary: dict) -> None:
    lines.extend(
        [
            f"{title}:",
            f"  target accuracy: {summary['target_accuracy']:.6f}",
            f"  target macro-F1: {summary['target_macro_f1']:.6f}",
            f"  IR014 precision: {summary['ir014_precision']:.6f}",
            f"  IR014 recall: {summary['ir014_recall']:.6f}",
            f"  IR014 F1: {summary['ir014_f1']:.6f}",
            f"  IR014 -> IR007 count: {summary['ir014_to_ir007_count']}",
            f"  IR007 precision: {summary['ir007_precision']:.6f}",
            f"  IR007 recall: {summary['ir007_recall']:.6f}",
        ]
    )


def main() -> None:
    set_seed(SEED)
    processed = np.load(PROCESSED_PATH, allow_pickle=False)
    splits = np.load(SPLIT_PATH, allow_pickle=False)

    source_train_indices = splits["train_source_indices"]
    source_val_indices = splits["val_source_indices"]
    target_train_indices = splits["train_target_unlabeled_indices"]
    test_target_indices = splits["test_target_indices"]

    source_train_dataset = CWRUWindowDataset(str(PROCESSED_PATH), indices=source_train_indices)
    source_val_dataset = CWRUWindowDataset(str(PROCESSED_PATH), indices=source_val_indices)
    target_train_dataset = CWRUWindowDataset(str(PROCESSED_PATH), indices=target_train_indices)
    test_target_dataset = CWRUWindowDataset(str(PROCESSED_PATH), indices=test_target_indices)

    source_train_loader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    source_val_loader = DataLoader(source_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    target_train_loader = DataLoader(target_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_target_loader = DataLoader(test_target_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepCORAL1D(num_classes=NUM_CLASSES).to(device)
    class_weights = compute_class_weights(processed["y"], source_train_indices, NUM_CLASSES).to(device)
    class_criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_epoch = 0
    best_val_macro_f1 = -1.0
    best_state_dict = None

    for epoch in range(1, EPOCHS + 1):
        total_loss, class_loss, coral_loss, train_acc, train_f1 = train_one_epoch_deep_coral(
            model,
            source_train_loader,
            target_train_loader,
            class_criterion,
            optimizer,
            device,
            coral_loss_weight=CORAL_LOSS_WEIGHT,
        )
        val_loss, val_acc, val_f1 = evaluate_source_deep_coral(
            model,
            source_val_loader,
            class_criterion,
            device,
        )
        print(
            f"Epoch {epoch:03d}/{EPOCHS:03d} | "
            f"total_loss={total_loss:.6f} class_loss={class_loss:.6f} "
            f"coral_loss={coral_loss:.6f} train_acc={train_acc:.6f} "
            f"train_f1={train_f1:.6f} | val_loss={val_loss:.6f} "
            f"val_acc={val_acc:.6f} val_f1={val_f1:.6f}"
        )
        if val_f1 > best_val_macro_f1:
            best_epoch = epoch
            best_val_macro_f1 = val_f1
            best_state_dict = copy.deepcopy(model.state_dict())

    final_y_true, final_y_pred = predict(model, test_target_loader, device)
    final_summary = summarize_predictions(final_y_true, final_y_pred)

    best_model = DeepCORAL1D(num_classes=NUM_CLASSES).to(device)
    best_model.load_state_dict(best_state_dict)
    best_y_true, best_y_pred = predict(best_model, test_target_loader, device)
    best_summary = summarize_predictions(best_y_true, best_y_pred)

    baseline_y_true, baseline_y_pred = read_baseline_predictions(BASELINE_PREDICTIONS_PATH)
    baseline_summary = summarize_predictions(baseline_y_true, baseline_y_pred)

    recall_gain = best_summary["ir014_recall"] - baseline_summary["ir014_recall"]
    f1_gain = best_summary["ir014_f1"] - baseline_summary["ir014_f1"]
    confusion_reduction = (
        baseline_summary["ir014_to_ir007_count"] - best_summary["ir014_to_ir007_count"]
    )
    if recall_gain >= 0.10 and f1_gain >= 0.10 and confusion_reduction >= 10:
        conclusion = (
            "new normalization clearly improves IR014 and reduces IR014->IR007 confusion; "
            "per-window z-score is likely part of the bottleneck."
        )
    elif recall_gain > 0 and confusion_reduction > 0:
        conclusion = (
            "new normalization gives partial IR014 improvement, but evidence is moderate."
        )
    else:
        conclusion = (
            "new normalization does not materially improve IR014; next step should consider "
            "harder transfer or backbone changes."
        )

    lines = [
        "Ablation: normalization vs IR014 confusion",
        "",
        "Normalization scheme: file-level z-score before windowing; no per-window z-score.",
        f"Processed data: {PROCESSED_PATH}",
        f"Split data: {SPLIT_PATH}",
        f"Best epoch: {best_epoch}",
        f"Best source-val macro-F1: {best_val_macro_f1:.6f}",
        "",
    ]
    add_summary(lines, "Baseline Deep CORAL best with per-window z-score", baseline_summary)
    lines.append("")
    add_summary(lines, "New Deep CORAL best without per-window z-score", best_summary)
    lines.append("")
    add_summary(lines, "New Deep CORAL final without per-window z-score", final_summary)
    lines.extend(
        [
            "",
            f"baseline IR014 recall: {baseline_summary['ir014_recall']:.6f}",
            f"new IR014 recall: {best_summary['ir014_recall']:.6f}",
            f"baseline IR014 -> IR007 count: {baseline_summary['ir014_to_ir007_count']}",
            f"new IR014 -> IR007 count: {best_summary['ir014_to_ir007_count']}",
            f"conclusion: {conclusion}",
            f"New best prediction distribution: {dict(Counter(best_y_pred.tolist()))}",
            f"New final prediction distribution: {dict(Counter(final_y_pred.tolist()))}",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for line in lines:
        print(line)
    print(f"Report path: {REPORT_PATH}")


if __name__ == "__main__":
    main()

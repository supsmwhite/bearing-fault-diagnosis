"""Evaluate no-window-zscore Deep CORAL checkpoints and save predictions."""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CWRUWindowDataset  # noqa: E402
from src.models.deep_coral import DeepCORAL1D  # noqa: E402


PROCESSED_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_windows_no_window_zscore.npz"
SPLIT_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_splits_3to2.npz"
EXPERIMENT_DIR = PROJECT_ROOT / "outputs/experiments/deep_coral_3to2_no_window_zscore"
BEST_CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints/deep_coral_best.pt"
FINAL_CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints/deep_coral_final.pt"
BATCH_SIZE = 64
NUM_CLASSES = 10
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


def safe_load_checkpoint(checkpoint_path: Path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}. "
            "Cannot generate per-sample prediction csv without a saved model state."
        )
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        print(
            "weights_only=True failed, falling back to weights_only=False "
            "for trusted local checkpoint."
        )
        return torch.load(checkpoint_path, map_location=device, weights_only=False)


@torch.no_grad()
def evaluate_model(model, dataloader, device) -> dict:
    model.eval()
    y_true_batches = []
    y_pred_batches = []

    for x, y in dataloader:
        x = x.to(device)
        class_logits, _features = model(x)
        y_pred_batches.append(torch.argmax(class_logits, dim=1).cpu().numpy())
        y_true_batches.append(y.cpu().numpy())

    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_predictions(path: Path, sample_indices: np.ndarray, metrics: dict, processed_data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_index",
                "y_true",
                "y_pred",
                "load_hp",
                "class_name",
                "file_name",
                "start_index",
            ],
        )
        writer.writeheader()
        for row_id, sample_index in enumerate(sample_indices):
            writer.writerow(
                {
                    "sample_index": int(sample_index),
                    "y_true": int(metrics["y_true"][row_id]),
                    "y_pred": int(metrics["y_pred"][row_id]),
                    "load_hp": int(processed_data["load_hp"][sample_index]),
                    "class_name": str(processed_data["class_name"][sample_index]),
                    "file_name": str(processed_data["file_name"][sample_index]),
                    "start_index": int(processed_data["start_index"][sample_index]),
                }
            )


def save_metrics(path: Path, checkpoint_path: Path, checkpoint: dict, metrics: dict) -> None:
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=np.arange(NUM_CLASSES))
    report = classification_report(
        metrics["y_true"],
        metrics["y_pred"],
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(f"checkpoint_path: {checkpoint_path}\n")
        file.write(f"checkpoint_epoch: {checkpoint.get('epoch')}\n")
        file.write(f"checkpoint_best_val_macro_f1: {checkpoint.get('best_val_macro_f1')}\n")
        file.write(f"target_accuracy: {metrics['accuracy']:.6f}\n")
        file.write(f"target_macro_f1: {metrics['macro_f1']:.6f}\n")
        file.write("\nconfusion_matrix:\n")
        file.write(np.array2string(cm))
        file.write("\n\nclassification_report:\n")
        file.write(report)


def evaluate_checkpoint(
    tag: str,
    checkpoint_path: Path,
    test_loader,
    test_target_indices: np.ndarray,
    processed_data,
    device,
) -> dict:
    checkpoint = safe_load_checkpoint(checkpoint_path, device)
    model = DeepCORAL1D(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_model(model, test_loader, device)
    predictions_path = EXPERIMENT_DIR / f"logs/cross_load_predictions_{tag}.csv"
    metrics_path = EXPERIMENT_DIR / f"logs/cross_load_test_metrics_{tag}.txt"
    save_predictions(predictions_path, test_target_indices, metrics, processed_data)
    save_metrics(metrics_path, checkpoint_path, checkpoint, metrics)

    print(f"Deep CORAL no-window-zscore {tag}:")
    print(f"  target accuracy: {metrics['accuracy']:.6f}")
    print(f"  target macro-F1: {metrics['macro_f1']:.6f}")
    print(f"  predictions path: {predictions_path}")
    print(f"  metrics path: {metrics_path}")

    return metrics


def main() -> None:
    processed_data = np.load(PROCESSED_PATH, allow_pickle=False)
    splits = np.load(SPLIT_PATH, allow_pickle=False)
    test_target_indices = splits["test_target_indices"]

    test_dataset = CWRUWindowDataset(
        str(PROCESSED_PATH),
        indices=test_target_indices,
        return_meta=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_checkpoint(
        "best",
        BEST_CHECKPOINT_PATH,
        test_loader,
        test_target_indices,
        processed_data,
        device,
    )
    evaluate_checkpoint(
        "final",
        FINAL_CHECKPOINT_PATH,
        test_loader,
        test_target_indices,
        processed_data,
        device,
    )


if __name__ == "__main__":
    main()

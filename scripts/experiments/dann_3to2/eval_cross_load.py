"""Evaluate trained DANN checkpoints on the target 2 hp test split."""

import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CWRUWindowDataset  # noqa: E402
from src.models.dann import DANN  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/experiments/dann_3to2.yaml"
BATCH_SIZE = 64
CNN1D_TARGET_ACCURACY = 0.816613
CNN1D_TARGET_MACRO_F1 = 0.686022
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


def safe_load_checkpoint(checkpoint_path, device):
    """
    Load a trusted local checkpoint with compatibility for PyTorch weights_only behavior.
    """
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except Exception:
        print(
            "weights_only=True failed, falling back to weights_only=False "
            "for trusted local checkpoint."
        )
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )


@torch.no_grad()
def evaluate_dann(model, dataloader, device):
    model.eval()
    y_true_batches = []
    y_pred_batches = []

    for x, y in dataloader:
        x = x.to(device)
        class_logits, _domain_logits = model(x)
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


def save_confusion_matrix(path: Path, cm: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(ticks, CLASS_NAMES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm[row, col] > threshold else "black"
            plt.text(col, row, str(cm[row, col]), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_metrics(path: Path, checkpoint_path: Path, checkpoint: dict, metrics: dict, cm: np.ndarray, report: str) -> None:
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
    config: dict,
    test_loader,
    test_target_indices: np.ndarray,
    processed_data,
    device,
) -> dict:
    experiment_dir = PROJECT_ROOT / config["outputs"]["experiment_dir"]
    model = DANN(
        num_classes=config["experiment"]["num_classes"],
        grl_lambda=config["training"]["grl_lambda"],
    ).to(device)
    checkpoint = safe_load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_dann(model, test_loader, device)
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=np.arange(len(CLASS_NAMES)))
    report = classification_report(
        metrics["y_true"],
        metrics["y_pred"],
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    metrics_path = experiment_dir / f"logs/cross_load_test_metrics_{tag}.txt"
    predictions_path = experiment_dir / f"logs/cross_load_predictions_{tag}.csv"
    confusion_matrix_path = experiment_dir / f"figures/cross_load_confusion_matrix_{tag}.png"

    save_predictions(predictions_path, test_target_indices, metrics, processed_data)
    save_confusion_matrix(confusion_matrix_path, cm, f"DANN {tag} Cross-load Confusion Matrix")
    save_metrics(metrics_path, checkpoint_path, checkpoint, metrics, cm, report)

    test_loads = processed_data["load_hp"][test_target_indices]
    test_labels = processed_data["y"][test_target_indices]

    print(f"DANN {tag}:")
    print(f"  checkpoint path: {checkpoint_path}")
    print("  test split name: test_target_indices")
    print(f"  test samples: {len(test_target_indices)}")
    print(f"  test load distribution: {dict(Counter(test_loads.tolist()))}")
    print(f"  test label distribution: {dict(Counter(test_labels.tolist()))}")
    print(f"  target accuracy: {metrics['accuracy']:.6f}")
    print(f"  target macro-F1: {metrics['macro_f1']:.6f}")
    print(f"  metrics path: {metrics_path}")
    print(f"  predictions path: {predictions_path}")
    print(f"  confusion matrix path: {confusion_matrix_path}")

    return metrics


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    processed_path = PROJECT_ROOT / config["data"]["processed_npz"]
    split_path = PROJECT_ROOT / config["data"]["split_npz"]
    processed_data = np.load(processed_path, allow_pickle=False)
    splits = np.load(split_path, allow_pickle=False)
    test_target_indices = splits[config["data"]["target_test_split"]]

    test_dataset = CWRUWindowDataset(
        str(processed_path),
        indices=test_target_indices,
        return_meta=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_metrics = evaluate_checkpoint(
        "best",
        PROJECT_ROOT / config["outputs"]["checkpoint"],
        config,
        test_loader,
        test_target_indices,
        processed_data,
        device,
    )
    final_metrics = evaluate_checkpoint(
        "final",
        PROJECT_ROOT / config["outputs"]["final_checkpoint"],
        config,
        test_loader,
        test_target_indices,
        processed_data,
        device,
    )

    print("Comparison:")
    print("CNN1D source-only:")
    print(f"  target_accuracy = {CNN1D_TARGET_ACCURACY:.6f}")
    print(f"  target_macro_f1 = {CNN1D_TARGET_MACRO_F1:.6f}")
    print("DANN best:")
    print(f"  target_accuracy = {best_metrics['accuracy']:.6f}")
    print(f"  target_macro_f1 = {best_metrics['macro_f1']:.6f}")
    print("DANN final:")
    print(f"  target_accuracy = {final_metrics['accuracy']:.6f}")
    print(f"  target_macro_f1 = {final_metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()


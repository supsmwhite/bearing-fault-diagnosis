"""Evaluate DANN gap-split checkpoints on the target 2 hp test split."""

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


CONFIG_PATH = PROJECT_ROOT / "configs/experiments/dann_3to2_gap.yaml"
CLASS_NAMES = ["Normal", "IR007", "B007", "OR007@6", "IR014", "B014", "OR014@6", "IR021", "B021", "OR021@6"]


def safe_load_checkpoint(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        print("weights_only=True failed, falling back to weights_only=False for trusted local checkpoint.")
        return torch.load(path, map_location=device, weights_only=False)


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    y_true_parts = []
    y_pred_parts = []
    for x, y in loader:
        x = x.to(device)
        class_logits, _domain_logits = model(x)
        y_true_parts.append(y.cpu().numpy())
        y_pred_parts.append(torch.argmax(class_logits, dim=1).cpu().numpy())
    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_predictions(path: Path, sample_indices: np.ndarray, metrics: dict, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_index", "y_true", "y_pred", "load_hp", "class_name", "file_name", "start_index"])
        writer.writeheader()
        for row_id, sample_index in enumerate(sample_indices):
            writer.writerow(
                {
                    "sample_index": int(sample_index),
                    "y_true": int(metrics["y_true"][row_id]),
                    "y_pred": int(metrics["y_pred"][row_id]),
                    "load_hp": int(data["load_hp"][sample_index]),
                    "class_name": str(data["class_name"][sample_index]),
                    "file_name": str(data["file_name"][sample_index]),
                    "start_index": int(data["start_index"][sample_index]),
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
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def evaluate_checkpoint(tag: str, checkpoint_path: Path, config: dict, loader, test_indices: np.ndarray, data, device) -> dict:
    experiment_dir = PROJECT_ROOT / config["outputs"]["experiment_dir"]
    model = DANN(
        num_classes=config["experiment"]["num_classes"],
        grl_lambda=config["training"]["grl_lambda"],
    ).to(device)
    checkpoint = safe_load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_model(model, loader, device)
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=np.arange(len(CLASS_NAMES)))
    report = classification_report(metrics["y_true"], metrics["y_pred"], labels=np.arange(len(CLASS_NAMES)), target_names=CLASS_NAMES, zero_division=0)
    metrics_path = experiment_dir / f"logs/cross_load_test_metrics_{tag}.txt"
    predictions_path = experiment_dir / f"logs/cross_load_predictions_{tag}.csv"
    figure_path = experiment_dir / f"figures/cross_load_confusion_matrix_{tag}.png"
    save_predictions(predictions_path, test_indices, metrics, data)
    save_confusion_matrix(figure_path, cm, f"DANN gap {tag} confusion matrix")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        f"checkpoint_path: {checkpoint_path}\n"
        f"checkpoint_epoch: {checkpoint.get('epoch')}\n"
        f"checkpoint_best_val_macro_f1: {checkpoint.get('best_val_macro_f1')}\n"
        f"target_accuracy: {metrics['accuracy']:.6f}\n"
        f"target_macro_f1: {metrics['macro_f1']:.6f}\n\n"
        f"confusion_matrix:\n{np.array2string(cm)}\n\nclassification_report:\n{report}",
        encoding="utf-8",
    )
    print(f"DANN gap {tag}:")
    print(f"  checkpoint path: {checkpoint_path}")
    print(f"  test samples: {len(test_indices)}")
    print(f"  test load distribution: {dict(Counter(data['load_hp'][test_indices].tolist()))}")
    print(f"  test label distribution: {dict(Counter(data['y'][test_indices].tolist()))}")
    print(f"  target accuracy: {metrics['accuracy']:.6f}")
    print(f"  target macro-F1: {metrics['macro_f1']:.6f}")
    print(f"  metrics path: {metrics_path}")
    print(f"  predictions path: {predictions_path}")
    print(f"  confusion matrix path: {figure_path}")
    return metrics


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data = np.load(PROJECT_ROOT / config["data"]["processed_npz"], allow_pickle=False)
    splits = np.load(PROJECT_ROOT / config["data"]["split_npz"], allow_pickle=False)
    test_indices = splits[config["data"]["target_test_split"]]
    loader = DataLoader(
        CWRUWindowDataset(str(PROJECT_ROOT / config["data"]["processed_npz"]), indices=test_indices),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best = evaluate_checkpoint("best", PROJECT_ROOT / config["outputs"]["checkpoint"], config, loader, test_indices, data, device)
    final = evaluate_checkpoint("final", PROJECT_ROOT / config["outputs"]["final_checkpoint"], config, loader, test_indices, data, device)
    print("Comparison:")
    print(f"  DANN gap best target_accuracy = {best['accuracy']:.6f}")
    print(f"  DANN gap best target_macro_f1 = {best['macro_f1']:.6f}")
    print(f"  DANN gap final target_accuracy = {final['accuracy']:.6f}")
    print(f"  DANN gap final target_macro_f1 = {final['macro_f1']:.6f}")


if __name__ == "__main__":
    main()


"""Organize CNN1D source-only outputs and plot loss, accuracy, and macro-F1 curves from the training log."""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_DIR = Path("outputs/experiments/cnn1d_source_3to2")
NEW_CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints/cnn1d_source_best.pt"
NEW_LOG_PATH = EXPERIMENT_DIR / "logs/train_log.csv"
FIGURE_DIR = EXPERIMENT_DIR / "figures"

OLD_CHECKPOINT_PATH = Path("outputs/checkpoints/cnn1d_source_best.pt")
OLD_LOG_PATH = Path("outputs/logs/cnn1d_source_train_log.csv")


def prepare_experiment_files() -> tuple:
    NEW_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    copied_old_log = False
    copied_old_checkpoint = False

    if NEW_LOG_PATH.exists():
        log_path = NEW_LOG_PATH
    elif OLD_LOG_PATH.exists():
        shutil.copy2(OLD_LOG_PATH, NEW_LOG_PATH)
        log_path = NEW_LOG_PATH
        copied_old_log = True
    else:
        raise FileNotFoundError(
            f"Training log not found: {NEW_LOG_PATH} or {OLD_LOG_PATH}"
        )

    if not NEW_CHECKPOINT_PATH.exists() and OLD_CHECKPOINT_PATH.exists():
        shutil.copy2(OLD_CHECKPOINT_PATH, NEW_CHECKPOINT_PATH)
        copied_old_checkpoint = True

    return log_path, copied_old_log, copied_old_checkpoint


def plot_curve(df: pd.DataFrame, y_columns: list, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for column in y_columns:
        plt.plot(df["epoch"], df[column], marker="o", label=column)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    log_path, copied_old_log, copied_old_checkpoint = prepare_experiment_files()
    df = pd.read_csv(log_path)

    loss_path = FIGURE_DIR / "loss_curve.png"
    accuracy_path = FIGURE_DIR / "accuracy_curve.png"
    macro_f1_path = FIGURE_DIR / "macro_f1_curve.png"

    plot_curve(df, ["train_loss", "val_loss"], "loss", loss_path)
    plot_curve(df, ["train_accuracy", "val_accuracy"], "accuracy", accuracy_path)
    plot_curve(df, ["train_macro_f1", "val_macro_f1"], "macro-F1", macro_f1_path)

    best_row = df.loc[df["val_macro_f1"].idxmax()]
    best_epoch = int(best_row["epoch"])
    best_val_macro_f1 = float(best_row["val_macro_f1"])

    print(f"Log path used: {log_path}")
    print(f"Copied old log: {copied_old_log}")
    print(f"Copied old checkpoint: {copied_old_checkpoint}")
    print(f"Loss curve path: {loss_path}")
    print(f"Accuracy curve path: {accuracy_path}")
    print(f"Macro-F1 curve path: {macro_f1_path}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val macro-F1: {best_val_macro_f1:.6f}")


if __name__ == "__main__":
    main()


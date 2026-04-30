"""Plot DANN training curves from the saved training log."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_DIR = Path("outputs/experiments/dann_3to2")
LOG_PATH = EXPERIMENT_DIR / "logs/train_log.csv"
FIGURE_DIR = EXPERIMENT_DIR / "figures"


def plot_curve(df: pd.DataFrame, columns: list, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for column in columns:
        plt.plot(df["epoch"], df[column], marker="o", label=column)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LOG_PATH)

    loss_path = FIGURE_DIR / "loss_curve.png"
    source_accuracy_path = FIGURE_DIR / "source_accuracy_curve.png"
    source_macro_f1_path = FIGURE_DIR / "source_macro_f1_curve.png"
    domain_accuracy_path = FIGURE_DIR / "domain_accuracy_curve.png"

    plot_curve(df, ["total_loss", "class_loss", "domain_loss", "val_loss"], "loss", loss_path)
    plot_curve(
        df,
        ["source_train_accuracy", "val_accuracy"],
        "accuracy",
        source_accuracy_path,
    )
    plot_curve(
        df,
        ["source_train_macro_f1", "val_macro_f1"],
        "macro-F1",
        source_macro_f1_path,
    )
    plot_curve(df, ["domain_accuracy"], "domain accuracy", domain_accuracy_path)

    print(f"Loss curve path: {loss_path}")
    print(f"Source accuracy curve path: {source_accuracy_path}")
    print(f"Source macro-F1 curve path: {source_macro_f1_path}")
    print(f"Domain accuracy curve path: {domain_accuracy_path}")


if __name__ == "__main__":
    main()


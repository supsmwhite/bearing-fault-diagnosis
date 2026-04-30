"""Plot CNN1D source-only gap-split training curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


CONFIG_PATH = Path("configs/experiments/cnn1d_source_3to2_gap.yaml")


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
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    experiment_dir = Path(config["outputs"]["experiment_dir"])
    figure_dir = experiment_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config["outputs"]["train_log"])
    plot_curve(df, ["train_loss", "val_loss"], "loss", figure_dir / "loss_curve.png")
    plot_curve(df, ["train_accuracy", "val_accuracy"], "accuracy", figure_dir / "accuracy_curve.png")
    plot_curve(df, ["train_macro_f1", "val_macro_f1"], "macro-F1", figure_dir / "macro_f1_curve.png")
    print(f"Figure dir: {figure_dir}")


if __name__ == "__main__":
    main()


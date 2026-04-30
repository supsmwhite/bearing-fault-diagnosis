"""Analyze why Deep CORAL best still confuses IR014 with IR007."""

import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PROCESSED_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_windows.npz"
SPLIT_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_splits_3to2.npz"
EXPERIMENT_DIR = PROJECT_ROOT / "outputs/experiments/deep_coral_3to2"
PREDICTIONS_PATH = EXPERIMENT_DIR / "logs/cross_load_predictions_best.csv"
REPORT_PATH = EXPERIMENT_DIR / "logs/ir014_hard_class_analysis.txt"
IR014_CORRECT_CSV = EXPERIMENT_DIR / "logs/ir014_correct_samples.csv"
IR014_TO_IR007_CSV = EXPERIMENT_DIR / "logs/ir014_misclassified_as_ir007_samples.csv"
PREDICTED_IR007_CSV = EXPERIMENT_DIR / "logs/predicted_ir007_samples.csv"
WAVEFORM_FIGURE_PATH = EXPERIMENT_DIR / "figures/ir014_correct_vs_ir007_wrong_waveform.png"
SPECTRUM_FIGURE_PATH = EXPERIMENT_DIR / "figures/ir014_ir007_average_spectrum.png"

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


def append(lines: list, text: str = "") -> None:
    print(text)
    lines.append(text)


def class_metric(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[label],
        zero_division=0,
    )
    return {
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
        "support": int(support[0]),
    }


def metadata_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "sample_index",
        "y_true",
        "y_pred",
        "load_hp",
        "class_name",
        "file_name",
        "start_index",
    ]
    return df[columns].copy()


def save_metadata_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_frame(df).to_csv(path, index=False, encoding="utf-8")


def sample_windows(data: np.lib.npyio.NpzFile, df: pd.DataFrame) -> np.ndarray:
    sample_indices = df["sample_index"].astype(int).to_numpy()
    return data["X"][sample_indices]


def plot_waveform(data: np.lib.npyio.NpzFile, correct_df: pd.DataFrame, wrong_df: pd.DataFrame) -> None:
    WAVEFORM_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))

    if not correct_df.empty:
        correct_mean = sample_windows(data, correct_df).mean(axis=0)
        plt.plot(correct_mean, label=f"IR014 correct mean (n={len(correct_df)})")
    if not wrong_df.empty:
        wrong_mean = sample_windows(data, wrong_df).mean(axis=0)
        plt.plot(wrong_mean, label=f"IR014 -> IR007 wrong mean (n={len(wrong_df)})")

    plt.title("IR014 Correct vs IR014->IR007 Wrong Mean Waveform")
    plt.xlabel("Window sample")
    plt.ylabel("Normalized amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(WAVEFORM_FIGURE_PATH, dpi=150)
    plt.close()


def mean_spectrum(data: np.lib.npyio.NpzFile, df: pd.DataFrame) -> np.ndarray:
    windows = sample_windows(data, df)
    spectra = np.abs(np.fft.rfft(windows, axis=1))
    return spectra.mean(axis=0)


def plot_spectrum(data: np.lib.npyio.NpzFile, ir014_df: pd.DataFrame, ir007_df: pd.DataFrame) -> None:
    SPECTRUM_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))

    if not ir014_df.empty:
        plt.plot(mean_spectrum(data, ir014_df), label=f"IR014 true mean spectrum (n={len(ir014_df)})")
    if not ir007_df.empty:
        plt.plot(mean_spectrum(data, ir007_df), label=f"IR007 true mean spectrum (n={len(ir007_df)})")

    plt.title("IR014 vs IR007 Mean Spectrum Magnitude")
    plt.xlabel("FFT bin")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SPECTRUM_FIGURE_PATH, dpi=150)
    plt.close()


def main() -> None:
    data = np.load(PROCESSED_PATH, allow_pickle=False)
    splits = np.load(SPLIT_PATH, allow_pickle=False)
    test_target_indices = splits["test_target_indices"].astype(np.int64)
    test_target_index_set = set(test_target_indices.tolist())

    df = pd.read_csv(PREDICTIONS_PATH)
    df = df[df["sample_index"].astype(int).isin(test_target_index_set)].copy()
    assert len(df) == len(test_target_indices), "Predictions must cover all test_target_indices."
    assert set(df["sample_index"].astype(int).tolist()) == test_target_index_set

    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    ir014_metrics = class_metric(y_true, y_pred, IR014_LABEL)
    ir007_metrics = class_metric(y_true, y_pred, IR007_LABEL)

    ir014_df = df[df["y_true"] == IR014_LABEL]
    ir014_correct_df = ir014_df[ir014_df["y_pred"] == IR014_LABEL]
    ir014_wrong_df = ir014_df[ir014_df["y_pred"] != IR014_LABEL]
    ir014_to_ir007_df = ir014_df[ir014_df["y_pred"] == IR007_LABEL]
    ir007_df = df[df["y_true"] == IR007_LABEL]
    predicted_ir007_df = df[df["y_pred"] == IR007_LABEL]
    other_predicted_ir007_df = predicted_ir007_df[predicted_ir007_df["y_true"] != IR007_LABEL]

    ir014_wrong_counts = Counter(ir014_wrong_df["y_pred"].astype(int).tolist())
    if ir014_wrong_counts:
        most_confused_label, most_confused_count = ir014_wrong_counts.most_common(1)[0]
        most_confused_class = CLASS_NAMES[most_confused_label]
    else:
        most_confused_label = None
        most_confused_count = 0
        most_confused_class = "None"

    save_metadata_csv(IR014_CORRECT_CSV, ir014_correct_df)
    save_metadata_csv(IR014_TO_IR007_CSV, ir014_to_ir007_df)
    save_metadata_csv(PREDICTED_IR007_CSV, predicted_ir007_df)
    plot_waveform(data, ir014_correct_df, ir014_to_ir007_df)
    plot_spectrum(data, ir014_df, ir007_df)

    lines = []
    append(lines, "IR014 hard-class analysis")
    append(lines)
    append(lines, "IR014:")
    append(lines, f"  IR014 total samples: {len(ir014_df)}")
    append(lines, f"  IR014 correct: {len(ir014_correct_df)}")
    append(lines, f"  IR014 wrong: {len(ir014_wrong_df)}")
    append(lines, f"  IR014 recall: {ir014_metrics['recall']:.6f}")
    append(lines, f"  IR014 F1: {ir014_metrics['f1']:.6f}")
    append(lines, f"  IR014 most confused class: {most_confused_class}")
    append(lines, f"  IR014 -> IR007 count: {len(ir014_to_ir007_df)}")
    append(lines, f"  IR014 wrong prediction distribution: {dict(ir014_wrong_counts)}")
    append(lines)
    append(lines, "IR007:")
    append(lines, f"  IR007 total samples: {len(ir007_df)}")
    append(lines, f"  IR007 correct count: {int((ir007_df['y_pred'] == IR007_LABEL).sum())}")
    append(lines, f"  IR007 recall: {ir007_metrics['recall']:.6f}")
    append(lines, f"  IR007 precision: {ir007_metrics['precision']:.6f}")
    append(lines, f"  other classes predicted as IR007: {len(other_predicted_ir007_df)}")
    append(lines, f"  predicted IR007 true-class distribution: {dict(Counter(predicted_ir007_df['y_true'].astype(int).tolist()))}")
    append(lines)
    append(lines, "Artifacts:")
    append(lines, f"  IR014 correct csv: {IR014_CORRECT_CSV}")
    append(lines, f"  IR014 -> IR007 csv: {IR014_TO_IR007_CSV}")
    append(lines, f"  predicted IR007 csv: {PREDICTED_IR007_CSV}")
    append(lines, f"  waveform figure: {WAVEFORM_FIGURE_PATH}")
    append(lines, f"  spectrum figure: {SPECTRUM_FIGURE_PATH}")
    append(lines, f"  Report path: {REPORT_PATH}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Deep CORAL IR014 analysis completed.")


if __name__ == "__main__":
    main()

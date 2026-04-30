"""Inspect variable names, shapes, and dtypes in raw CWRU .mat files to prepare later loading rules."""

import csv
from pathlib import Path

from scipy.io import loadmat


RAW_DIR = Path("data/raw/cwru_12k_de")
OUTPUT_PATH = Path("outputs/logs/cwru_mat_summary.csv")
MATLAB_SYSTEM_FIELDS = {"__header__", "__version__", "__globals__"}


def infer_variable_type(variable_name: str) -> str:
    upper_name = variable_name.upper()

    if "DE" in upper_name:
        return "DE"
    if "FE" in upper_name:
        return "FE"
    if "BA" in upper_name:
        return "BA"
    if "RPM" in upper_name:
        return "RPM"
    return "other"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(RAW_DIR.glob("*.mat"))
    rows = []

    for mat_path in mat_files:
        mat_data = loadmat(mat_path)
        variables = [
            (name, value)
            for name, value in mat_data.items()
            if name not in MATLAB_SYSTEM_FIELDS
        ]

        inferred_types = {infer_variable_type(name) for name, _ in variables}
        has_de = "DE" in inferred_types
        has_fe = "FE" in inferred_types
        has_ba = "BA" in inferred_types
        has_rpm = "RPM" in inferred_types

        print(f"File: {mat_path.name}")
        print(f"  DE signal: {has_de}")
        print(f"  FE signal: {has_fe}")
        print(f"  BA signal: {has_ba}")
        print(f"  RPM: {has_rpm}")

        for variable_name, value in variables:
            inferred_type = infer_variable_type(variable_name)
            shape = tuple(value.shape)
            dtype = str(value.dtype)

            print(f"  Variable: {variable_name}")
            print(f"    shape: {shape}")
            print(f"    dtype: {dtype}")
            print(f"    inferred type: {inferred_type}")

            rows.append(
                {
                    "file_name": mat_path.name,
                    "variable_name": variable_name,
                    "shape": str(shape),
                    "dtype": dtype,
                    "inferred_type": inferred_type,
                    "has_de_signal": has_de,
                    "has_fe_signal": has_fe,
                    "has_ba_signal": has_ba,
                    "has_rpm": has_rpm,
                }
            )

    fieldnames = [
        "file_name",
        "variable_name",
        "shape",
        "dtype",
        "inferred_type",
        "has_de_signal",
        "has_fe_signal",
        "has_ba_signal",
        "has_rpm",
    ]

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


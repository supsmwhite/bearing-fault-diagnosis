# Bearing Fault Diagnosis under Cross-Load Domain Shift

## Project Goal

This project studies deep learning generalization across motor load conditions using the CWRU 12k Drive End bearing fault dataset.

The final evaluation does not use ordinary random splitting. Instead, it uses a cross-load transfer setting:

- Source domain: 3 hp
- Target domain: 2 hp
- Baseline: CNN1D source-only
- Improved model: DANN
- Goal: verify whether domain adaptation improves cross-load generalization

## Project Architecture

`src/`

- Reusable Python modules, including `data`, `models`, `train`, `eval`, and `utils`

`scripts/checks/`

- Environment checks, data checks, and split sanity checks

`scripts/data/`

- Data download, raw `.mat` reading, window building, and split generation

`scripts/experiments/`

- Training, evaluation, and analysis entrypoints for each experiment

`configs/experiments/`

- Experiment-level configuration files

`docs/experiments/`

- Detailed experiment descriptions and result analysis

`outputs/experiments/`

- Generated experiment artifacts, including checkpoints, logs, figures, and predictions

## Dataset

- Dataset: CWRU Bearing Data Center
- Signal: 12k Drive End `DE_time`
- Classes: 10
- Window size: 1024
- Stride: 512
- Normalization: per-window z-score
- Source load: 3 hp
- Target load: 2 hp

| Label | Class |
| ----: | ------- |
| 0 | Normal |
| 1 | IR007 |
| 2 | B007 |
| 3 | OR007@6 |
| 4 | IR014 |
| 5 | B014 |
| 6 | OR014@6 |
| 7 | IR021 |
| 8 | B021 |
| 9 | OR021@6 |

## Experiment Design

| Split | Load | Usage |
| ---------------------- | ---: | -------------------------------------------- |
| train_source | 3 hp | supervised classification training |
| val_source | 3 hp | source validation |
| train_target_unlabeled | 2 hp | DANN domain adaptation only, labels not used |
| test_target | 2 hp | final target evaluation |

## Main Results

| Model | Source Load | Target Load | Target Accuracy | Target Macro-F1 |
| ----------------- | ----------: | ----------: | --------------: | --------------: |
| CNN1D source-only | 3 hp | 2 hp | 0.816613 | 0.686022 |
| DANN best | 3 hp | 2 hp | 0.889968 | 0.831587 |
| DANN final | 3 hp | 2 hp | 0.864078 | 0.820324 |

DANN best improves target macro-F1 by 0.145565 compared with the CNN1D source-only baseline.

## Error Analysis

| Class | CNN1D F1 | DANN best F1 | Main Confusion after DANN |
| ----- | -------: | -----------: | ------------------------- |
| IR014 | 0.000000 | 0.121951 | IR007 |
| B021 | 0.000000 | 0.789916 | B007 |

- DANN significantly improves B021.
- DANN remains insufficient for IR014.
- Domain adaptation improves overall generalization but does not fully solve all class-level transfer shifts.

## How to Run

Python interpreter:

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe
```

Environment check:

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\checks\check_env.py
```

Build data windows:

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\data\build_windows.py
```

Create split:

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\data\create_splits.py
```

Train CNN1D:

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\experiments\cnn1d_source_3to2\train.py
```

Evaluate CNN1D:

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\experiments\cnn1d_source_3to2\eval_cross_load.py
```

Train DANN:

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\experiments\dann_3to2\train.py
```

Evaluate DANN:

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\experiments\dann_3to2\eval_cross_load.py
```

## Current Status

Current status: CNN1D baseline and DANN have been completed for the 3 hp → 2 hp transfer task.

## Future Work

- Add Deep CORAL as a simpler domain adaptation baseline
- Test a harder 0 hp → 3 hp transfer setting
- Investigate hard classes such as IR014
- Add feature visualization such as t-SNE or UMAP


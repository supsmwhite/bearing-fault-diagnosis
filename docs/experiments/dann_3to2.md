# DANN: 3 hp → 2 hp

## Purpose

DANN uses domain adversarial training to learn more domain-invariant features between source and target load conditions, improving cross-load generalization.

## Setup

| Item | Value |
| ---------------------- | ---------------------- |
| Dataset | CWRU 12k Drive End |
| Source load | 3 hp |
| Target load | 2 hp |
| Model | DANN |
| Source train split | train_source |
| Source val split | val_source |
| Target unlabeled split | train_target_unlabeled |
| Target test split | test_target |

## Training Logic

- Class loss only uses source labels.
- Domain loss uses source and target domain labels.
- Target class labels are not used during training.
- GRL is used for adversarial domain adaptation.

## Results

| Model | Target Accuracy | Target Macro-F1 |
| ----------------- | --------------: | --------------: |
| CNN1D source-only | 0.816613 | 0.686022 |
| DANN best | 0.889968 | 0.831587 |
| DANN final | 0.864078 | 0.820324 |

## Improvement

DANN best improves target macro-F1 by 0.145565 over CNN1D source-only.

## Error Analysis

| Class | CNN1D F1 | DANN best F1 | Most confused with after DANN |
| ----- | -------: | -----------: | ----------------------------- |
| IR014 | 0.000000 | 0.121951 | IR007 |
| B021 | 0.000000 | 0.789916 | B007 |

## Conclusion

DANN improves overall cross-load generalization, especially for B021, but IR014 remains difficult. This shows that domain adaptation improves the baseline but does not fully solve all class-level shifts.

## Commands

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\experiments\dann_3to2\train.py
```

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\experiments\dann_3to2\plot_training.py
```

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\experiments\dann_3to2\eval_cross_load.py
```

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\experiments\dann_3to2\analyze_results.py
```


# Gap Split Recheck

> Note: This check is exploratory. Its results are not used as the main project conclusion because the DANN final checkpoint showed unusually high target performance and requires further investigation.

## Purpose

This recheck evaluates whether the CNN1D vs DANN conclusion remains stable after adding a gap between train/validation/test regions to reduce overlap-related optimism.

## Split

| Split | Samples |
|---|---:|
| train_source | 2306 |
| val_source | 622 |
| train_target_unlabeled | 1995 |
| test_target | 927 |
| discarded_gap | 305 |

## Planned Experiments

| Experiment | Status |
|---|---|
| CNN1D source-only with gap split | Todo |
| DANN with gap split | Todo |

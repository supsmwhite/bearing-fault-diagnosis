# CNN1D Source-only Baseline: 3 hp → 2 hp

## Purpose

This experiment validates the baseline performance of a standard CNN1D model in the cross-load bearing fault diagnosis setting.

## Setup

| Item | Value |
| ----------- | ------------------ |
| Dataset | CWRU 12k Drive End |
| Source load | 3 hp |
| Target load | 2 hp |
| Model | CNN1D |
| Train split | train_source |
| Val split | val_source |
| Test split | test_target |

## Results

| Metric | Value |
| -------------------------- | -------: |
| Source validation macro-F1 | 1.000000 |
| Target test accuracy | 0.816613 |
| Target test macro-F1 | 0.686022 |

## Error Analysis

| Class | F1 | Most confused with |
| ----- | -------: | ------------------ |
| IR014 | 0.000000 | IR007 |
| B021 | 0.000000 | OR014@6 |

## Conclusion

CNN1D performs very well on the source-domain validation set, but its macro-F1 drops clearly after transfer to the 2 hp target test set. This shows that source-only CNN1D has insufficient cross-load generalization ability.


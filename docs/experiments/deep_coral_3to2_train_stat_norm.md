# Deep CORAL train-stat-only normalization ablation: 3 hp -> 2 hp

## Purpose

This stricter ablation tests whether the strong no-window-zscore result remains when normalization statistics are computed only from train_source + train_target_unlabeled windows, excluding test_target statistics.

## Setting

- Dataset: CWRU 12k Drive End
- Source load: 3 hp
- Target load: 2 hp
- Model: DeepCORAL1D
- Normalization: global z-score using train_source + train_target_unlabeled windows only
- No per-window z-score
- Seed: 42
- Epochs: 20
- Batch size: 64
- CORAL loss weight: 1.0

## Results

| Metric | Best | Final |
|---|---:|---:|
| target accuracy | 0.911543 | 0.919094 |
| target macro-F1 | 0.850717 | 0.859991 |
| IR014 recall | 0.028169 | 0.000000 |
| IR014 F1 | 0.054795 | 0.000000 |
| IR014 -> IR007 count | 0 | 0 |
| IR007 precision | 1.000000 | 1.000000 |
| IR007 recall | 1.000000 | 1.000000 |

## Interpretation

- Train-stat-only normalization does not reproduce the strong no-window-zscore improvement.
- Final target macro-F1 = 0.859991 is lower than the original per-window Deep CORAL best macro-F1 = 0.889172.
- IR014 remains unresolved and drops to zero recall in the final checkpoint.
- IR014 -> IR007 count = 0 should not be interpreted as success because IR014 recall is also 0; the model no longer confuses IR014 with IR007 but still fails to identify IR014.
- Therefore, the previous no-window-zscore result should remain an exploratory preprocessing observation, not a replacement for the original main result.

## Conclusion

The strict train-stat-only control weakens the claim that no-window normalization alone robustly solves IR014. The main project conclusion should remain based on the original per-window z-score setting, where Deep CORAL best is the strongest validated model. The normalization branch is useful as a diagnostic finding showing that preprocessing affects IR014, but it should not replace the main result.

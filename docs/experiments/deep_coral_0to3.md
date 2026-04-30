# Deep CORAL 0 hp -> 3 hp robustness extension

## Purpose

This experiment evaluates whether the 3 hp -> 2 hp finding generalizes to a harder transfer direction, 0 hp -> 3 hp.

## Setting

- Dataset: CWRU 12k Drive End
- Source load: 0 hp
- Target load: 3 hp
- Classes: 10
- Window size: 1024
- Stride: 512
- Normalization: per-window z-score
- Source train split: train_source_indices
- Source val split: val_source_indices
- Target unlabeled split: train_target_unlabeled_indices
- Target test split: test_target_indices
- Models compared:
  - CNN1D source-only
  - Deep CORAL

## Split summary

| Split | Samples | Load |
|---|---:|---:|
| train_source | 2079 | 0 hp |
| val_source | 526 | 0 hp |
| train_target_unlabeled | 2150 | 3 hp |
| test_target | 931 | 3 hp |

## Results

| Model / Checkpoint | Target Accuracy | Target Macro-F1 |
|---|---:|---:|
| CNN1D source-only | 0.836735 | 0.729576 |
| Deep CORAL best | 0.809882 | 0.714721 |
| Deep CORAL final | 0.842105 | 0.742283 |

## Comparison with main 3 hp -> 2 hp result

| Transfer | CNN1D Macro-F1 | Deep CORAL Macro-F1 | Gain |
|---|---:|---:|---:|
| 3 hp -> 2 hp | 0.686022 | 0.889172 | +0.203150 |
| 0 hp -> 3 hp | 0.729576 | 0.742283 | +0.012707 |

## Error analysis

Worst classes under Deep CORAL 0->3:

- IR014: F1 = 0.00, 0/71 recognized, mainly predicted as B014.
- B021: F1 = 0.00, 0/72 recognized, mainly predicted as B014.
- B014: F1 = 0.51, high recall but low precision, absorbing IR014 and B021.

## Interpretation

Deep CORAL provides only a small improvement over CNN1D source-only in the harder 0 hp -> 3 hp setting. Unlike the main 3 hp -> 2 hp experiment, global covariance alignment is not sufficient to resolve class-level confusion under the larger load gap. The main failure mode is that B014 becomes an attractor class and absorbs IR014 and B021.

## Conclusion

The 0 hp -> 3 hp robustness extension does not overturn the main 3 hp -> 2 hp finding. Instead, it shows the limitation of Deep CORAL under a harder transfer direction: it remains slightly better at the final checkpoint but the improvement is marginal. The main project conclusion should remain based on the original 3 hp -> 2 hp setting.

# Deep CORAL 3 hp to 2 hp Experiment

## 实验目的

Deep CORAL 是比 DANN 更简单的 domain adaptation baseline，通过 source / target feature covariance alignment 改善跨负载泛化。

## 实验设置

| Item | Value |
| --- | --- |
| Dataset | CWRU 12k Drive End |
| Source load | 3 hp |
| Target load | 2 hp |
| Source train | train_source |
| Source val | val_source |
| Target unlabeled train | train_target_unlabeled |
| Target test | test_target |
| Model | DeepCORAL1D |
| coral_loss_weight | 1.0 |
| epochs | 20 |
| batch_size | 64 |
| seed | 42 |

## 训练观察

| Item | Value |
| --- | ---: |
| Best epoch | 4 |
| Best source-val macro-F1 | 1.000000 |

Epoch 11 source-val 曾出现一次性下降，但后续恢复；最终解释以 target test 和 audit 为准。

## Target Test Results

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| CNN1D source-only | 0.816613 | 0.686022 |
| DANN best | 0.889968 | 0.831587 |
| DANN final | 0.864078 | 0.820324 |
| Deep CORAL best | 0.928803 | 0.889172 |
| Deep CORAL final | 0.928803 | 0.877385 |

## Improvement

| Comparison | Macro-F1 Gain |
| --- | ---: |
| Deep CORAL best vs CNN1D | 0.203150 |
| Deep CORAL best vs DANN best | 0.057585 |

## Audit Result

| Check | Result |
| --- | --- |
| split overlap | all zero |
| test_target samples | 927 |
| test_target load distribution | `{2: 927}` |
| best prediction csv | passed |
| final prediction csv | passed |
| audit | passed |

## Error Analysis

| Checkpoint | Class | Precision | Recall | F1 | Most Confused With | Count |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Deep CORAL best | IR014 | 0.733333 | 0.154930 | 0.255814 | IR007 | 55 |
| Deep CORAL best | B021 | 1.000000 | 1.000000 | 1.000000 | None | 0 |
| Deep CORAL final | IR014 | - | - | 0.131579 | IR007 | - |
| Deep CORAL final | B021 | - | - | 1.000000 | None | 0 |

## Conclusion

Deep CORAL is the current strongest model on the original 3 hp → 2 hp split.

However, IR014 remains the main unresolved hard class, so the result should be reported as stronger overall domain adaptation, not complete class-level robustness.

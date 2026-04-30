# DANN: 3 hp → 2 hp

## 实验目的

DANN 通过 domain adversarial training 学习源域和目标域之间更域不敏感的特征，用于改善跨负载泛化能力。

## 实验设置

| 项目 | 值 |
| ---------------------- | ---------------------- |
| 数据集 | CWRU 12k Drive End |
| 源负载 | 3 hp |
| 目标负载 | 2 hp |
| 模型 | DANN |
| 源域训练 split | train_source |
| 源域验证 split | val_source |
| 目标域无标签 split | train_target_unlabeled |
| 目标域测试 split | test_target |

## 训练逻辑

- class loss 只使用 source label。
- domain loss 使用 source 和 target 的 domain label。
- target class label 不参与训练。
- GRL 用于实现对抗式域适应。

## 结果

| Model | Target Accuracy | Target Macro-F1 |
| ----------------- | --------------: | --------------: |
| CNN1D source-only | 0.816613 | 0.686022 |
| DANN best | 0.889968 | 0.831587 |
| DANN final | 0.864078 | 0.820324 |

## 提升幅度

DANN best 相比 CNN1D source-only 将 target macro-F1 提升 0.145565。

## 错误分析

| Class | CNN1D F1 | DANN best F1 | DANN 后主要错分为 |
| ----- | -------: | -----------: | ----------------------------- |
| IR014 | 0.000000 | 0.121951 | IR007 |
| B021 | 0.000000 | 0.789916 | B007 |

## 结论

DANN 改善了整体跨负载泛化能力，尤其明显改善了 B021。但 IR014 仍然困难，说明 domain adaptation 相比 source-only baseline 有明显提升，但还不能完全解决所有类别级别的迁移偏移。

## Additional Reliability Check

A gap split recheck was explored to reduce overlap-related optimism. However, because the DANN final checkpoint showed near-perfect target performance, this result is not used as a main conclusion. The main reported result remains the original 3 hp → 2 hp comparison between CNN1D source-only and DANN.

## 运行命令

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

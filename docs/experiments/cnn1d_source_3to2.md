# CNN1D Source-only Baseline: 3 hp → 2 hp

## 实验目的

该实验用于验证普通 CNN1D 在跨负载轴承故障诊断场景下的 baseline 表现。

## 实验设置

| 项目 | 值 |
| ----------- | ------------------ |
| 数据集 | CWRU 12k Drive End |
| 源负载 | 3 hp |
| 目标负载 | 2 hp |
| 模型 | CNN1D |
| 训练 split | train_source |
| 验证 split | val_source |
| 测试 split | test_target |

## 结果

| 指标 | 数值 |
| -------------------------- | -------: |
| Source validation macro-F1 | 1.000000 |
| Target test accuracy | 0.816613 |
| Target test macro-F1 | 0.686022 |

## 错误分析

| Class | F1 | 主要错分为 |
| ----- | -------: | ------------------ |
| IR014 | 0.000000 | IR007 |
| B021 | 0.000000 | OR014@6 |

## 结论

CNN1D 在源域验证集上表现很好，但迁移到 2 hp 目标测试集后 macro-F1 明显下降。这说明 source-only CNN1D 的跨负载泛化能力不足。


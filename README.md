# 跨负载工况下的轴承故障诊断

## 项目目标

本项目基于 CWRU 12k Drive End 轴承故障数据，研究深度学习模型在不同电机负载工况之间的泛化能力。

最终结果不采用普通随机切分，而是使用更接近真实迁移场景的跨负载设置：

- 源域：3 hp
- 目标域：2 hp
- baseline：CNN1D source-only
- 改进模型：DANN、Deep CORAL
- 目标：验证 domain adaptation 是否能提升跨负载泛化能力

## 项目结构

`src/`

- 可复用 Python 模块，包括 `data`、`models`、`train`、`eval`、`utils`

`scripts/checks/`

- 环境检查、数据检查、split 合理性检查

`scripts/data/`

- 数据下载、原始 `.mat` 读取、滑窗构建、split 生成

`scripts/experiments/`

- 每个实验的训练、评估和结果分析入口

`configs/experiments/`

- 实验级配置文件

`docs/experiments/`

- 每个实验的详细说明和结果分析

`outputs/experiments/`

- 实验生成产物，包括 checkpoint、log、figure、prediction 等

## 数据集

- 数据集：CWRU Bearing Data Center
- 信号：12k Drive End `DE_time`
- 类别数：10
- 窗口长度：1024
- 步长：512
- 归一化：每个窗口单独做 z-score
- 源负载：3 hp
- 目标负载：2 hp

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

## 实验设计

| Split | Load | 用途 |
| ---------------------- | ---: | -------------------------------------------- |
| train_source | 3 hp | 有监督分类训练 |
| val_source | 3 hp | 源域验证 |
| train_target_unlabeled | 2 hp | 仅用于 DANN / Deep CORAL 域适应，不使用类别标签 |
| test_target | 2 hp | 最终目标域评估 |

## 主要结果

| Model | Source Load | Target Load | Target Accuracy | Target Macro-F1 |
| ----------------- | ----------: | ----------: | --------------: | --------------: |
| CNN1D source-only | 3 hp | 2 hp | 0.816613 | 0.686022 |
| DANN best | 3 hp | 2 hp | 0.889968 | 0.831587 |
| DANN final | 3 hp | 2 hp | 0.864078 | 0.820324 |
| Deep CORAL best | 3 hp | 2 hp | 0.928803 | 0.889172 |
| Deep CORAL final | 3 hp | 2 hp | 0.928803 | 0.877385 |

Deep CORAL best 是 original 3 hp → 2 hp split 上当前最强模型：相对 CNN1D source-only，target macro-F1 从 0.686022 提升到 0.889172；相对 DANN best，从 0.831587 提升到 0.889172。

## 错误分析

| Class | CNN1D F1 | DANN best F1 | Deep CORAL best F1 | Deep CORAL 后主要错分为 |
| ----- | -------: | -----------: | ------------------: | ------------------------- |
| IR014 | 0.000000 | 0.121951 | 0.255814 | IR007 |
| B021 | 0.000000 | 0.789916 | 1.000000 | None |

- DANN 显著改善了 B021。
- Deep CORAL 在该 split 中完全修复了 B021。
- IR014 仍然困难，Deep CORAL 下仍主要错分为 IR007。
- domain adaptation 改善了整体跨负载泛化能力，但没有完全解决所有类别级别的迁移问题。

## Normalization ablation note

- no-window-zscore 探索性 ablation 显示，预处理方式会明显影响 IR014。
- 但更严格的 train-stat-only normalization control 没有复现该提升。
- Train-stat-norm Deep CORAL final 的 target macro-F1 = 0.859991，IR014 recall = 0.000000。
- 因此 normalization 结果只作为诊断性观察，不作为主结论。
- 当前主比较仍基于 original per-window z-score 设置。

## Harder transfer extension: 0 hp → 3 hp

- A harder 0 hp → 3 hp transfer was tested as a robustness extension.
- CNN1D source-only achieved target macro-F1 = 0.729576.
- Deep CORAL final achieved target macro-F1 = 0.742283.
- The gain is much smaller than in 3 hp → 2 hp.
- Main failure classes were IR014 and B021, both mainly absorbed by B014.
- This extension suggests that larger load gaps may require stronger class-conditional alignment; it does not replace the main 3 hp → 2 hp conclusion.

## 如何运行

固定 Python 解释器：

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe
```

环境检查：

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\checks\check_env.py
```

构建滑窗数据：

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\data\build_windows.py
```

创建 split：

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\data\create_splits.py
```

训练 CNN1D：

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\experiments\cnn1d_source_3to2\train.py
```

评估 CNN1D：

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\experiments\cnn1d_source_3to2\eval_cross_load.py
```

训练 DANN：

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\experiments\dann_3to2\train.py
```

评估 DANN：

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\experiments\dann_3to2\eval_cross_load.py
```

训练 Deep CORAL：

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\experiments\deep_coral_3to2\train.py
```

评估 Deep CORAL：

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\experiments\deep_coral_3to2\eval_cross_load.py
```

## 当前状态

当前状态：CNN1D baseline、DANN 和 Deep CORAL 已完成 CWRU 3 hp → 2 hp original split 迁移任务。Deep CORAL best 当前整体指标最高，但 IR014 仍未解决。

## 说明

A gap split reliability check was explored, but its results are not used as the main conclusion because the final-checkpoint behavior requires further investigation.

## 后续计划

- 可选探索 DANN 0 hp → 3 hp
- 进一步分析 IR014 等困难类别
- 添加 t-SNE 或 UMAP 等特征可视化

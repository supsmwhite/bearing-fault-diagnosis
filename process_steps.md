# Process Steps

| Step | 任务 | 状态 | 说明 |
| --- | --- | --- | --- |
| 0 | 明确项目方向 | ✅ Done | CWRU 跨负载轴承故障诊断 |
| 1 | 创建项目骨架 | ✅ Done | 创建 README、process_steps、目录结构 |
| 2 | 检查 PyTorch 环境 | ✅ Done | 使用指定 Python 解释器，PyTorch 与 CUDA 检查通过 |
| 3 | 下载 CWRU 数据 | ✅ Done | 20 个最小 CWRU 原始 `.mat` 文件已下载并检查通过 |
| 4 | 读取 .mat 数据 | ✅ Done | 已确认 20 个文件均包含 DE signal，并保存变量检查 summary |
| 5 | 构建滑窗数据集 | ✅ Done | 已完成 DE signal 读取与滑窗切片，保存 processed npz |
| 6 | 创建 original split | ✅ Done | 已创建 3 hp → 2 hp original split，并验证无交集 |
| 7 | CNN1D baseline training | ✅ Done | Original split CNN1D source-only 训练完成 |
| 8 | CNN1D cross-load evaluation | ✅ Done | Original split target macro-F1 = 0.686022 |
| 9 | DANN forward check | ✅ Done | DANN 模型结构 forward 检查通过 |
| 10 | DANN training | ✅ Done | Original split DANN 训练完成 |
| 11 | DANN cross-load evaluation | ✅ Done | Original split DANN best target macro-F1 = 0.831587 |
| 12 | Result documentation for original split | ✅ Done | README 和主要实验文档只使用 original split 作为主结论 |
| 13 | Gap split reliability check | ✅ Exploratory | 已探索 gap split，但不作为主结论；DANN final 近乎满分，仍需进一步调查 |
| 14 | Deep CORAL baseline | ⬜ Todo | 后续新增更简单的 domain adaptation baseline |
| 15 | 0 hp → 3 hp harder transfer | ⬜ Todo | 后续验证更困难跨负载设置 |
| 16 | IR014 hard-class investigation | ⬜ Todo | 分析 IR014 仍难迁移的原因 |
| 17 | Feature visualization | ⬜ Todo | 添加 t-SNE 或 UMAP 特征可视化 |

Current main conclusion:
CNN1D source-only target macro-F1 = 0.686022.
DANN best target macro-F1 = 0.831587.


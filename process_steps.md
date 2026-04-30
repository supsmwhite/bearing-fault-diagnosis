# Process Steps

| Step | 任务 | 状态 | 说明 |
| --- | --- | --- | --- |
| 0 | 明确项目方向 | ✅ Done | CWRU 跨负载轴承故障诊断 |
| 1 | 创建项目骨架 | ✅ Done | 创建 README、process_steps、目录结构 |
| 2 | 检查 PyTorch 环境 | ✅ Done | 使用指定 Python 解释器，PyTorch 与 CUDA 检查通过 |
| 3 | 下载 CWRU 数据 | ✅ Done | 20 个最小 CWRU 原始 `.mat` 文件已下载并检查通过 |
| 4 | 读取 .mat 数据 | ✅ Done | 已确认 20 个文件均包含 DE signal，并保存变量检查 summary |
| 5 | 构建滑窗数据集 | ✅ Done | 已完成 DE signal 读取与滑窗切片，保存 processed npz |
| 6 | 创建跨负载 split | ✅ Done | 已创建 PyTorch Dataset 和 3 hp → 2 hp split，并验证无交集 |
| 7 | CNN1D baseline training | ✅ Done | 已完成 CNN1D source-only 训练和 source validation |
| 8 | CNN1D cross-load evaluation | ✅ Done | Target accuracy = 0.816613，target macro-F1 = 0.686022 |
| 9 | 轻量架构整理 | ✅ Done | 已整理 scripts、configs、docs 的实验结构 |
| 10 | DANN forward check | ✅ Done | 已实现 DANN 模型结构并通过 forward 检查 |
| 11 | DANN training | ✅ Done | 已完成 DANN 训练、日志保存和训练曲线生成 |
| 12 | DANN cross-load evaluation | ✅ Done | DANN best target macro-F1 = 0.831587，DANN final target macro-F1 = 0.820324 |
| 13 | Result documentation | ✅ Done | 已整理 README、实验文档、配置结果和关键错误分析 |
| 14 | Add Deep CORAL | ⬜ Todo | 后续新增更简单的 domain adaptation baseline |
| 15 | Test harder 0 hp → 3 hp transfer | ⬜ Todo | 后续验证更困难跨负载设置 |
| 16 | Investigate IR014 hard class | ⬜ Todo | 分析 IR014 仍难迁移的原因 |
| 17 | Add feature visualization | ⬜ Todo | 添加 t-SNE 或 UMAP 特征可视化 |

Current milestone:
CNN1D source-only and DANN have been completed on the CWRU 3 hp → 2 hp transfer task. DANN best improves target macro-F1 from 0.686022 to 0.831587.


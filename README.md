# CWRU Bearing Fault Diagnosis

## 项目目标

基于 CWRU Bearing Data Center 数据集，构建轴承故障诊断深度学习项目，用于后续验证跨负载工况下的故障分类能力。

## 数据集

- 数据来源：CWRU Bearing Data Center
- 当前阶段不下载数据
- 原始数据预留目录：`data/raw/`
- 处理后数据预留目录：`data/processed/`

## 技术路线

- Baseline：1D-CNN
- 改进方向：DANN / Deep CORAL
- 当前阶段：Step 1，只做项目骨架和环境检查

## Python 环境

固定 Python 解释器：

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe
```

## 环境检查

```powershell
E:\anaconda\anaconda_app\envs\pytorch\python.exe scripts\01_check_env.py
```

脚本输出：

- Python 路径
- Python 版本
- PyTorch 版本
- CUDA 是否可用
- GPU 名称，如果有
- 当前工作目录


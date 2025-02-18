#!/bin/bash

# 创建新的 conda 环境
conda create -n swt-bench python=3.9 -y

# 激活环境
conda activate swt-bench

# 安装基本依赖
pip install requests datasets docker unidiff python-dotenv tqdm fire editdistance GitPython

# 安装开发依赖
pip install pre-commit

# 安装测试依赖
pip install pytest pytest-cov pytest-xdist

# 安装图表相关依赖
pip install tiktoken numpy tabulate venny4py nltk

# 安装项目本身（以可编辑模式）
pip install -e .

echo "环境配置完成！使用 'conda activate swt-bench' 来激活环境" 
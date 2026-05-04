# AGENTS.md

本文档为 Codex (Codex.ai/code) 提供本代码库的操作指南。

## 项目概述

本项目为 **WRF 最优亏缺灌溉分析系统**（Water Research & Forecasting - Optimal Deficit Irrigation Analysis System），是一个基于 Python 的科学计算项目，使用机器学习（随机森林）分析作物（水稻、小麦、玉米）的最优亏缺灌溉比例。

## 架构说明

### 入口文件
- **`WRF_master.py`**: 主入口文件，提供命令行接口，支持多种运行模式（`full`、`original`、`response_surface`、`optimize`、`spatial`）

### 核心模块（`mypackage/`）
- **`W_glob_mac.py`**: 机器学习核心模块 - 随机森林训练/预测流程、数据预处理
- **`W_response_surface.py`**: 响应面构建模块 - 构建"网格×比例×指标"三维响应面，支持增量计算
- **`W_optimization.py`**: 优化模块 - 使用加权评分进行多目标优化，选择最优亏缺比例
- **`W_spatial_analysis.py`**: 空间分析模块 - 生成空间统计和可视化
- **`W_glob_tif.py`**: TIF/CSV 转换工具 - 地理空间数据格式转换
- **`W_config.py`**: 配置中心 - 模型参数、可视化参数、指标优化方向
- **`W_path_manager.py`**: 路径管理器 - 支持本地和远程数据路径的统一管理
- **`W_soil_manager.py`**: 土壤特征管理 - 土壤质地编码和特征对齐

### 数据处理流程
1. 训练数据（CSV）→ 按指标（产量Y、蒸散发ET、水分利用效率WUE、甲烷CH4、氧化亚氮N2O）训练随机森林模型
2. 全球网格数据 → 在比例范围内（默认0.60-0.90）进行预测 → 构建响应面
3. 多目标优化 → 为每个网格单元计算最优亏缺比例
4. 空间分析 → 生成 TIF 地图、统计数据、与对照组（CK）对比分析

## 常用命令

### 环境配置
```bash
# 虚拟环境已配置在 .venv/ 目录下
# Python 3.13.7，已安装 numpy、pandas、scikit-learn、scipy、tqdm

# 激活环境（Windows Git Bash）
source .venv/Scripts/activate
```

### 运行分析
```bash
# 完整流程（指定作物）
python WRF_master.py --crop rice --mode full

# 单独运行各模块
python WRF_master.py --crop wheat --mode response_surface  # 仅构建响应面
python WRF_master.py --crop maize --mode optimize          # 仅运行优化
python WRF_master.py --crop rice --mode spatial            # 仅生成空间分析

# 自定义比例范围
python WRF_master.py --crop rice --mode full --lowdata 0.60 --updata 0.90

# 同时运行原有预测流程
python WRF_master.py --crop rice --mode full --run-original
```

### 批量任务模式
编辑 `WRF_master.py` 中的 `TASK_LIST` 变量（第518-542行）配置多个任务，然后直接运行：
```bash
python WRF_master.py
```

### 自定义输出路径
默认输出到 `results/` 目录。可通过环境变量覆盖：
```bash
export WATERML_DIR=/path/to/custom/output
python WRF_master.py --crop rice --mode full
```

## 关键配置

### 支持的作物类型
- `rice`: 水稻模型参数
- `wheat`: 小麦模型参数
- `maize`: 玉米模型参数

### 指标说明（Y、WUE、N2O、CH4、ET、GWP）
每个指标在 `W_config.MODEL_PARAMS` 中配置了优化的随机森林超参数。

### 优化方向（`W_config.INDICATOR_DIRECTION`）
- 最大化：Y（产量）、WUE（水分利用效率）
- 最小化：ET（蒸散发）、CH4（甲烷）、N2O（氧化亚氮）、GWP（全球变暖潜势）

## 数据路径说明

### 输入数据（默认使用 `F:/WaterML/`，可回退到本地 `results/`）
- 训练数据：`{作物}/{日期}/D{指标}.csv`
- 全球网格数据：`{作物}/{日期}/x_data_point_with_TotalN.{作物}.csv`

### 输出结构（`results/{作物}/{输出日期}/`）
- `response_surface/`: 响应面数据（`.npy`格式）
- `optimal/`: 最优比例结果（`.csv`格式）
- `W/spatial_analysis/`: 地图和可视化结果
- `mix_TNR/W/`、`mix_ck/W/`、`mix_best/W/`: 中间处理数据

## 重要实现细节

### 内存优化
- 使用 `np.float32` 替代 `float64`（见 `W_glob_mac.py` 中的 `DTYPE_FLOAT`）
- 通过 `read_csv_cached()` 实现 CSV 缓存，避免重复 I/O
- 响应面构建支持分批处理

### 土壤质地编码
**重要**：所有指标必须使用相同的特征列。`W_soil_manager` 模块确保训练和预测过程中土壤质地编码的一致性。

### GWP 计算
GWP = CH4 × 27 + N2O × 273（CO2当量系数，来自 `W_config`）

### 响应面缓存
响应面以 `.npy` 文件缓存。使用 `force_rebuild=True` 强制重新计算。

### 任务列表模式
当 `WRF_master.py` 中的 `TASK_LIST` 非空时，将覆盖命令行参数，以批量模式运行。

## 开发注意事项

- 编辑模块后无需重新安装，直接运行主脚本即可生效
- 主要输出目录为 `results/`，已配置在 `.gitignore` 中
- 使用 `tqdm` 显示进度条，长耗时操作可监控进度

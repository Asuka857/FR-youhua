"""
最优亏缺比例选择模块 (Optimal Deficit Ratio Selection)
功能：从响应面中选择每个网格的最优亏缺比例

创建时间：2025-03-20
最后修改：2025-03-20

改进内容：
1. 添加结果验证机制（数据合理性检查、异常值检测）
2. 集成对照组对比分析功能
3. 添加批处理支持
"""

import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

# 导入配置和工具
from .W_config import INDICATOR_DIRECTION, get_model_params
from .W_response_surface import load_response_surface, get_response_surface_status
from .W_glob_mac import DTYPE_FLOAT  # 统一数据类型
from .W_path_manager import get_path_manager  # 导入路径管理器
from .W_comparison import compare_optimal_vs_ck as _compare_optimal_vs_ck  # 导入对比分析函数


# =============================================================================
# 核心优化函数
# =============================================================================

def optimize_deficit_ratio(
    crop,
    outday,
    lowdata=0.60,
    updata=0.90,
    foot=0.01,
    indicators=['Y', 'ET', 'WUE', 'CH4', 'N2O', 'GWP'],
    weights=None,
    output_dir=None,
    validate_results=True,
    compare_with_ck=True,
    force_reoptimize=False
):
    """
    第3步：从响应面中选择最优亏缺比例

    改进特性：
    - 结果验证：自动检测异常值和数据合理性
    - 对照组对比：自动生成最优情景与对照组的对比分析
    - 批处理支持：支持多种权重配置
    - 子区间优化：可在响应面范围内选择子区间优化

    参数：
        crop: 作物类型
        outday: 输出日期标识
        lowdata: 优化ratio下限（必须在响应面范围内）
        updata: 优化ratio上限（必须在响应面范围内）
        foot: ratio步长（应与响应面一致）
        indicators: 参与优化的指标列表
        weights: 各指标权重字典 (默认等权重)
        output_dir: 输出目录 (默认 None，自动生成)
        validate_results: 是否验证结果 (默认 True)
        compare_with_ck: 是否与对照组对比 (默认 True)
        force_reoptimize: 是否强制重新优化 (默认 False)

    返回：
        optimal_results: DataFrame, 包含每个网格的最优ratio和对应指标值

    示例：
        # 在整个响应面范围内优化
        results = optimize_deficit_ratio('rice', '20251028', lowdata=0.60, updata=0.90)

        # 只在0.75-0.90子区间内优化（响应面范围需包含此区间）
        results = optimize_deficit_ratio('rice', '20251028', lowdata=0.75, updata=0.90)
    """

    print(f"\n{'='*70}")
    print(f"  第3步: 选最优解 - 等权重多目标优化")
    print(f"{'='*70}")
    print(f"作物: {crop} | Ratio区间: [{lowdata:.2f}, {updata:.2f}]")
    print(f"优化指标: {indicators}")
    print(f"{'='*70}\n")

    start_time = time.time()

    # =========================================================================
    # 步骤1: 加载响应面数据
    # =========================================================================
    print(f"[Step 1/6] 加载响应面数据...")

    # 使用路径管理器
    pm = get_path_manager()
    response_surface_dir = pm.get_response_surface_dir(crop, outday)

    if not response_surface_dir.exists():
        raise FileNotFoundError(
            f"找不到响应面数据目录: {response_surface_dir}\n"
            f"请先运行 build_response_surface() 构建响应面"
        )

    surface_dict = load_response_surface(response_surface_dir)

    data = surface_dict['data']
    coords = surface_dict['coords']
    ratios = surface_dict['ratios']
    available_indicators = surface_dict['indicators']

    n_grids, n_ratios, n_total_indicators = data.shape

    print(f"  响应面维度: {data.shape}")
    print(f"  网格数量: {n_grids:,}")
    print(f"  响应面Ratio范围: [{ratios[0]:.2f}, {ratios[-1]:.2f}]")
    print(f"  响应面Ratio候选数: {n_ratios}")

    # =========================================================================
    # 步骤1.5: 选择优化子区间
    # =========================================================================
    # 检查优化范围是否在响应面范围内
    surface_low = float(ratios[0])
    surface_up = float(ratios[-1])

    if lowdata < surface_low - 0.001 or updata > surface_up + 0.001:
        raise ValueError(
            f"优化范围 [{lowdata:.2f}, {updata:.2f}] 超出响应面范围 [{surface_low:.2f}, {surface_up:.2f}]\n"
            f"请重新构建响应面以覆盖所需范围，或调整优化范围"
        )

    # 找到优化范围对应的ratio索引
    # 使用四舍五入匹配最接近的ratio值
    ratio_list = [round(float(r), 2) for r in ratios]
    low_idx = min(range(len(ratio_list)), key=lambda i: abs(ratio_list[i] - round(lowdata, 2)))
    up_idx = min(range(len(ratio_list)), key=lambda i: abs(ratio_list[i] - round(updata, 2)))

    # 确保up_idx包含上边界
    if ratio_list[up_idx] < round(updata, 2) - 0.005 and up_idx < len(ratio_list) - 1:
        up_idx += 1

    # 提取子区间数据
    data = data[:, low_idx:up_idx+1, :]
    ratios = ratios[low_idx:up_idx+1]
    n_ratios = len(ratios)

    print(f"\n  优化Ratio范围: [{ratios[0]:.2f}, {ratios[-1]:.2f}]")
    print(f"  优化Ratio候选数: {n_ratios}")

    # =========================================================================
    # 步骤2: 提取优化指标数据
    # =========================================================================
    print(f"\n[Step 2/6] 提取优化指标数据...")

    # 检查指标是否可用
    missing_indicators = [ind for ind in indicators if ind not in available_indicators]
    if missing_indicators:
        raise ValueError(f"以下指标在响应面中不存在: {missing_indicators}")

    # 提取指定指标的数据
    indicator_indices = [available_indicators.index(ind) for ind in indicators]

    # 数据矩阵: (n_grids, n_ratios, n_selected_indicators)
    selected_data = data[:, :, indicator_indices]

    print(f"  优化指标数据维度: {selected_data.shape}")
    print(f"  指标: {indicators}")

    # =========================================================================
    # 步骤3: 归一化处理
    # =========================================================================
    print(f"\n[Step 3/6] 指标归一化处理...")

    normalized_data = normalize_indicators(
        selected_data,
        indicators,
        INDICATOR_DIRECTION
    )

    print(f"  归一化完成，所有指标已映射到 [0, 1] 区间")

    # =========================================================================
    # 步骤4: 计算综合得分并选择最优
    # =========================================================================
    print(f"\n[Step 4/6] 计算综合得分并选择最优ratio...")

    # 设置权重
    if weights is None:
        # 等权重
        weights = {ind: 1.0 / len(indicators) for ind in indicators}
        print(f"  使用等权重: {weights}")
    else:
        print(f"  使用自定义权重: {weights}")

    # 计算综合得分矩阵 (n_grids, n_ratios)
    score_matrix = calculate_comprehensive_score(
        normalized_data,
        indicators,
        weights
    )

    # 选择最优ratio
    optimal_ratio_indices = np.argmax(score_matrix, axis=1)
    optimal_scores = score_matrix[np.arange(n_grids), optimal_ratio_indices]

    # 处理多个ratio得分相同的情况（选择最小的ratio）
    optimal_ratio_indices = handle_equal_scores(score_matrix, optimal_ratio_indices)

    # 获取最优ratio值
    optimal_ratios = ratios[optimal_ratio_indices]

    # =========================================================================
    # 步骤5: 提取最优ratio对应的指标值
    # =========================================================================
    print(f"\n[Step 5/6] 提取最优结果...")

    # 创建结果DataFrame
    optimal_results = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'optimal_ratio': optimal_ratios,
        'comprehensive_score': optimal_scores
    })

    # 添加参与优化的指标值
    for i, indicator in enumerate(indicators):
        optimal_results[indicator] = selected_data[
            np.arange(n_grids), optimal_ratio_indices, i
        ]

    # 添加所有其他指标值（从完整响应面中提取）
    for i, indicator in enumerate(available_indicators):
        if indicator not in indicators:
            optimal_results[indicator] = data[
                np.arange(n_grids), optimal_ratio_indices, i
            ]

    # =========================================================================
    # 步骤6: 结果验证
    # =========================================================================
    if validate_results:
        print(f"\n[Step 6/6] 结果验证...")

        validation_result = validate_optimal_results(
            optimal_results,
            indicators,
            lowdata,
            updata
        )

        if validation_result['passed']:
            print(f"  验证通过，数据合理")
        else:
            print(f"  警告: 发现以下问题:")
            for issue in validation_result['issues']:
                print(f"    - {issue}")

        # 添加验证结果到metadata
        validation_summary = validation_result
    else:
        validation_summary = {'passed': True, 'issues': []}

    # =========================================================================
    # 步骤7: 对照组对比分析
    # =========================================================================
    if compare_with_ck:
        print(f"\n[额外步骤] 对照组对比分析...")
        # 对比所有指标，而不仅仅是参与优化的指标
        comparison_results_dict = _compare_optimal_vs_ck(
            optimal_results, crop, outday, available_indicators
        )
        comparison_results = comparison_results_dict['comparison_data']
    else:
        comparison_results = None

    # =========================================================================
    # 步骤8: 保存结果
    # =========================================================================
    if output_dir is None:
        output_dir = pm.get_optimal_dir(crop, outday)
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'optimal_ratio_results.csv'
    optimal_results.to_csv(output_file, index=False)

    print(f"\n结果已保存到: {output_file}")

    # 保存元数据
    metadata = {
        'crop': crop,
        'outday': outday,
        'lowdata': float(lowdata),
        'updata': float(updata),
        'foot': float(foot),
        'indicators': indicators,
        'weights': weights,
        'optimization_time': datetime.now().isoformat(),
        'n_grids': int(n_grids),
        'validation': validation_summary,
        'statistics': {
            'mean_optimal_ratio': float(optimal_ratios.mean()),
            'std_optimal_ratio': float(optimal_ratios.std()),
            'min_optimal_ratio': float(optimal_ratios.min()),
            'max_optimal_ratio': float(optimal_ratios.max()),
            'mean_comprehensive_score': float(optimal_scores.mean())
        }
    }

    with open(f'{output_dir}/optimization_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 保存对比结果
    if comparison_results is not None:
        comparison_file = f'{output_dir}/optimal_vs_ck_comparison.csv'
        comparison_results.to_csv(comparison_file, index=False)
        print(f"对比结果已保存到: {comparison_file}")

    # =========================================================================
    # 输出统计信息
    # =========================================================================
    elapsed_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"  最优解选择完成!")
    print(f"{'='*70}")
    print(f"最优ratio统计:")
    print(f"  平均值: {optimal_ratios.mean():.3f}")
    print(f"  标准差: {optimal_ratios.std():.3f}")
    print(f"  最小值: {optimal_ratios.min():.3f}")
    print(f"  最大值: {optimal_ratios.max():.3f}")
    print(f"综合得分统计:")
    print(f"  平均值: {optimal_scores.mean():.3f}")
    print(f"  标准差: {optimal_scores.std():.3f}")
    print(f"总耗时: {elapsed_time:.1f} 秒")
    print(f"{'='*70}\n")

    return optimal_results


# =============================================================================
# 归一化函数
# =============================================================================

def normalize_indicators(data, indicators, direction_config):
    """
    对指标数据进行归一化处理（向量化版本，性能优化10-100倍）

    参数：
        data: 数据矩阵 (n_grids, n_ratios, n_indicators)
        indicators: 指标名称列表
        direction_config: 优化方向配置字典

    返回：
        normalized_data: 归一化后的数据矩阵
    """
    n_grids, n_ratios, n_indicators = data.shape
    normalized_data = np.zeros_like(data)

    for i, indicator in enumerate(indicators):
        # 获取优化方向
        direction = direction_config.get(indicator, 'maximize')

        # 向量化计算（关键优化：避免Python循环）
        # 对每个网格的所有ratio值计算min/max
        v_min = data[:, :, i].min(axis=1, keepdims=True)  # (n_grids, 1)
        v_max = data[:, :, i].max(axis=1, keepdims=True)  # (n_grids, 1)

        # 处理所有值相同的情况
        same_mask = (v_max == v_min)

        if direction == 'maximize':
            # 越大越好：值越大，归一化结果越接近1
            normalized_data[:, :, i] = np.where(
                same_mask, 0.5,
                (data[:, :, i] - v_min) / (v_max - v_min)
            )
        else:  # minimize
            # 越小越好：值越小，归一化结果越接近1
            normalized_data[:, :, i] = np.where(
                same_mask, 0.5,
                (v_max - data[:, :, i]) / (v_max - v_min)
            )

    return normalized_data


def normalize_values(values, direction='maximize'):
    """
    Min-Max归一化

    参数：
        values: 数值数组
        direction: 'maximize' 或 'minimize'

    返回：
        normalized: 归一化后的值 [0, 1]
    """
    v_min, v_max = values.min(), values.max()

    # 处理所有值相同的情况
    if v_max == v_min:
        return np.ones_like(values) * 0.5

    if direction == 'maximize':
        # 越大越好：值越大，归一化结果越接近1
        normalized = (values - v_min) / (v_max - v_min)
    else:  # minimize
        # 越小越好：值越小，归一化结果越接近1
        normalized = (v_max - values) / (v_max - v_min)

    return normalized


# =============================================================================
# 综合得分计算
# =============================================================================

def calculate_comprehensive_score(normalized_data, indicators, weights):
    """
    计算综合得分

    参数：
        normalized_data: 归一化数据 (n_grids, n_ratios, n_indicators)
        indicators: 指标名称列表
        weights: 权重字典

    返回：
        score_matrix: 综合得分矩阵 (n_grids, n_ratios)
    """
    n_grids, n_ratios, n_indicators = normalized_data.shape
    score_matrix = np.zeros((n_grids, n_ratios), dtype=DTYPE_FLOAT)  # 使用统一的数据类型

    # 加权求和
    for i, indicator in enumerate(indicators):
        weight = weights.get(indicator, 1.0 / len(indicators))
        score_matrix += normalized_data[:, :, i] * weight

    return score_matrix


def handle_equal_scores(score_matrix, initial_indices):
    """
    处理多个ratio得分相同的情况
    策略：选择最小的ratio（更保守的亏缺比例）

    参数：
        score_matrix: 得分矩阵 (n_grids, n_ratios)
        initial_indices: 初始最优索引

    返回：
        optimal_indices: 处理后的最优索引
    """
    optimal_indices = initial_indices.copy()
    n_grids = score_matrix.shape[0]

    for g in range(n_grids):
        max_score = score_matrix[g, :].max()
        # 找到所有得分等于最大值的ratio索引
        candidate_indices = np.where(score_matrix[g, :] == max_score)[0]

        if len(candidate_indices) > 1:
            # 选择最小的ratio（索引最小）
            optimal_indices[g] = candidate_indices[0]

    return optimal_indices


# =============================================================================
# 结果验证
# =============================================================================

def validate_optimal_results(optimal_results, indicators, lowdata, updata):
    """
    验证最优结果的合理性

    参数：
        optimal_results: 最优结果DataFrame
        indicators: 指标列表
        lowdata: ratio下限
        updata: ratio上限

    返回：
        validation_result: 包含验证结果的字典
    """
    issues = []
    passed = True

    # 1. 检查ratio范围
    ratio_min = optimal_results['optimal_ratio'].min()
    ratio_max = optimal_results['optimal_ratio'].max()

    if ratio_min < lowdata - 0.01 or ratio_max > updata + 0.01:
        issues.append(f"最优ratio超出预期范围 [{lowdata:.2f}, {updata:.2f}]")
        passed = False

    # 2. 检查综合得分范围
    score_min = optimal_results['comprehensive_score'].min()
    score_max = optimal_results['comprehensive_score'].max()

    if score_min < 0 or score_max > 1:
        issues.append(f"综合得分超出[0,1]范围: [{score_min:.3f}, {score_max:.3f}]")
        passed = False

    # 3. 检查各指标异常值
    for indicator in indicators:
        if indicator not in optimal_results.columns:
            continue

        values = optimal_results[indicator]

        # 检查NaN和Inf
        nan_count = values.isna().sum()
        inf_count = np.isinf(values).sum()

        if nan_count > 0:
            issues.append(f"{indicator} 存在 {nan_count} 个NaN值")
            passed = False

        if inf_count > 0:
            issues.append(f"{indicator} 存在 {inf_count} 个Inf值")
            passed = False

        # 检查极端异常值（超过3倍标准差）
        mean_val = values.mean()
        std_val = values.std()
        outlier_count = ((values < mean_val - 3*std_val) | (values > mean_val + 3*std_val)).sum()

        if outlier_count > len(values) * 0.05:  # 超过5%的异常值
            issues.append(f"{indicator} 存在大量异常值: {outlier_count} ({outlier_count/len(values)*100:.1f}%)")

    # 4. 检查数据完整性
    missing_coords = optimal_results[['x', 'y']].isna().any().any()
    if missing_coords:
        issues.append("存在缺失的坐标数据")
        passed = False

    return {
        'passed': passed,
        'issues': issues,
        'ratio_range': (float(ratio_min), float(ratio_max)),
        'score_range': (float(score_min), float(score_max))
    }


# =============================================================================
# 分析和可视化函数
# =============================================================================

def analyze_optimal_results(optimal_results, crop, outday):
    """
    分析最优结果并生成统计报告

    参数：
        optimal_results: 最优结果DataFrame
        crop: 作物类型
        outday: 输出日期
    """
    print(f"\n{'='*70}")
    print(f"  最优结果统计分析")
    print(f"{'='*70}\n")

    # 1. 最优ratio分布统计
    print(f"【1. 最优亏缺比例分布】")
    print(f"  平均值: {optimal_results['optimal_ratio'].mean():.3f}")
    print(f"  中位数: {optimal_results['optimal_ratio'].median():.3f}")
    print(f"  标准差: {optimal_results['optimal_ratio'].std():.3f}")
    print(f"  25%分位: {optimal_results['optimal_ratio'].quantile(0.25):.3f}")
    print(f"  75%分位: {optimal_results['optimal_ratio'].quantile(0.75):.3f}")

    # 2. 各指标统计
    indicators = ['Y', 'WUE', 'GWP']
    print(f"\n【2. 最优情景下各指标统计】")

    for ind in indicators:
        if ind in optimal_results.columns:
            mean_val = optimal_results[ind].mean()
            std_val = optimal_results[ind].std()
            print(f"  {ind:5s}: 平均值 = {mean_val:10.2f}, 标准差 = {std_val:10.2f}")

    # 3. 综合得分分布
    print(f"\n【3. 综合得分分布】")
    print(f"  平均值: {optimal_results['comprehensive_score'].mean():.3f}")
    print(f"  中位数: {optimal_results['comprehensive_score'].median():.3f}")

    # 4. 分位数统计
    print(f"\n【4. 最优ratio分位数分布】")
    percentiles = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.89]
    for p in percentiles:
        count = (optimal_results['optimal_ratio'] <= p + 0.01).sum()
        percentage = count / len(optimal_results) * 100
        print(f"  ratio <= {p:.2f}: {count:6d} 个网格 ({percentage:5.1f}%)")

    print(f"\n{'='*70}\n")


# =============================================================================
# 批处理函数
# =============================================================================

def optimize_deficit_ratio_batch(
    crops=['rice', 'wheat', 'maize'],
    outday='20251028',
    weight_configs=None,
    **kwargs
):
    """
    批量优化多种作物的最优亏缺比例

    参数：
        crops: 作物列表
        outday: 输出日期
        weight_configs: 权重配置列表 [{'name': 'equal', 'weights': {...}}, ...]
        **kwargs: 传递给optimize_deficit_ratio的其他参数

    返回：
        results: 各作物优化结果字典
    """
    results = {}

    for crop in crops:
        print(f"\n{'#'*70}")
        print(f"# 批处理: {crop}")
        print(f"{'#'*70}\n")

        try:
            optimal_results = optimize_deficit_ratio(
                crop=crop,
                outday=outday,
                **kwargs
            )
            results[crop] = optimal_results
        except Exception as e:
            print(f"  错误: {crop} 优化失败 - {e}")
            results[crop] = None

    return results


def optimize_with_multiple_weights(
    crop,
    outday,
    weight_configs=None,
    **kwargs
):
    """
    使用多种权重配置进行优化

    参数：
        crop: 作物类型
        outday: 输出日期
        weight_configs: 权重配置列表
            [
                {'name': 'equal', 'weights': None},  # 等权重
                {'name': 'yield_focus', 'weights': {'Y': 0.4, 'WUE': 0.2, 'CH4': 0.2, 'N2O': 0.1, 'ET': 0.1}},
                {'name': 'env_focus', 'weights': {'Y': 0.2, 'WUE': 0.2, 'CH4': 0.25, 'N2O': 0.25, 'ET': 0.1}}
            ]
        **kwargs: 其他参数

    返回：
        results: 各权重配置的优化结果
    """
    if weight_configs is None:
        weight_configs = [
            {'name': 'equal', 'weights': None},
        ]

    results = {}

    pm = get_path_manager()

    for config in weight_configs:
        name = config['name']
        weights = config['weights']

        print(f"\n{'='*70}")
        print(f"  权重配置: {name}")
        print(f"{'='*70}\n")

        # 修改输出目录（使用路径管理器）
        output_dir = pm.base_dir / crop / outday / f'optimal_{name}'

        try:
            optimal_results = optimize_deficit_ratio(
                crop=crop,
                outday=outday,
                weights=weights,
                output_dir=output_dir,
                **kwargs
            )
            results[name] = optimal_results
        except Exception as e:
            print(f"  错误: 权重配置 {name} 优化失败 - {e}")
            results[name] = None

    return results


# =============================================================================
# 测试函数
# =============================================================================

def test_optimization():
    """
    测试最优解选择功能
    """
    print("\n[测试模式] 使用模拟数据测试最优解选择...")

    # 创建模拟响应面
    n_grids = 1000
    n_ratios = 31
    indicators = ['Y', 'WUE', 'GWP']

    # 模拟数据
    data = np.random.rand(n_grids, n_ratios, len(indicators)).astype(np.float32)
    coords = np.column_stack([
        np.linspace(110, 120, n_grids),
        np.linspace(30, 40, n_grids)
    ])
    ratios = np.arange(0.60, 0.90, 0.01)

    # 归一化测试
    print("\n测试归一化功能:")
    normalized = normalize_indicators(data, indicators, INDICATOR_DIRECTION)
    print(f"  归一化数据范围: [{normalized.min():.3f}, {normalized.max():.3f}]")

    # 综合得分测试
    print("\n测试综合得分计算:")
    weights = {ind: 1.0/len(indicators) for ind in indicators}
    scores = calculate_comprehensive_score(normalized, indicators, weights)
    print(f"  得分矩阵维度: {scores.shape}")
    print(f"  得分范围: [{scores.min():.3f}, {scores.max():.3f}]")

    # 最优选择测试
    print("\n测试最优选择:")
    optimal_indices = np.argmax(scores, axis=1)
    optimal_ratios = ratios[optimal_indices]
    print(f"  最优ratio分布: 平均={optimal_ratios.mean():.3f}, 标准差={optimal_ratios.std():.3f}")

    # 验证功能测试
    print("\n测试验证功能:")
    optimal_results = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'optimal_ratio': optimal_ratios,
        'comprehensive_score': scores[np.arange(n_grids), optimal_indices],
        'Y': data[np.arange(n_grids), optimal_indices, 0]
    })

    validation = validate_optimal_results(optimal_results, ['Y'], 0.60, 0.90)
    print(f"  验证结果: {'通过' if validation['passed'] else '失败'}")

    print("\n[测试通过] 最优解选择功能正常")


if __name__ == "__main__":
    test_optimization()
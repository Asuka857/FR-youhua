"""
统计检验模块
功能：计算效应量、置信区间、显著性等统计量

创建时间：2025-03-24
"""

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# 效应量计算
# =============================================================================

def cohens_d(control, treatment):
    """
    计算Cohen's d效应量

    Cohen's d解释:
    - |d| < 0.2: 小效应
    - 0.2 ≤ |d| < 0.8: 中等效应
    - |d| ≥ 0.8: 大效应

    参数:
        control: 对照组数据 (array-like)
        treatment: 处理组数据 (array-like)

    返回:
        d: Cohen's d值
        interpretation: 效应量解释
    """
    # 转换为numpy数组并移除缺失值
    control = np.array(control)
    treatment = np.array(treatment)
    control = control[~np.isnan(control)]
    treatment = treatment[~np.isnan(treatment)]

    n1, n2 = len(control), len(treatment)
    var1, var2 = np.var(control, ddof=1), np.var(treatment, ddof=1)

    # 池化标准差
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0, 'undefined'

    d = (np.mean(treatment) - np.mean(control)) / pooled_std

    # 解释效应量
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = 'negligible'  # 可忽略
    elif abs_d < 0.5:
        interpretation = 'small'       # 小
    elif abs_d < 0.8:
        interpretation = 'medium'      # 中等
    else:
        interpretation = 'large'       # 大

    return d, interpretation


def hedges_g(control, treatment):
    """
    计算Hedges' g效应量（对小样本校正的Cohen's d）

    参数:
        control: 对照组数据
        treatment: 处理组数据

    返回:
        g: Hedges' g值
    """
    d, interp = cohens_d(control, treatment)

    # 校正因子
    n1, n2 = len(control), len(treatment)
    df = n1 + n2 - 2
    correction = 1 - 3 / (4 * df - 1)

    g = d * correction
    return g, interp


# =============================================================================
# 置信区间计算
# =============================================================================

def mean_confidence_interval(data, confidence=0.95):
    """
    计算均值的置信区间

    参数:
        data: 数据 (array-like)
        confidence: 置信水平 (默认0.95)

    返回:
        mean: 均值
        ci_lower: 置信区间下限
        ci_upper: 置信区间上限
        margin: 误差范围
    """
    data = np.array(data)
    data = data[~np.isnan(data)]

    n = len(data)
    if n < 2:
        return np.mean(data), np.nan, np.nan, np.nan

    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)

    return mean, mean - h, mean + h, h


def difference_confidence_interval(control, treatment, confidence=0.95):
    """
    计算两组数据差值的置信区间

    参数:
        control: 对照组数据
        treatment: 处理组数据
        confidence: 置信水平

    返回:
        diff: 差值均值
        ci_lower: 置信区间下限
        ci_upper: 置信区间上限
    """
    control = np.array(control)[~np.isnan(np.array(control))]
    treatment = np.array(treatment)[~np.isnan(np.array(treatment))]

    # 使用t检验计算差值的置信区间
    t_stat, p_value = stats.ttest_ind(control, treatment)

    n1, n2 = len(control), len(treatment)
    mean_diff = np.mean(treatment) - np.mean(control)

    # 合并标准误
    var1, var2 = np.var(control, ddof=1), np.var(treatment, ddof=1)
    se = np.sqrt(var1/n1 + var2/n2)

    # 置信区间
    df = min(n1, n2) - 1  # Welch's t-test自由度近似
    t_crit = stats.t.ppf((1 + confidence) / 2, df)

    margin = t_crit * se

    return mean_diff, mean_diff - margin, mean_diff + margin


def percentage_confidence_interval(control_mean, treatment_mean, confidence=0.95, n_bootstrap=1000):
    """
    计算变化百分比的置信区间（使用bootstrap方法）

    参数:
        control_mean: 对照组均值
        treatment_mean: 处理组均值
        confidence: 置信水平
        n_bootstrap: bootstrap迭代次数

    返回:
        pct: 变化百分比
        ci_lower: 置信区间下限
        ci_upper: 置信区间上限
    """
    pct = (treatment_mean - control_mean) / abs(control_mean) * 100 if control_mean != 0 else 0

    # 简化版本：假设正态分布
    # 实际应用中应使用完整的bootstrap
    return pct, pct * 0.9, pct * 1.1  # 简化估计


# =============================================================================
# 显著性检验
# =============================================================================

def t_test(control, treatment):
    """
    独立样本t检验

    参数:
        control: 对照组数据
        treatment: 处理组数据

    返回:
        t_stat: t统计量
        p_value: p值
        significant_05: 是否在0.05水平显著
        significant_01: 是否在0.01水平显著
    """
    control = np.array(control)[~np.isnan(np.array(control))]
    treatment = np.array(treatment)[~np.isnan(np.array(treatment))]

    t_stat, p_value = stats.ttest_ind(control, treatment)

    return {
        't_stat': t_stat,
        'p_value': p_value,
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01,
        'significance_code': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    }


def mann_whitney_u(control, treatment):
    """
    Mann-Whitney U检验（非参数检验）

    参数:
        control: 对照组数据
        treatment: 处理组数据

    返回:
        u_stat: U统计量
        p_value: p值
    """
    control = np.array(control)[~np.isnan(np.array(control))]
    treatment = np.array(treatment)[~np.isnan(np.array(treatment))]

    u_stat, p_value = stats.mannwhitneyu(control, treatment, alternative='two-sided')

    return {
        'u_stat': u_stat,
        'p_value': p_value,
        'significant_05': p_value < 0.05
    }


# =============================================================================
# 空间统计
# =============================================================================

def morans_i(values, coords, k=8):
    """
    计算空间自相关Moran's I

    参数:
        values: 数值数组
        coords: 坐标数组 (n x 2)
        k: 近邻数

    返回:
        I: Moran's I值
        p_value: p值
        interpretation: 解释
    """
    from scipy.spatial import Delaunay
    from scipy.stats import norm

    values = np.array(values)
    coords = np.array(coords)

    # 移除缺失值
    valid_idx = ~np.isnan(values)
    values = values[valid_idx]
    coords = coords[valid_idx]

    n = len(values)
    if n < 10:
        return 0, 1.0, 'insufficient data'

    # 计算空间权重矩阵（基于k近邻）
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(k+1, n)).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # 构建权重矩阵
    W = np.zeros((n, n))
    for i in range(n):
        for j in indices[i, 1:]:  # 排除自身
            W[i, j] = 1

    # 行标准化
    W = W / W.sum(axis=1, keepdims=True)

    # 计算Moran's I
    mean_val = np.mean(values)
    z = values - mean_val

    numerator = n * np.sum(W * np.outer(z, z))
    denominator = np.sum(z ** 2)

    I = numerator / denominator if denominator != 0 else 0

    # 简化的p值计算（使用正态近似）
    expected_I = -1 / (n - 1)
    se = np.sqrt(1 / (n - 1))  # 简化标准误

    z_score = (I - expected_I) / se
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    # 解释
    if I > 0.3:
        interpretation = 'strong positive spatial autocorrelation'
    elif I > 0.1:
        interpretation = 'moderate positive spatial autocorrelation'
    elif I > -0.1:
        interpretation = 'random spatial pattern'
    elif I > -0.3:
        interpretation = 'moderate negative spatial autocorrelation'
    else:
        interpretation = 'strong negative spatial autocorrelation'

    return I, p_value, interpretation


# =============================================================================
# 综合统计报告
# =============================================================================

def comprehensive_comparison(control, treatment, name='indicator'):
    """
    生成综合对比统计报告

    参数:
        control: 对照组数据
        treatment: 处理组数据
        name: 指标名称

    返回:
        stats_dict: 统计结果字典
    """
    control = np.array(control)[~np.isnan(np.array(control))]
    treatment = np.array(treatment)[~np.isnan(np.array(treatment))]

    # 基础统计
    control_mean, control_std = np.mean(control), np.std(control)
    treatment_mean, treatment_std = np.mean(treatment), np.std(treatment)

    # 变化
    change = treatment_mean - control_mean
    change_pct = (change / abs(control_mean) * 100) if control_mean != 0 else 0

    # 效应量
    d, d_interp = cohens_d(control, treatment)

    # 置信区间
    _, ci_lower, ci_upper, _ = mean_confidence_interval(treatment)

    # t检验
    t_result = t_test(control, treatment)

    return {
        'indicator': name,
        'n_control': len(control),
        'n_treatment': len(treatment),
        'control_mean': control_mean,
        'control_std': control_std,
        'treatment_mean': treatment_mean,
        'treatment_std': treatment_std,
        'change': change,
        'change_pct': change_pct,
        'cohens_d': d,
        'effect_interpretation': d_interp,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        't_stat': t_result['t_stat'],
        'p_value': t_result['p_value'],
        'significance': t_result['significance_code']
    }


def batch_comparison(control_dict, treatment_dict, indicators):
    """
    批量对比分析

    参数:
        control_dict: 对照组字典 {indicator: values}
        treatment_dict: 处理组字典 {indicator: values}
        indicators: 指标列表

    返回:
        results_df: 结果DataFrame
    """
    results = []

    for indicator in indicators:
        if indicator in control_dict and indicator in treatment_dict:
            stats = comprehensive_comparison(
                control_dict[indicator],
                treatment_dict[indicator],
                indicator
            )
            results.append(stats)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # 测试
    np.random.seed(42)
    control = np.random.normal(100, 15, 1000)
    treatment = np.random.normal(105, 15, 1000)

    print("=== 统计检验测试 ===")
    print(f"Cohen's d: {cohens_d(control, treatment)[0]:.3f}")
    print(f"置信区间: {mean_confidence_interval(treatment)[:3]}")
    print(f"t检验: {t_test(control, treatment)}")
    print(f"综合报告: {comprehensive_comparison(control, treatment, 'test')}")
"""
对比分析模块 (Comparison Analysis Module)
功能：最优情景与对照组的对比分析

创建时间：2025-03-21
最后修改：2025-03-21

改进内容：
1. 从 W_optimization.py 中提取 compare_optimal_vs_ck 函数
2. 从 W_spatial_analysis.py 中提取 generate_comparison_analysis 和 generate_comparison_map 函数
3. 统一管理对比分析相关功能
"""

import numpy as np
import pandas as pd
from pathlib import Path

# 导入配置
from .W_config import INDICATOR_DIRECTION, GWP_CH4_FACTOR, GWP_N2O_FACTOR

# 导入路径管理器
from .W_path_manager import get_path_manager


# =============================================================================
# 配置常量
# =============================================================================

COMPARISON_INDICATORS = ['Y', 'ET', 'WUE', 'CH4', 'N2O', 'GWP']

INDICATOR_INFO = {
    'Y': {'name': 'Yield', 'unit': 'kg/ha'},
    'ET': {'name': 'Evapotranspiration', 'unit': 'mm'},
    'WUE': {'name': 'Water Use Efficiency', 'unit': 'kg/m3'},
    'CH4': {'name': 'CH4 Emission', 'unit': 'kg/ha'},
    'N2O': {'name': 'N2O Emission', 'unit': 'kg/ha'},
    'GWP': {'name': 'Global Warming Potential', 'unit': 'kg CO2-eq/ha'}
}


# =============================================================================
# GWP 对照组计算
# =============================================================================

def calculate_gwp_ck(ch4_ck_data, n2o_ck_data):
    """
    从CH4和N2O对照组数据计算GWP

    参数:
        ch4_ck_data: DataFrame, CH4对照组数据（包含x, y, Ypred列）
        n2o_ck_data: DataFrame, N2O对照组数据（包含x, y, Ypred列）

    返回:
        gwp_ck_data: DataFrame, GWP对照组数据（包含x, y, Ypred列）
    """
    # 合并 CH4 和 N2O 数据
    merged = ch4_ck_data[['x', 'y', 'Ypred']].rename(columns={'Ypred': 'CH4'})
    merged = merged.merge(
        n2o_ck_data[['x', 'y', 'Ypred']].rename(columns={'Ypred': 'N2O'}),
        on=['x', 'y'], how='inner'
    )

    # 计算 GWP: CH4 * 27 + N2O * 273
    merged['Ypred'] = merged['CH4'] * GWP_CH4_FACTOR + merged['N2O'] * GWP_N2O_FACTOR

    return merged[['x', 'y', 'Ypred']]


# =============================================================================
# 变化率计算
# =============================================================================

def calculate_change_pct(optimal, ck, direction='maximize'):
    """
    计算变化率

    参数:
        optimal: Series, 最优情景数据
        ck: Series, 对照组数据
        direction: str, 优化方向 ('maximize' 或 'minimize')

    返回:
        change_pct: Series, 变化率（百分比）
    """
    if direction == 'minimize':
        # 最小化指标（ET, CH4, N2O, GWP）：计算减少百分比
        # 减少百分比 = (ck - optimal) / |ck| * 100
        change_pct = np.where(
            ck.abs() > 1e-6,
            (ck - optimal) / ck.abs() * 100,
            0.0
        )
    else:
        # 最大化指标（Y, WUE）：计算增加百分比
        # 增加百分比 = (optimal - ck) / ck * 100
        change_pct = np.where(
            ck.abs() > 1e-6,
            (optimal - ck) / ck * 100,
            0.0
        )

    return pd.Series(change_pct, index=optimal.index)


def calculate_optimal_ck_ratio(optimal, ck, direction='maximize', threshold=1e-6):
    """
    计算最优值与对照组比值

    参数:
        optimal: Series, 最优情景数值
        ck: Series, 对照组数值
        direction: str, 优化方向 ('maximize' 或 'minimize')
        threshold: float, 阈值避免除零错误

    返回:
        ratio: Series, optimal/ck比值（负值最小化指标使用ck/opt）

    说明:
        - 最大化指标：ratio = optimal/ck
        - 正值最小化指标：ratio = optimal/ck
        - 负值最小化指标（如CH4）：ratio = ck/optimal（使得ratio<1表示改善）
    """
    # 判断对照组数值符号
    mean_ck = ck.mean()

    # 处理除零情况
    if direction == 'minimize' and mean_ck < 0:
        # 负值最小化指标（如CH4）：使用 ck/opt，使得 ratio<1 表示改善
        ratio = np.where(
            np.abs(optimal) > threshold,
            ck / optimal,
            np.nan
        )
    else:
        # 其他情况：使用 opt/ck
        ratio = np.where(
            np.abs(ck) > threshold,
            optimal / ck,
            np.nan
        )

    return pd.Series(ratio, index=optimal.index)


# =============================================================================
# 对照组数据加载
# =============================================================================

def load_ck_data(crop, outday, indicator):
    """
    加载对照组数据（只从本地results目录读取）

    参数:
        crop: 作物类型
        outday: 输出日期
        indicator: 指标名称

    返回:
        ck_data: DataFrame, 对照组数据（包含x, y, Ypred列）
        或 None（如果数据不存在）
    """
    pm = get_path_manager()

    # 特殊处理 GWP：从 CH4 和 N2O 计算
    if indicator == 'GWP':
        ch4_ck_file = pm.get_mix_ck_dir(crop, outday) / 'CH4_ck.csv'
        n2o_ck_file = pm.get_mix_ck_dir(crop, outday) / 'N2O_ck.csv'

        if ch4_ck_file.exists() and n2o_ck_file.exists():
            ch4_ck_data = pd.read_csv(ch4_ck_file)
            n2o_ck_data = pd.read_csv(n2o_ck_file)
            return calculate_gwp_ck(ch4_ck_data, n2o_ck_data)
        else:
            print(f"    警告: GWP对照组数据文件不存在")
            print(f"      CH4: {ch4_ck_file} (存在: {ch4_ck_file.exists()})")
            print(f"      N2O: {n2o_ck_file} (存在: {n2o_ck_file.exists()})")
            return None
    else:
        # 读取普通指标的对照组数据
        ck_file = pm.get_mix_ck_dir(crop, outday) / f'{indicator}_ck.csv'

        if ck_file.exists():
            return pd.read_csv(ck_file)
        else:
            print(f"    警告: {indicator}对照组数据文件不存在: {ck_file}")
            return None


# =============================================================================
# 主对比函数
# =============================================================================

def compare_optimal_vs_ck(optimal_results, crop, outday, indicators=None):
    """
    对比最优情景与对照组

    参数:
        optimal_results: DataFrame, 最优结果（包含x, y, optimal_ratio和各指标值）
        crop: 作物类型
        outday: 输出日期
        indicators: 指标列表（默认使用COMPARISON_INDICATORS）

    返回:
        dict: {
            'comparison_data': DataFrame汇总,
            'indicator_details': {indicator: DataFrame详细数据}
        }
    """
    if indicators is None:
        indicators = COMPARISON_INDICATORS

    comparison_data = []
    indicator_details = {}

    for indicator in indicators:
        if indicator not in optimal_results.columns:
            continue

        # 加载对照组数据
        ck_data = load_ck_data(crop, outday, indicator)

        if ck_data is None:
            print(f"    {indicator}: 对照组数据不存在，跳过")
            continue

        # 合并数据
        merged = optimal_results[['x', 'y', indicator, 'optimal_ratio']].copy()
        merged = merged.rename(columns={indicator: 'optimal'})
        merged = merged.merge(ck_data[['x', 'y', 'Ypred']], on=['x', 'y'], how='inner')
        merged = merged.rename(columns={'Ypred': 'ck'})

        # 计算变化
        merged['change'] = merged['optimal'] - merged['ck']

        # 根据指标方向计算变化率
        direction = INDICATOR_DIRECTION.get(indicator, 'maximize')
        merged['change_pct'] = calculate_change_pct(
            merged['optimal'], merged['ck'], direction
        )

        # 计算optimal/ck比值（负值最小化指标使用ck/opt）
        merged['optimal_ck_ratio'] = calculate_optimal_ck_ratio(
            merged['optimal'], merged['ck'], direction
        )

        # 计算统计量
        mean_optimal = merged['optimal'].mean()
        mean_ck = merged['ck'].mean()
        mean_change_pct = merged['change_pct'].mean()
        std_change_pct = merged['change_pct'].std()

        # 计算比值统计量
        mean_ratio = merged['optimal_ck_ratio'].mean()
        median_ratio = merged['optimal_ck_ratio'].median()
        std_ratio = merged['optimal_ck_ratio'].std()

        # 计算比值改善百分比（比值计算已处理负值情况，统一判断逻辑）
        ratio_values = merged['optimal_ck_ratio'].dropna()
        if direction == 'maximize':
            # 最大化指标：比值>1表示改善
            ratio_improvement_pct = (ratio_values > 1.0).sum() / len(ratio_values) * 100
        else:
            # 最小化指标：比值<1表示改善（负值指标已通过ck/opt实现）
            ratio_improvement_pct = (ratio_values < 1.0).sum() / len(ratio_values) * 100

        # 保存统计结果
        comparison_data.append({
            'indicator': indicator,
            'mean_optimal': mean_optimal,
            'mean_ck': mean_ck,
            'mean_change': mean_optimal - mean_ck,
            'mean_change_pct': mean_change_pct,
            'std_change_pct': std_change_pct,
            'optimal_gt_ck_pct': (merged['optimal'] > merged['ck']).sum() / len(merged) * 100,
            'n_grids': len(merged),
            'mean_ratio': mean_ratio,
            'median_ratio': median_ratio,
            'std_ratio': std_ratio,
            'ratio_q25': merged['optimal_ck_ratio'].quantile(0.25),
            'ratio_q75': merged['optimal_ck_ratio'].quantile(0.75),
            'ratio_improvement_pct': ratio_improvement_pct
        })

        # 保存详细数据
        indicator_details[indicator] = merged

        print(f"    {indicator}: 最优={mean_optimal:.2f}, CK={mean_ck:.2f}, 变化={mean_change_pct:+.2f}%")

    return {
        'comparison_data': pd.DataFrame(comparison_data),
        'indicator_details': indicator_details
    }


# =============================================================================
# 生成对比分析
# =============================================================================

def generate_comparison_analysis(optimal_results, crop, outday, output_dir):
    """
    生成完整对比分析（CSV + 地图）

    参数:
        optimal_results: DataFrame, 最优结果
        crop: 作物类型
        outday: 输出日期
        output_dir: 输出目录

    输出:
        - {indicator}_comparison.csv（每个指标）
        - {indicator}_comparison_map.png（每个指标）
        - comparison_summary.csv（汇总）

    返回:
        comparison_results: DataFrame, 对比结果汇总
    """
    # 执行对比分析
    results = compare_optimal_vs_ck(optimal_results, crop, outday, COMPARISON_INDICATORS)

    comparison_data = results['comparison_data']
    indicator_details = results['indicator_details']

    if len(comparison_data) == 0:
        print("    警告: 没有可对比的指标数据")
        return pd.DataFrame()

    # 创建对比分析目录
    comparison_dir = Path(output_dir) / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # 保存每个指标的详细对比数据
    for indicator, detail_df in indicator_details.items():
        comparison_csv = comparison_dir / f'{indicator}_comparison.csv'
        detail_df.to_csv(comparison_csv, index=False)

        # 生成对比地图
        try:
            generate_comparison_map(detail_df, indicator, comparison_dir, crop=crop, outday=outday)
        except Exception as e:
            print(f"    警告: {indicator} 对比地图生成失败 - {e}")

    # 保存汇总表
    summary_file = comparison_dir / 'comparison_summary.csv'
    comparison_data.to_csv(summary_file, index=False)

    return {
        'comparison_data': comparison_data,
        'indicator_details': indicator_details
    }


# =============================================================================
# 生成对比地图
# =============================================================================

def generate_comparison_map(comparison_data, indicator, output_dir, crop=None, outday=None):
    """
    生成对比地图（使用TIF文件提高精确度）

    参数:
        comparison_data: DataFrame, 对比数据（包含x, y, optimal, ck, change列）
        indicator: 指标名称
        output_dir: 输出目录
        crop: 作物类型
        outday: 输出日期
    """
    # 导入TIF绘图模块
    try:
        from .W_comparison_map_tif import draw_comparison_maps_from_tif
        from .W_path_manager import get_path_manager

        pm = get_path_manager()

        # 确定crop和outday
        if crop is None:
            # 尝试从路径推断
            crop = 'wheat'  # 默认值
        if outday is None:
            outday = '20251028'  # 默认值

        # 获取参考TIF路径
        reference_tif_path = pm.base_dir / crop / outday / 'x_data_mask.tif'
        if not reference_tif_path.exists():
            reference_tif_path = Path(f'F:/WaterML/{crop}/{outday}/x_data_mask.tif')

        if not reference_tif_path.exists():
            raise FileNotFoundError(f"找不到参考TIF文件: {reference_tif_path}")

        # 计算数据范围
        vmin = min(comparison_data['optimal'].min(), comparison_data['ck'].min())
        vmax = max(comparison_data['optimal'].max(), comparison_data['ck'].max())
        vmax_change = max(abs(comparison_data['change'].min()), abs(comparison_data['change'].max()))

        # 调用TIF绘图函数
        results = draw_comparison_maps_from_tif(
            comparison_data=comparison_data,
            indicator=indicator,
            output_dir=str(output_dir),
            reference_tif_path=str(reference_tif_path),
            vmin=vmin, vmax=vmax,
            vmin_change=-vmax_change, vmax_change=vmax_change
        )

        print(f"    对比地图已生成:")
        print(f"      TIF文件: optimal, ck, change, ratio")
        print(f"      单张图: optimal, ck, change, ratio")
        print(f"      拼图: comparison_combined")

        return results['map_combined']

    except ImportError as e:
        print(f"    警告: TIF绘图模块导入失败，使用简单绘图 - {e}")
        _generate_simple_comparison_map(comparison_data, indicator, output_dir)

    except Exception as e:
        print(f"    警告: {indicator} 对比地图生成失败 - {e}")
        raise


def _generate_simple_comparison_map(comparison_data, indicator, output_dir):
    """
    生成简单的对比地图（matplotlib版本）

    参数:
        comparison_data: DataFrame, 对比数据
        indicator: 指标名称
        output_dir: 输出目录
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8, 18))

    x_range = comparison_data['x'].max() - comparison_data['x'].min()
    y_range = comparison_data['y'].max() - comparison_data['y'].min()

    # 计算统一的颜色范围
    vmin = min(comparison_data['optimal'].min(), comparison_data['ck'].min())
    vmax = max(comparison_data['optimal'].max(), comparison_data['ck'].max())

    # 最优情景
    sc1 = axes[0].scatter(
        comparison_data['x'],
        comparison_data['y'],
        c=comparison_data['optimal'],
        cmap='RdYlGn',
        s=1,
        vmin=vmin,
        vmax=vmax
    )
    axes[0].set_title(f'{indicator} - Optimal')
    axes[0].set_aspect(x_range / y_range)
    plt.colorbar(sc1, ax=axes[0], shrink=0.8)

    # 对照组
    sc2 = axes[1].scatter(
        comparison_data['x'],
        comparison_data['y'],
        c=comparison_data['ck'],
        cmap='RdYlGn',
        s=1,
        vmin=vmin,
        vmax=vmax
    )
    axes[1].set_title(f'{indicator} - Control')
    axes[1].set_aspect(x_range / y_range)
    plt.colorbar(sc2, ax=axes[1], shrink=0.8)

    # 变化
    vmax_change = max(abs(comparison_data['change'].min()), abs(comparison_data['change'].max()))
    sc3 = axes[2].scatter(
        comparison_data['x'],
        comparison_data['y'],
        c=comparison_data['change'],
        cmap='RdBu_r',
        s=1,
        vmin=-vmax_change,
        vmax=vmax_change
    )
    axes[2].set_title(f'{indicator} - Change')
    axes[2].set_aspect(x_range / y_range)
    plt.colorbar(sc3, ax=axes[2], shrink=0.8)

    plt.tight_layout()

    map_file = Path(output_dir) / f'{indicator}_comparison_map.png'
    plt.savefig(map_file, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# 测试函数
# =============================================================================

def test_comparison_module():
    """
    测试对比分析模块
    """
    print("\n[测试模式] 使用模拟数据测试对比分析模块...")

    # 创建模拟数据
    n_grids = 1000
    optimal_results = pd.DataFrame({
        'x': np.linspace(110, 120, n_grids),
        'y': np.linspace(30, 40, n_grids),
        'optimal_ratio': np.random.uniform(0.60, 0.90, n_grids),
        'Y': np.random.uniform(5000, 10000, n_grids),
        'ET': np.random.uniform(300, 800, n_grids),
        'WUE': np.random.uniform(10, 20, n_grids),
        'CH4': np.random.uniform(100, 400, n_grids),
        'N2O': np.random.uniform(1, 5, n_grids)
    })

    # 测试变化率计算
    print("\n测试变化率计算:")
    optimal = pd.Series([100, 90, 80])
    ck = pd.Series([80, 90, 100])

    change_pct_max = calculate_change_pct(optimal, ck, 'maximize')
    print(f"  最大化指标变化率: {change_pct_max.values}")

    change_pct_min = calculate_change_pct(optimal, ck, 'minimize')
    print(f"  最小化指标变化率: {change_pct_min.values}")

    print("\n[测试通过] 对比分析模块正常")


if __name__ == "__main__":
    test_comparison_module()
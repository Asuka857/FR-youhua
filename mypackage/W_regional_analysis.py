"""
区域分析模块 (Regional Analysis Module)
功能：按大洲、国家、气候带等区域划分进行统计分析

创建时间：2025-03-20
最后修改：2025-03-20
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict


# =============================================================================
# 区域定义
# =============================================================================

# 大洲边界定义（简化版，基于经纬度范围）
CONTINENT_BOUNDS = {
    'Asia': {'lon_min': 25, 'lon_max': 180, 'lat_min': -10, 'lat_max': 80},
    'Europe': {'lon_min': -25, 'lon_max': 60, 'lat_min': 35, 'lat_max': 80},
    'Africa': {'lon_min': -20, 'lon_max': 55, 'lat_min': -35, 'lat_max': 37},
    'North America': {'lon_min': -170, 'lon_max': -50, 'lat_min': 10, 'lat_max': 85},
    'South America': {'lon_min': -82, 'lon_max': -34, 'lat_min': -56, 'lat_max': 12},
    'Oceania': {'lon_min': 110, 'lon_max': 180, 'lat_min': -50, 'lat_max': 0}
}

# 主要水稻种植区域定义（更精确）
RICE_REGIONS = {
    'East Asia': {'lon_min': 100, 'lon_max': 145, 'lat_min': 20, 'lat_max': 50},
    'Southeast Asia': {'lon_min': 90, 'lon_max': 140, 'lat_min': -10, 'lat_max': 25},
    'South Asia': {'lon_min': 60, 'lon_max': 100, 'lat_min': 5, 'lat_max': 40},
    'Central Asia': {'lon_min': 50, 'lon_max': 90, 'lat_min': 35, 'lat_max': 55},
    'Middle East': {'lon_min': 25, 'lon_max': 65, 'lat_min': 12, 'lat_max': 45},
    'Mediterranean': {'lon_min': -10, 'lon_max': 40, 'lat_min': 30, 'lat_max': 45},
    'West Africa': {'lon_min': -20, 'lon_max': 20, 'lat_min': 5, 'lat_max': 20},
    'East Africa': {'lon_min': 30, 'lon_max': 50, 'lat_min': -15, 'lat_max': 15},
    'Central America': {'lon_min': -120, 'lon_max': -80, 'lat_min': 10, 'lat_max': 30},
    'South America': {'lon_min': -82, 'lon_max': -34, 'lat_min': -35, 'lat_max': 10},
    'North America': {'lon_min': -125, 'lon_max': -75, 'lat_min': 25, 'lat_max': 45}
}

# 气候带定义（基于纬度）
CLIMATE_ZONES = {
    'Tropical': {'lat_min': -23.5, 'lat_max': 23.5},
    'Subtropical': {'lat_min': 23.5, 'lat_max': 35, 'lat_min_neg': -35, 'lat_max_neg': -23.5},
    'Temperate': {'lat_min': 35, 'lat_max': 66.5, 'lat_min_neg': -66.5, 'lat_max_neg': -35}
}


# =============================================================================
# 主分析函数
# =============================================================================

def calculate_regional_statistics(
    optimal_results,
    crop,
    outday,
    region_type='rice_regions',
    indicators=['Y', 'ET', 'WUE', 'CH4', 'N2O']
):
    """
    按区域计算统计数据

    参数：
        optimal_results: 最优结果DataFrame
        crop: 作物类型
        outday: 输出日期
        region_type: 区域类型 ('continents', 'rice_regions', 'climate_zones')
        indicators: 统计指标列表

    返回：
        regional_stats: 区域统计DataFrame
    """
    print(f"  计算区域统计 (区域类型: {region_type})...")

    # 选择区域定义
    if region_type == 'continents':
        regions = CONTINENT_BOUNDS
    elif region_type == 'rice_regions':
        regions = RICE_REGIONS
    elif region_type == 'climate_zones':
        regions = CLIMATE_ZONES
    else:
        raise ValueError(f"未知的区域类型: {region_type}")

    # 为每个网格分配区域
    optimal_results = assign_region_to_grids(optimal_results, regions, region_type)

    # 按区域分组计算统计
    stats_list = []

    for region_name in optimal_results['region'].unique():
        if region_name == 'Other':
            continue

        region_data = optimal_results[optimal_results['region'] == region_name]

        if len(region_data) == 0:
            continue

        stats = {
            'region': region_name,
            'n_grids': len(region_data),
            'grid_pct': len(region_data) / len(optimal_results) * 100
        }

        # 最优ratio统计
        stats['optimal_ratio_mean'] = region_data['optimal_ratio'].mean()
        stats['optimal_ratio_std'] = region_data['optimal_ratio'].std()
        stats['optimal_ratio_median'] = region_data['optimal_ratio'].median()
        stats['optimal_ratio_q25'] = region_data['optimal_ratio'].quantile(0.25)
        stats['optimal_ratio_q75'] = region_data['optimal_ratio'].quantile(0.75)

        # 综合得分统计
        stats['score_mean'] = region_data['comprehensive_score'].mean()
        stats['score_std'] = region_data['comprehensive_score'].std()

        # 各指标统计
        for indicator in indicators:
            if indicator in region_data.columns:
                stats[f'{indicator}_mean'] = region_data[indicator].mean()
                stats[f'{indicator}_std'] = region_data[indicator].std()
                stats[f'{indicator}_median'] = region_data[indicator].median()

        stats_list.append(stats)

    # 转换为DataFrame
    regional_stats = pd.DataFrame(stats_list)

    # 按网格数量排序
    regional_stats = regional_stats.sort_values('n_grids', ascending=False)

    return regional_stats


def assign_region_to_grids(df, regions, region_type='continents'):
    """
    为每个网格分配所属区域

    参数：
        df: 包含x, y坐标的DataFrame
        regions: 区域定义字典
        region_type: 区域类型

    返回：
        df: 添加了region列的DataFrame
    """
    df = df.copy()
    df['region'] = 'Other'

    if region_type == 'climate_zones':
        # 气候带特殊处理
        for idx, row in df.iterrows():
            lat = abs(row['y'])

            if lat <= 23.5:
                df.loc[idx, 'region'] = 'Tropical'
            elif lat <= 35:
                df.loc[idx, 'region'] = 'Subtropical'
            elif lat <= 66.5:
                df.loc[idx, 'region'] = 'Temperate'
    else:
        # 基于边界框的区域分配
        for region_name, bounds in regions.items():
            lon_min = bounds['lon_min']
            lon_max = bounds['lon_max']
            lat_min = bounds['lat_min']
            lat_max = bounds['lat_max']

            # 处理跨180度经线的情况
            if lon_min > lon_max:
                # 跨越180度经线
                mask = ((df['x'] >= lon_min) | (df['x'] <= lon_max)) & \
                       (df['y'] >= lat_min) & (df['y'] <= lat_max)
            else:
                mask = (df['x'] >= lon_min) & (df['x'] <= lon_max) & \
                       (df['y'] >= lat_min) & (df['y'] <= lat_max)

            # 只更新尚未分配的网格
            df.loc[mask & (df['region'] == 'Other'), 'region'] = region_name

    return df


# =============================================================================
# 详细区域分析
# =============================================================================

def detailed_regional_analysis(
    optimal_results,
    crop,
    outday,
    indicators=['Y', 'ET', 'WUE', 'CH4', 'N2O']
):
    """
    详细的区域分析，包括大洲和种植区域

    参数：
        optimal_results: 最优结果DataFrame
        crop: 作物类型
        outday: 输出日期
        indicators: 指标列表

    返回：
        analysis_results: 分析结果字典
    """
    print(f"\n{'='*70}")
    print(f"  详细区域分析")
    print(f"{'='*70}\n")

    results = {}

    # 1. 大洲级统计
    print(f"[1/3] 大洲级统计...")
    results['continents'] = calculate_regional_statistics(
        optimal_results, crop, outday, 'continents', indicators
    )
    print(f"  已计算 {len(results['continents'])} 个大洲的统计")

    # 2. 主要种植区域统计
    print(f"\n[2/3] 主要种植区域统计...")
    results['rice_regions'] = calculate_regional_statistics(
        optimal_results, crop, outday, 'rice_regions', indicators
    )
    print(f"  已计算 {len(results['rice_regions'])} 个种植区域的统计")

    # 3. 气候带统计
    print(f"\n[3/3] 气候带统计...")
    results['climate_zones'] = calculate_regional_statistics(
        optimal_results, crop, outday, 'climate_zones', indicators
    )
    print(f"  已计算 {len(results['climate_zones'])} 个气候带的统计")

    # 4. 区域差异分析
    print(f"\n[区域差异分析]")
    analyze_regional_differences(results, indicators)

    return results


def analyze_regional_differences(results, indicators):
    """
    分析区域间差异

    参数：
        results: 各类区域统计结果
        indicators: 指标列表
    """
    # 最优ratio区域差异
    if 'rice_regions' in results and len(results['rice_regions']) > 0:
        rice_stats = results['rice_regions']

        max_ratio_region = rice_stats.loc[rice_stats['optimal_ratio_mean'].idxmax(), 'region']
        min_ratio_region = rice_stats.loc[rice_stats['optimal_ratio_mean'].idxmin(), 'region']
        max_ratio = rice_stats['optimal_ratio_mean'].max()
        min_ratio = rice_stats['optimal_ratio_mean'].min()

        print(f"  最优ratio最高区域: {max_ratio_region} ({max_ratio:.3f})")
        print(f"  最优ratio最低区域: {min_ratio_region} ({min_ratio:.3f})")
        print(f"  区域差异: {max_ratio - min_ratio:.3f}")

    # 产量区域差异
    if 'Y_mean' in rice_stats.columns:
        max_yield_region = rice_stats.loc[rice_stats['Y_mean'].idxmax(), 'region']
        min_yield_region = rice_stats.loc[rice_stats['Y_mean'].idxmin(), 'region']
        max_yield = rice_stats['Y_mean'].max()
        min_yield = rice_stats['Y_mean'].min()

        print(f"\n  产量最高区域: {max_yield_region} ({max_yield:.1f})")
        print(f"  产量最低区域: {min_yield_region} ({min_yield:.1f})")


# =============================================================================
# 区域对比分析
# =============================================================================

def compare_regions(
    optimal_results,
    crop,
    outday,
    region_names=None,
    indicators=['Y', 'ET', 'WUE', 'CH4', 'N2O']
):
    """
    对比指定区域的指标差异

    参数：
        optimal_results: 最优结果DataFrame
        crop: 作物类型
        outday: 输出日期
        region_names: 要对比的区域名称列表
        indicators: 指标列表

    返回：
        comparison_df: 对比结果DataFrame
    """
    # 分配区域
    optimal_results = assign_region_to_grids(optimal_results, RICE_REGIONS, 'rice_regions')

    if region_names is None:
        # 默认对比几个主要区域
        region_names = ['East Asia', 'Southeast Asia', 'South Asia']

    comparison_data = []

    for region in region_names:
        region_data = optimal_results[optimal_results['region'] == region]

        if len(region_data) == 0:
            continue

        stats = {
            'region': region,
            'n_grids': len(region_data),
            'optimal_ratio': region_data['optimal_ratio'].mean()
        }

        for indicator in indicators:
            if indicator in region_data.columns:
                stats[indicator] = region_data[indicator].mean()

        comparison_data.append(stats)

    comparison_df = pd.DataFrame(comparison_data)

    # 打印对比结果
    print(f"\n{'='*70}")
    print(f"  区域对比分析")
    print(f"{'='*70}\n")

    print(comparison_df.to_string(index=False))

    return comparison_df


# =============================================================================
# 国家级分析（简化版）
# =============================================================================

def analyze_by_country_simple(
    optimal_results,
    crop,
    outday,
    country_bounds=None,
    indicators=['Y', 'ET', 'WUE', 'CH4', 'N2O']
):
    """
    简化的国家级分析（基于边界框）

    参数：
        optimal_results: 最优结果DataFrame
        crop: 作物类型
        outday: 输出日期
        country_bounds: 国家边界定义字典
        indicators: 指标列表

    返回：
        country_stats: 国家统计DataFrame
    """
    if country_bounds is None:
        # 默认几个主要水稻生产国的简化边界
        country_bounds = {
            'China': {'lon_min': 73, 'lon_max': 135, 'lat_min': 18, 'lat_max': 54},
            'India': {'lon_min': 68, 'lon_max': 97, 'lat_min': 6, 'lat_max': 36},
            'Indonesia': {'lon_min': 95, 'lon_max': 141, 'lat_min': -11, 'lat_max': 6},
            'Bangladesh': {'lon_min': 88, 'lon_max': 93, 'lat_min': 20, 'lat_max': 27},
            'Vietnam': {'lon_min': 102, 'lon_max': 110, 'lat_min': 8, 'lat_max': 24},
            'Thailand': {'lon_min': 97, 'lon_max': 106, 'lat_min': 5, 'lat_max': 21},
            'Myanmar': {'lon_min': 92, 'lon_max': 102, 'lat_min': 10, 'lat_max': 29},
            'Philippines': {'lon_min': 117, 'lon_max': 127, 'lat_min': 5, 'lat_max': 20},
            'Japan': {'lon_min': 129, 'lon_max': 146, 'lat_min': 31, 'lat_max': 46},
            'Brazil': {'lon_min': -74, 'lon_max': -34, 'lat_min': -34, 'lat_max': 5},
            'USA': {'lon_min': -125, 'lon_max': -66, 'lat_min': 25, 'lat_max': 49}
        }

    # 计算国家级统计
    stats_list = []

    for country_name, bounds in country_bounds.items():
        mask = (optimal_results['x'] >= bounds['lon_min']) & \
               (optimal_results['x'] <= bounds['lon_max']) & \
               (optimal_results['y'] >= bounds['lat_min']) & \
               (optimal_results['y'] <= bounds['lat_max'])

        country_data = optimal_results[mask]

        if len(country_data) == 0:
            continue

        stats = {
            'country': country_name,
            'n_grids': len(country_data),
            'optimal_ratio_mean': country_data['optimal_ratio'].mean(),
            'optimal_ratio_std': country_data['optimal_ratio'].std()
        }

        for indicator in indicators:
            if indicator in country_data.columns:
                stats[f'{indicator}_mean'] = country_data[indicator].mean()
                stats[f'{indicator}_std'] = country_data[indicator].std()

        stats_list.append(stats)

    country_stats = pd.DataFrame(stats_list)
    country_stats = country_stats.sort_values('n_grids', ascending=False)

    return country_stats


# =============================================================================
# 辅助函数
# =============================================================================

def get_region_summary(regional_stats, top_n=10):
    """
    获取区域统计摘要

    参数：
        regional_stats: 区域统计DataFrame
        top_n: 显示前N个区域

    返回：
        summary_str: 摘要字符串
    """
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  区域统计摘要 (Top {top_n})")
    lines.append(f"{'='*70}\n")

    # 表头
    lines.append(f"{'区域':<20} {'网格数':>10} {'最优Ratio':>12} {'综合得分':>12}")
    lines.append("-" * 70)

    # 数据行
    for _, row in regional_stats.head(top_n).iterrows():
        lines.append(
            f"{row['region']:<20} {row['n_grids']:>10,} "
            f"{row['optimal_ratio_mean']:>12.3f} {row['score_mean']:>12.3f}"
        )

    return '\n'.join(lines)


def export_regional_stats(regional_stats, output_dir, filename='regional_statistics.csv'):
    """
    导出区域统计结果

    参数：
        regional_stats: 区域统计DataFrame
        output_dir: 输出目录
        filename: 输出文件名
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    regional_stats.to_csv(output_path, index=False)
    print(f"区域统计已导出到: {output_path}")


# =============================================================================
# 测试函数
# =============================================================================

def test_regional_analysis():
    """
    测试区域分析功能
    """
    print("\n[测试模式] 使用模拟数据测试区域分析...")

    # 创建模拟数据
    n_grids = 10000
    optimal_results = pd.DataFrame({
        'x': np.random.uniform(-180, 180, n_grids),
        'y': np.random.uniform(-60, 60, n_grids),
        'optimal_ratio': np.random.uniform(0.60, 0.90, n_grids),
        'comprehensive_score': np.random.uniform(0.5, 1.0, n_grids),
        'Y': np.random.uniform(5000, 10000, n_grids),
        'ET': np.random.uniform(300, 800, n_grids),
        'WUE': np.random.uniform(10, 20, n_grids),
        'CH4': np.random.uniform(100, 400, n_grids),
        'N2O': np.random.uniform(1, 5, n_grids)
    })

    # 测试区域分配
    print("\n测试区域分配:")
    optimal_results = assign_region_to_grids(optimal_results, RICE_REGIONS, 'rice_regions')
    print(f"  区域分布:\n{optimal_results['region'].value_counts()}")

    # 测试区域统计
    print("\n测试区域统计:")
    regional_stats = calculate_regional_statistics(
        optimal_results, 'rice', '20251028', 'rice_regions'
    )
    print(f"  统计了 {len(regional_stats)} 个区域")

    # 测试国家分析
    print("\n测试国家分析:")
    country_stats = analyze_by_country_simple(optimal_results, 'rice', '20251028')
    print(f"  统计了 {len(country_stats)} 个国家")

    print("\n[测试通过] 区域分析功能正常")


if __name__ == "__main__":
    test_regional_analysis()
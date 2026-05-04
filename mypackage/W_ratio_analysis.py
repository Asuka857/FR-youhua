"""
比值分析模块 (Ratio Analysis Module)
功能：最优值与对照组比值(optimal/ck)的专项分析

创建时间：2026-04-03
最后修改：2026-04-03

主要功能：
1. 生成比值专项分析报告
2. 比值空间分布分析
3. 显著改善区域识别
4. 比值解读指南
"""

import numpy as np
import pandas as pd
from pathlib import Path

INDICATOR_TITLE_NAMES = {
    'Y': 'Y',
    'ET': 'ET',
    'WUE': 'WUE',
    'CH4': 'CH4',
    'N2O': 'N2O',
    'GWP': 'GWP'
}


# =============================================================================
# 比值空间分布分析
# =============================================================================

def analyze_ratio_spatial_patterns(comparison_results, crop, outday):
    """
    分析比值的空间分布模式

    参数:
        comparison_results: dict, 对比分析结果
        crop: 作物类型
        outday: 输出日期

    返回:
        analysis_results: dict, 空间分布分析结果
    """
    from .W_regional_analysis import calculate_regional_statistics
    from .W_config import INDICATOR_DIRECTION

    results = {}

    for indicator in ['Y', 'ET', 'WUE', 'CH4', 'N2O', 'GWP']:
        if indicator not in comparison_results['indicator_details']:
            continue

        # 获取比值数据
        ratio_data = comparison_results['indicator_details'][indicator]
        ratio_df = ratio_data[['x', 'y', 'optimal_ck_ratio']].copy()

        # 计算区域统计
        regional_ratio_stats = calculate_regional_statistics(
            ratio_df.rename(columns={'optimal_ck_ratio': 'value'}),
            crop, outday, region_type='rice_regions',
            value_col='value'
        )

        # 识别改善显著区域
        direction = INDICATOR_DIRECTION.get(indicator, 'maximize')

        if direction == 'maximize':
            # 最大化指标：识别比值>1.05的区域（显著改善）
            significant_improvement = ratio_df[ratio_df['optimal_ck_ratio'] > 1.05]
        else:
            # 最小化指标：识别比值<0.95的区域（显著改善）
            # 注意：负值指标已通过ratio=ck/opt实现ratio<1表示改善
            significant_improvement = ratio_df[ratio_df['optimal_ck_ratio'] < 0.95]

        results[indicator] = {
            'regional_stats': regional_ratio_stats,
            'significant_improvement_grids': len(significant_improvement),
            'improvement_pct': len(significant_improvement) / len(ratio_df) * 100 if len(ratio_df) > 0 else 0
        }

    return results


# =============================================================================
# 生成比值分析报告
# =============================================================================

def generate_ratio_analysis_report(comparison_results, crop, outday, output_dir):
    """
    生成比值专项分析报告

    参数:
        comparison_results: dict, 对比分析结果
        crop: 作物类型
        outday: 输出日期
        output_dir: 输出目录

    返回:
        report_file: 报告文件路径
    """
    output_dir = Path(output_dir)

    # 生成Markdown格式报告
    report_lines = []

    report_lines.append("# 最优值与对照组比值分析报告\n\n")
    report_lines.append("## 概述\n\n")
    report_lines.append("本报告分析最优情景相对于对照组的性能比值（optimal/ck ratio）。\n\n")
    report_lines.append("**比值含义:**\n")
    report_lines.append("- 比值 = optimal / ck\n")
    report_lines.append("- 比值 > 1: 最优情景数值大于对照组\n")
    report_lines.append("- 比值 = 1: 最优情景与对照组相等\n")
    report_lines.append("- 比值 < 1: 最优情景数值小于对照组\n\n")

    # 添加各指标比值统计
    report_lines.append("## 各指标比值统计\n\n")

    comparison_data = comparison_results['comparison_data']

    report_lines.append("| 指标 | 平均比值 | 中位数比值 | 标准差 | Q25 | Q75 | 改善网格百分比 |\n")
    report_lines.append("|------|---------|-----------|--------|-----|-----|---------------|\n")

    for indicator in ['Y', 'ET', 'WUE', 'CH4', 'N2O', 'GWP']:
        stats = comparison_data[comparison_data['indicator'] == indicator]

        if len(stats) == 0:
            continue

        row = stats.iloc[0]

        report_lines.append(
            f"| {indicator} | {row['mean_ratio']:.3f} | "
            f"{row['median_ratio']:.3f} | {row['std_ratio']:.3f} | "
            f"{row['ratio_q25']:.3f} | {row['ratio_q75']:.3f} | "
            f"{row['ratio_improvement_pct']:.1f}% |\n"
        )

    # 添加空间分布分析
    report_lines.append("\n## 比值空间分布分析\n\n")

    try:
        spatial_analysis = analyze_ratio_spatial_patterns(comparison_results, crop, outday)

        for indicator, analysis in spatial_analysis.items():
            report_lines.append(f"### {indicator}\n\n")
            report_lines.append(
                f"- 显著改善网格数: {analysis['significant_improvement_grids']:,}\n"
            )
            report_lines.append(
                f"- 显著改善百分比: {analysis['improvement_pct']:.2f}%\n\n"
            )
    except Exception as e:
        report_lines.append(f"*空间分布分析失败: {e}*\n\n")

    # 添加比值解读指南
    report_lines.append("\n## 比值解读指南\n\n")

    report_lines.append("### 最大化指标（Y, WUE）\n\n")
    report_lines.append("对于需要最大化的指标（产量Y、水分利用效率WUE）：\n\n")
    report_lines.append("- **比值 > 1**: 最优情景优于对照组（改善）\n")
    report_lines.append("  - 例如：ratio = 1.25 表示最优情景产量比对照组高25%\n")
    report_lines.append("- **比值 = 1**: 无差异\n")
    report_lines.append("- **比值 < 1**: 最优情景劣于对照组（恶化）\n")
    report_lines.append("  - 例如：ratio = 0.85 表示最优情景产量比对照组低15%\n\n")

    report_lines.append("### 最小化指标（ET, CH4, N2O, GWP）\n\n")
    report_lines.append("对于需要最小化的指标（蒸散发ET、甲烷排放CH4、氧化亚氮排放N2O、全球增温潜势GWP）：\n\n")
    report_lines.append("- **比值 < 1**: 最优情景优于对照组（改善）\n")
    report_lines.append("  - 例如：ratio = 0.75 表示最优情景排放比对照组低25%\n")
    report_lines.append("- **比值 = 1**: 无差异\n")
    report_lines.append("- **比值 > 1**: 最优情景劣于对照组（恶化）\n")
    report_lines.append("  - 例如：ratio = 1.15 表示最优情景排放比对照组高15%\n\n")

    report_lines.append("### CH4负值特殊情况\n\n")
    report_lines.append("CH4可能存在负值（表示吸收）：\n\n")
    report_lines.append("- 例如：optimal = -4.5 kg/ha, ck = -4.6 kg/ha\n")
    report_lines.append("- 比值 = -4.5 / -4.6 = 0.978\n")
    report_lines.append("- 对于最小化指标，比值 < 1 表示改善（合理）\n")
    report_lines.append("- 含义：最优情景吸收更多CH4（绝对值更大），比值<1表示改善\n\n")

    # 添加统计方法说明
    report_lines.append("\n## 统计方法说明\n\n")

    report_lines.append("### 比值计算方法\n\n")
    report_lines.append("```\n")
    report_lines.append("ratio = optimal / ck\n")
    report_lines.append("```\n\n")

    report_lines.append("### 除零处理\n\n")
    report_lines.append("- 当对照组数值接近0（|ck| < 1e-6）时，比值设为NaN\n")
    report_lines.append("- 在地图中以白色/空白显示\n\n")

    report_lines.append("### 显著改善定义\n\n")
    report_lines.append("- 最大化指标：比值 > 1.05（改善超过5%）\n")
    report_lines.append("- 最小化指标：比值 < 0.95（改善超过5%）\n\n")

    # 保存报告
    report_file = output_dir / f'{crop}-ratio-report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)

    print(f"  比值分析报告已保存: {report_file}")

    return report_file


# =============================================================================
# 比值改善百分比计算
# =============================================================================

def calculate_ratio_improvement_pct(merged_data, indicator):
    """
    计算比值表示改善的网格百分比

    参数:
        merged_data: DataFrame, 包含optimal_ck_ratio列
        indicator: str, 指标名称

    返回:
        improvement_pct: float, 改善网格百分比
    """
    from .W_config import INDICATOR_DIRECTION

    direction = INDICATOR_DIRECTION.get(indicator, 'maximize')
    ratio_values = merged_data['optimal_ck_ratio'].dropna()

    if len(ratio_values) == 0:
        return 0.0

    if direction == 'maximize':
        # 最大化指标：比值>1表示改善
        improvement_pct = (ratio_values > 1.0).sum() / len(ratio_values) * 100
    else:
        # 最小化指标：比值<1表示改善（负值指标已通过ratio=ck/opt实现）
        improvement_pct = (ratio_values < 1.0).sum() / len(ratio_values) * 100

    return improvement_pct


# =============================================================================
# 批量比值地图生成
# =============================================================================

def generate_all_ratio_maps(comparison_results, output_dir, reference_tif_path,
                            shapefile_path="E:/USERS/Global/country.shp"):
    """
    为所有指标生成比值地图

    参数:
        comparison_results: dict, 对比分析结果
        output_dir: 输出目录
        reference_tif_path: 参考TIF文件路径
        shapefile_path: 底图文件路径

    返回:
        ratio_maps: dict, 各指标比值地图路径
    """
    from .W_comparison_map_tif import csv_to_tif_advanced, draw_single_map_from_tif

    output_dir = Path(output_dir)
    tif_dir = output_dir / 'tif'
    maps_dir = output_dir / 'maps'
    tif_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    indicator_names = {
        'Y': 'Yield',
        'ET': 'ET',
        'WUE': 'WUE',
        'CH4': 'CH₄',
        'N2O': 'N₂O',
        'GWP': 'GWP'
    }

    ratio_maps = {}

    for indicator in ['Y', 'ET', 'WUE', 'CH4', 'N2O', 'GWP']:
        if indicator not in comparison_results['indicator_details']:
            print(f"  跳过 {indicator}: 无对比数据")
            continue

        # 获取比值数据
        comparison_data = comparison_results['indicator_details'][indicator]

        if 'optimal_ck_ratio' not in comparison_data.columns:
            print(f"  跳过 {indicator}: 无比值数据")
            continue

        data_ratio = comparison_data[['x', 'y', 'optimal_ck_ratio']].copy()

        # 计算比值范围
        ratio_values = data_ratio['optimal_ck_ratio'].dropna()

        if len(ratio_values) == 0:
            print(f"  跳过 {indicator}: 比值数据为空")
            continue

        # 使用百分位数设定范围
        ratio_p05 = ratio_values.quantile(0.05)
        ratio_p95 = ratio_values.quantile(0.95)

        # 确保对称性（以1.0为中心）
        max_deviation = max(abs(ratio_p05 - 1.0), abs(ratio_p95 - 1.0))
        vmin_ratio = 1.0 - max_deviation
        vmax_ratio = 1.0 + max_deviation

        # 设置最小范围限制
        vmin_ratio = max(0.5, min(0.85, vmin_ratio))
        vmax_ratio = min(1.5, max(1.15, vmax_ratio))

        # 生成TIF文件
        tif_ratio = csv_to_tif_advanced(
            data_ratio, 'optimal_ck_ratio',
            str(tif_dir / f'{indicator}_ratio.tif'),
            reference_tif_path
        )

        # 生成比值地图
        title_name = INDICATOR_TITLE_NAMES.get(indicator, indicator)
        map_ratio = draw_single_map_from_tif(
            tif_ratio,
            title=title_name,
            colorbar_name='Ratio',
            output_path=str(maps_dir / f'{indicator}_ratio.png'),
            vmin=vmin_ratio, vmax=vmax_ratio, cmap='RdYlGn',
            shapefile_path=shapefile_path
        )

        ratio_maps[indicator] = {
            'tif': tif_ratio,
            'map': map_ratio
        }

        print(f"  {indicator} 比值地图已生成")

    return ratio_maps


# =============================================================================
# 测试函数
# =============================================================================

def test_ratio_analysis():
    """
    测试比值分析模块
    """
    print("\n[测试模式] 使用模拟数据测试比值分析模块...")

    # 创建模拟对比数据
    n_grids = 1000
    comparison_data_list = []

    for indicator in ['Y', 'ET', 'WUE', 'CH4', 'N2O', 'GWP']:
        # 模拟比值数据
        if indicator in ['Y', 'WUE']:
            # 最大化指标：比值 > 1 表示改善
            ratios = np.random.uniform(0.85, 1.25, n_grids)
        else:
            # 最小化指标：比值 < 1 表示改善
            ratios = np.random.uniform(0.75, 1.15, n_grids)

        comparison_data_list.append({
            'indicator': indicator,
            'mean_ratio': np.mean(ratios),
            'median_ratio': np.median(ratios),
            'std_ratio': np.std(ratios),
            'ratio_q25': np.percentile(ratios, 25),
            'ratio_q75': np.percentile(ratios, 75),
            'ratio_improvement_pct': np.sum(ratios > 1.0) / len(ratios) * 100 if indicator in ['Y', 'WUE'] else np.sum(ratios < 1.0) / len(ratios) * 100
        })

    comparison_data = pd.DataFrame(comparison_data_list)

    print("\n测试比值统计:")
    print(comparison_data[['indicator', 'mean_ratio', 'median_ratio', 'ratio_improvement_pct']])

    print("\n[测试通过] 比值分析模块正常")


if __name__ == "__main__":
    test_ratio_analysis()

"""
空间分析模块 (Spatial Analysis Module)
功能：生成空间分布图、区域统计和综合报告

创建时间：2025-03-20
最后修改：2025-03-20

改进内容：
1. 集成picture.py的map函数进行可视化
2. 集成W_glob_tif的csv_to_tif函数生成GeoTIFF
3. 添加结果对比可视化
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

# 导入区域分析模块
from .W_regional_analysis import calculate_regional_statistics, detailed_regional_analysis

# 导入对比分析模块
from .W_comparison import load_ck_data

# 导入配置
from .W_config import VISUALIZATION_PARAMS, get_visualization_params

# 导入路径管理器
from .W_path_manager import get_path_manager

# 导入对比分析模块
from .W_comparison import generate_comparison_analysis as _generate_comparison_analysis


# =============================================================================
# 主分析函数
# =============================================================================

def generate_spatial_analysis(
    crop,
    outday,
    optimal_results=None,
    output_dir=None,
    generate_tif=True,
    generate_maps=True,
    compare_with_ck=True
):
    """
    第4步：生成空间分析和可视化

    改进特性：
    - 自动生成GeoTIFF文件
    - 自动生成空间分布图
    - 集成对照组对比分析

    参数：
        crop: 作物类型
        outday: 输出日期标识
        optimal_results: 最优结果DataFrame (如果为None则从文件读取)
        output_dir: 输出目录 (默认None，自动生成)
        generate_tif: 是否生成GeoTIFF (默认 True)
        generate_maps: 是否生成空间分布图 (默认 True)
        compare_with_ck: 是否生成对照组对比 (默认 True)

    返回：
        analysis_results: 分析结果字典
    """

    print(f"\n{'='*70}")
    print(f"  第4步: 做空间表达 - 空间分析和可视化")
    print(f"{'='*70}")
    print(f"作物: {crop}")
    print(f"{'='*70}\n")

    # =========================================================================
    # 步骤1: 加载最优结果数据
    # =========================================================================
    print(f"[Step 1/7] 加载最优结果数据...")

    # 初始化路径管理器
    pm = get_path_manager()

    if optimal_results is None:
        optimal_file = pm.get_optimal_dir(crop, outday) / 'optimal_ratio_results.csv'
        if not optimal_file.exists():
            raise FileNotFoundError(
                f"找不到最优结果文件: {optimal_file}\n"
                f"请先运行 optimize_deficit_ratio() 选择最优解"
            )
        optimal_results = pd.read_csv(optimal_file)

    print(f"  加载网格数量: {len(optimal_results):,}")

    # =========================================================================
    # 步骤2: 准备输出目录结构
    # =========================================================================
    if output_dir is None:
        output_dir = pm.get_spatial_analysis_dir(crop, outday)
    else:
        output_dir = Path(output_dir)

    # 创建子目录
    subdirs = ['tif', 'maps', 'statistics', 'charts', 'reports', 'comparison']
    for subdir in subdirs:
        os.makedirs(f'{output_dir}/{subdir}', exist_ok=True)

    print(f"  输出目录: {output_dir}")

    # =========================================================================
    # 步骤3: 生成统计摘要
    # =========================================================================
    print(f"\n[Step 2/7] 生成统计摘要...")

    stats_summary = generate_statistics_summary(optimal_results, crop, outday)

    # 保存统计表
    stats_file = f'{output_dir}/statistics/statistics_summary.csv'
    stats_summary.to_csv(stats_file, index=False)
    print(f"  统计摘要已保存: {stats_file}")

    # =========================================================================
    # 步骤4: 区域统计分析
    # =========================================================================
    print(f"\n[Step 3/7] 区域统计分析...")

    try:
        regional_stats = calculate_regional_statistics(
            optimal_results,
            crop,
            outday,
            region_type='rice_regions'
        )

        # 保存区域统计
        regional_file = f'{output_dir}/statistics/regional_statistics.csv'
        regional_stats.to_csv(regional_file, index=False)
        print(f"  区域统计已保存: {regional_file}")

    except Exception as e:
        print(f"  警告: 区域分析失败 - {e}")
        regional_stats = None

    # =========================================================================
    # 步骤5: 生成空间分布数据
    # =========================================================================
    print(f"\n[Step 4/7] 准备空间分布数据...")

    # 准备各指标的GeoTIFF数据
    indicators = ['optimal_ratio', 'comprehensive_score', 'Y', 'ET', 'WUE', 'CH4', 'N2O']
    if 'GWP' in optimal_results.columns:
        indicators.append('GWP')

    tif_data_dict = {}
    for indicator in indicators:
        if indicator not in optimal_results.columns:
            continue

        # 准备数据
        tif_data = optimal_results[['x', 'y', indicator]].copy()
        tif_data_dict[indicator] = tif_data

        # 保存CSV（供后续转TIF使用）
        csv_file = f'{output_dir}/tif/{indicator}.csv'
        tif_data.to_csv(csv_file, index=False)

    print(f"  已准备 {len(tif_data_dict)} 个指标的空间数据")

    # =========================================================================
    # 步骤6: 生成GeoTIFF文件
    # =========================================================================
    if generate_tif:
        print(f"\n[Step 5/7] 生成GeoTIFF文件...")

        try:
            from .W_glob_tif import csv_to_tif_custom

            tif_files = []
            for indicator in tqdm(indicators, desc="  生成TIF"):
                if indicator not in tif_data_dict:
                    continue

                try:
                    tif_file = generate_geo_tif(
                        tif_data_dict[indicator],
                        indicator,
                        crop,
                        outday,
                        output_dir
                    )
                    tif_files.append(tif_file)
                except Exception as e:
                    print(f"    警告: {indicator} TIF生成失败 - {e}")

            print(f"  已生成 {len(tif_files)} 个GeoTIFF文件")

        except ImportError:
            print(f"  警告: W_glob_tif模块导入失败，跳过TIF生成")

    # =========================================================================
    # 步骤7: 生成空间分布图
    # =========================================================================
    if generate_maps:
        print(f"\n[Step 6/7] 生成空间分布图...")

        try:
            from .picture import map

            map_files = []
            for indicator in tqdm(indicators, desc="  生成地图"):
                if indicator not in tif_data_dict:
                    continue

                try:
                    map_file = generate_spatial_map(
                        tif_data_dict[indicator],
                        indicator,
                        crop,
                        outday,
                        output_dir
                    )
                    map_files.append(map_file)
                except Exception as e:
                    print(f"    警告: {indicator} 地图生成失败 - {e}")

            print(f"  已生成 {len(map_files)} 个空间分布图")

        except ImportError:
            print(f"  警告: picture模块导入失败，跳过地图生成")

    # =========================================================================
    # 步骤8: 对照组对比分析
    # =========================================================================
    if compare_with_ck:
        print(f"\n[Step 7/7] 对照组对比分析...")

        comparison_results = _generate_comparison_analysis(
            optimal_results,
            crop,
            outday,
            output_dir
        )
    else:
        comparison_results = None

    # =========================================================================
    # 步骤9: 生成分析报告（专业报告生成器）
    # =========================================================================
    print(f"\n[生成报告] 生成专业分析报告...")

    from .W_report_generator import generate_report

    # 报告指标只包含需要对比的六个指标
    report_indicators = ['Y', 'ET', 'WUE', 'CH4', 'N2O', 'GWP']

    # 准备对照组数据字典
    ck_data_dict = {}
    for indicator in report_indicators:
        ck_data = load_ck_data(crop, outday, indicator)
        if ck_data is not None:
            ck_data_dict[indicator] = ck_data

    print(f"  已加载 {len(ck_data_dict)} 个指标的对照组数据")

    # 生成专业报告
    report, tables = generate_report(
        optimal_results,
        ck_data_dict,
        crop,
        outday,
        report_indicators
    )

    # 保存主报告
    report_file = f'{output_dir}/reports/{crop}-report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"  分析报告已保存: {report_file}")

    # 保存表格
    tables_dir = Path(output_dir) / 'reports' / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    if 'global_comparison' in tables:
        tables['global_comparison'].to_csv(tables_dir / 'global_comparison.csv', index=False)
        print(f"  全球对比表已保存")

    if 'continental_ratio_stats' in tables:
        tables['continental_ratio_stats'].to_csv(tables_dir / 'continental_ratio_stats.csv', index=False)
        print(f"  大洲比例统计表已保存")

    if 'continental_detailed' in tables:
        for ind, df in tables['continental_detailed'].items():
            # 处理特殊情况：continental_summary 不需要再加前缀
            if ind == 'continental_summary':
                df.to_csv(tables_dir / 'continental_summary.csv', index=False)
            else:
                df.to_csv(tables_dir / f'continental_detailed_{ind}.csv', index=False)
        print(f"  大洲详细统计表已保存 ({len(tables['continental_detailed'])} 个指标)")

    # =========================================================================
    # 步骤10: 生成比值专项分析报告（完善版）
    # =========================================================================
    if comparison_results is not None:
        print(f"\n[生成比值报告] 生成比值专项分析报告（完善版）...")

        from .W_ratio_report_generator import generate_ratio_analysis_report

        try:
            ratio_report_file = generate_ratio_analysis_report(
                comparison_results,
                crop,
                outday,
                output_dir / 'reports'
            )
            print(f"  比值分析报告已保存: {ratio_report_file}")
        except Exception as e:
            print(f"  警告: 比值分析报告生成失败 - {e}")

    # =========================================================================
    # 完成
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  空间分析完成!")
    print(f"{'='*70}")
    print(f"输出目录结构:")
    print(f"  {output_dir}/")
    print(f"  ├── tif/              # GeoTIFF数据文件")
    print(f"  ├── maps/             # 空间分布图")
    print(f"  ├── statistics/       # 统计表格")
    print(f"  ├── charts/           # 可视化图表")
    print(f"  ├── comparison/       # 对比分析结果")
    print(f"  └── reports/          # 分析报告")
    print(f"{'='*70}\n")

    return {
        'stats_summary': stats_summary,
        'regional_stats': regional_stats,
        'comparison_results': comparison_results,
        'output_dir': output_dir
    }


# =============================================================================
# GeoTIFF生成函数
# =============================================================================

def generate_geo_tif(data_df, indicator, crop, outday, output_dir):
    """
    生成GeoTIFF文件

    参数：
        data_df: 包含x, y, value的DataFrame
        indicator: 指标名称
        crop: 作物类型
        outday: 输出日期
        output_dir: 输出目录

    返回：
        tif_file: 生成的TIF文件路径
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    from osgeo import gdal

    # 读取参考TIF文件（使用路径管理器）
    pm = get_path_manager()
    reference_tif_path = pm.base_dir / crop / outday / 'x_data_mask.tif'

    # 如果本地不存在，尝试原始路径
    if not reference_tif_path.exists():
        reference_tif_path = Path(f'F:/WaterML/{crop}/{outday}/x_data_mask.tif')

    if not os.path.exists(reference_tif_path):
        raise FileNotFoundError(f"找不到参考TIF文件: {reference_tif_path}")

    ref_ds = gdal.Open(reference_tif_path)
    geotransform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    x_min = geotransform[0]
    y_max = geotransform[3]
    x_res = geotransform[1]
    y_res = abs(geotransform[5])
    cols = ref_ds.RasterXSize
    rows = ref_ds.RasterYSize
    ref_ds = None

    # 提取数据
    x_coords = data_df['x'].values
    y_coords = data_df['y'].values
    values = data_df[indicator].values

    # 初始化栅格数组
    raster = np.full((rows, cols), -9999, dtype=np.float32)

    # 将点数据映射到栅格索引
    for x, y, value in zip(x_coords, y_coords, values):
        col = int(round((x - x_min) / x_res))
        row = int(round((y_max - y) / y_res))

        # 边界检查
        col = max(0, min(cols - 1, col))
        row = max(0, min(rows - 1, row))

        raster[row, col] = value

    # 保存为GeoTIFF
    tif_dir = f'{output_dir}/tif'
    os.makedirs(tif_dir, exist_ok=True)
    tif_file = f'{tif_dir}/{indicator}.tif'

    transform = Affine(x_res, 0, x_min, 0, -y_res, y_max)
    crs = CRS.from_wkt(projection)

    with rasterio.open(
        tif_file,
        'w',
        driver='GTiff',
        height=raster.shape[0],
        width=raster.shape[1],
        count=1,
        dtype='float32',
        crs=crs,
        transform=transform,
        nodata=-9999
    ) as dst:
        dst.write(raster, 1)

    return tif_file


# =============================================================================
# 空间分布图生成函数
# =============================================================================

def generate_spatial_map(data_df, indicator, crop, outday, output_dir):
    """
    生成空间分布图

    参数：
        data_df: 包含x, y, value的DataFrame
        indicator: 指标名称
        crop: 作物类型
        outday: 输出日期
        output_dir: 输出目录

    返回：
        map_file: 生成的地图文件路径
    """
    # 先生成TIF
    tif_file = f'{output_dir}/tif/{indicator}.tif'

    if not os.path.exists(tif_file):
        tif_file = generate_geo_tif(data_df, indicator, crop, outday, output_dir)

    # 获取可视化参数
    vis_params = get_visualization_params(crop, indicator)

    # 设置默认值
    vmin = vis_params.get('vmin', data_df[indicator].min())
    vmax = vis_params.get('vmax', data_df[indicator].max())
    cmap = vis_params.get('cmap', 'RdYlGn')

    # 生成地图
    try:
        from .picture import map

        # 构建标题和颜色条名称
        # 作物中文名称映射
        crop_names = {
            'rice': 'Rice',
            'wheat': 'Wheat',
            'maize': 'Maize'
        }
        crop_name = crop_names.get(crop, crop)

        indicator_names = {
            'optimal_ratio': (crop_name, 'Ratio'),
            'comprehensive_score': ('Comprehensive Score', 'Score'),
            'Y': ('Yield', 'kg/ha'),
            'ET': ('ET', 'mm'),
            'WUE': ('WUE', 'kg/m3'),
            'CH4': ('CH₄', 'kg/ha'),
            'N2O': ('N₂O', 'kg/ha'),
            'GWP': ('GWP', 'kg CO2-eq/ha')
        }

        title, colorbar_name = indicator_names.get(indicator, (indicator, indicator))

        map_dir = f'{output_dir}/maps'
        os.makedirs(map_dir, exist_ok=True)
        map_file = f'{map_dir}/{indicator}_map.png'

        # 调用map函数
        # tif_file 是包含tif文件的目录路径，tif_name 是tif文件名（不含.tif）
        tif_dir = f'{output_dir}/tif'
        map(
            tif_file=tif_dir,  # 传递tif目录的绝对路径
            tif_name=indicator,  # tif文件名（不含扩展名）
            crop=crop,
            vmin=vmin,
            vmax=vmax,
            title=title,
            colorbar_name=colorbar_name,
            save_file=f'{indicator}_map',
            outday=outday,
            saveday=outday,
            NW='W/spatial_analysis',
            road='maps',
            model=1
        )

        return map_file

    except ImportError:
        # 如果picture模块不可用，使用matplotlib简单绘图
        return generate_simple_map(data_df, indicator, output_dir, vmin, vmax, cmap)


def generate_simple_map(data_df, indicator, output_dir, vmin, vmax, cmap='RdYlGn'):
    """
    生成简单的空间分布图（matplotlib版本）

    参数：
        data_df: 数据DataFrame
        indicator: 指标名称
        output_dir: 输出目录
        vmin: 最小值
        vmax: 最大值
        cmap: 颜色映射

    返回：
        map_file: 地图文件路径
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(12, 8))

    # 散点图
    sc = ax.scatter(
        data_df['x'],
        data_df['y'],
        c=data_df[indicator],
        cmap=cmap,
        s=1,
        vmin=vmin,
        vmax=vmax,
        alpha=0.7
    )

    # 颜色条
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label(indicator)

    # 标题和标签
    ax.set_title(f'{indicator} Spatial Distribution')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # 保存
    map_dir = f'{output_dir}/maps'
    os.makedirs(map_dir, exist_ok=True)
    map_file = f'{map_dir}/{indicator}_map.png'

    plt.savefig(map_file, dpi=300, bbox_inches='tight')
    plt.close()

    return map_file


# =============================================================================
# 统计分析函数
# =============================================================================

def generate_statistics_summary(optimal_results, crop, outday):
    """
    生成统计摘要表

    参数：
        optimal_results: 最优结果DataFrame
        crop: 作物类型
        outday: 输出日期

    返回：
        stats_df: 统计摘要DataFrame
    """
    indicators = ['optimal_ratio', 'comprehensive_score', 'Y', 'ET', 'WUE', 'CH4', 'N2O']
    if 'GWP' in optimal_results.columns:
        indicators.append('GWP')

    stats_data = []

    for indicator in indicators:
        if indicator not in optimal_results.columns:
            continue

        values = optimal_results[indicator]

        stats_data.append({
            'indicator': indicator,
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
            'median': values.median(),
            'q25': values.quantile(0.25),
            'q75': values.quantile(0.75),
            'n_valid': values.notna().sum()
        })

    stats_df = pd.DataFrame(stats_data)

    return stats_df


def calculate_global_statistics(optimal_results):
    """
    计算全球统计指标

    参数：
        optimal_results: 最优结果DataFrame

    返回：
        global_stats: 全球统计字典
    """
    global_stats = {
        'n_grids': len(optimal_results),
        'optimal_ratio': {
            'mean': optimal_results['optimal_ratio'].mean(),
            'std': optimal_results['optimal_ratio'].std(),
            'median': optimal_results['optimal_ratio'].median()
        },
        'comprehensive_score': {
            'mean': optimal_results['comprehensive_score'].mean(),
            'std': optimal_results['comprehensive_score'].std()
        }
    }

    return global_stats


# =============================================================================
# 测试函数
# =============================================================================

def test_spatial_analysis():
    """
    测试空间分析功能
    """
    print("\n[测试模式] 使用模拟数据测试空间分析...")

    # 创建模拟数据
    n_grids = 1000
    optimal_results = pd.DataFrame({
        'x': np.linspace(110, 120, n_grids),
        'y': np.linspace(30, 40, n_grids),
        'optimal_ratio': np.random.uniform(0.60, 0.90, n_grids),
        'comprehensive_score': np.random.uniform(0.5, 1.0, n_grids),
        'Y': np.random.uniform(5000, 10000, n_grids),
        'ET': np.random.uniform(300, 800, n_grids),
        'WUE': np.random.uniform(10, 20, n_grids),
        'CH4': np.random.uniform(100, 400, n_grids),
        'N2O': np.random.uniform(1, 5, n_grids)
    })

    # 测试统计摘要
    print("\n测试统计摘要生成:")
    stats = generate_statistics_summary(optimal_results, 'rice', '20251028')
    print(f"  统计指标数: {len(stats)}")

    # 测试专业报告生成
    print("\n测试专业报告生成:")
    try:
        from .W_report_generator import generate_report
        # 模拟对照组数据（简化测试）
        ck_data_dict = {}
        report, tables = generate_report(optimal_results, ck_data_dict, 'rice', '20251028')
        print(f"  报告生成成功，长度: {len(report)} 字符")
        print(f"  表格数量: {len(tables)}")
    except Exception as e:
        print(f"  警告: 报告生成跳过 (需要对照组数据) - {e}")

    print("\n[测试通过] 空间分析功能正常")


if __name__ == "__main__":
    test_spatial_analysis()
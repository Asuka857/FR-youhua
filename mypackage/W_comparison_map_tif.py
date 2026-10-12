"""
对比地图绘制模块 - 高精度版本
功能：先转TIF再绘图，提高精确度，生成单张图和拼图

创建时间：2025-03-24
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from osgeo import gdal
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter, LinearLocator
import geopandas as gpd
from cartopy import crs as ccrs

INDICATOR_TITLE_NAMES = {
    'Y': 'Yield',
    'ET': 'ET',
    'WUE': 'WUE',
    'CH4': 'CH₄',
    'N2O': 'N₂O',
    'GWP': 'GWP'
}

INDICATOR_COLORBAR_NAMES = {
    'Y': 'kg/ha',
    'ET': 'mm',
    'WUE': 'kg/m3',
    'CH4': 'kg/ha',
    'N2O': 'kg/ha',
    'GWP': r'kg CO$_2$-eq/ha'
}


# =============================================================================
# CSV转TIF函数
# =============================================================================

def csv_to_tif_advanced(data_df, value_col, output_tif_path, reference_tif_path):
    """
    将CSV点数据转换为GeoTIFF栅格文件

    参数:
        data_df: DataFrame, 包含x, y列和值列
        value_col: str, 值列名称
        output_tif_path: str, 输出TIF文件路径
        reference_tif_path: str, 参考TIF文件路径

    返回:
        output_tif_path: 生成的TIF文件路径
    """
    # 读取参考TIF文件
    ref_ds = gdal.Open(str(reference_tif_path))
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
    values = data_df[value_col].values

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
    os.makedirs(os.path.dirname(output_tif_path), exist_ok=True)

    transform = Affine(x_res, 0, x_min, 0, -y_res, y_max)
    crs = CRS.from_wkt(projection)

    with rasterio.open(
        output_tif_path,
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

    return output_tif_path


# =============================================================================
# 单张地图绘制函数（仿照 picture.map 风格，颜色条在内部，无Global Mean）
# =============================================================================

def format_global_mean_text(global_avg_val, as_percentage=False):
    if as_percentage:
        return f'Global Mean: {(global_avg_val - 1.0) * 100:+.1f}%'
    return f'Global Mean: {global_avg_val:.2f}'


def draw_single_map_from_tif(tif_path, title, colorbar_name, output_path,
                              vmin=None, vmax=None, cmap='RdYlGn',
                              shapefile_path="E:/USERS/Global/country.shp",
                              show_global_mean=True,
                              global_mean_as_percentage=False):
    """
    从TIF文件绑制单张地图
    完全仿照 picture.map 函数风格，颜色条放在地图内部，显示Global Mean

    参数:
        tif_path: TIF文件路径
        title: 图标题
        colorbar_name: 颜色条名称
        output_path: 输出图片路径
        vmin, vmax: 数据范围
        cmap: 颜色映射
        shapefile_path: 底图文件路径
        show_global_mean: 是否显示全球平均值（默认True）
        global_mean_as_percentage: 是否将Global Mean显示为相对1.0的百分比变化
    """
    # 1. 读取栅格数据
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        bounds = src.bounds
        nodata = src.nodata

    # 手动设置 NoData 值（如果未正确读取）
    if nodata is None:
        nodata = -9999

    # 替换 NoData 值为 np.nan
    data = np.where(data == nodata, np.nan, data)

    # 如果没有传入vmin和vmax，自动设置
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    # 2. 读取底图文件
    gdf = gpd.read_file(shapefile_path)

    # 3. 设置地图投影和范围（与 map 函数完全一致）
    fig, ax = plt.subplots(
        figsize=(9, 7),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    # 确保地图范围与栅格数据匹配
    ax.set_extent([bounds.left, bounds.right, bounds.bottom, bounds.top], crs=ccrs.PlateCarree())

    # 4. 绘制栅格数据
    im = ax.imshow(
        data,
        cmap=cmap,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin='upper',
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest'
    )

    # 5. 绘制底图（矢量边界）
    gdf.plot(
        ax=ax,
        edgecolor='black',
        facecolor='none',
        linewidth=0.6
    )

    # 颜色条在地图内部（与 map 函数风格一致，但位置改为内部）
    cax = ax.inset_axes([0.05, 0.02, 0.9, 0.035])

    # 添加颜色条
    cbar = plt.colorbar(
        im,
        cax=cax,
        orientation='horizontal'
    )

    cbar.locator = LinearLocator(numticks=5)
    cbar.formatter = FormatStrFormatter('%.2f')
    cbar.update_ticks()
    # 设置颜色条边框的线宽
    cbar.outline.set_linewidth(0.9)

    # 设置全局字体为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    # 设置颜色条刻度属性，刻度朝上（与 map 函数完全一致）
    cbar.ax.tick_params(
        direction='out',
        length=1.7,
        width=0.8,
        labelsize=16,
        labeltop=True,
        labelbottom=False,
        top=True,
        bottom=False
    )

    # 获取颜色条的所有刻度标签
    labels = cbar.ax.get_xticklabels()

    # 设置最后一个标签右对齐
    labels[-1].set_ha('right')

    # 设置第一个标签左对齐
    labels[0].set_ha('left')

    # 设置其余标签居中对齐
    for label in labels[1:-1]:
        label.set_ha('center')

    # 修改颜色条刻度标签字体
    for label in cbar.ax.get_xticklabels():
        label.set_fontname('Times New Roman')

    # 添加颜色条标题（在颜色条上方靠左，与 map 函数完全一致）
    cbar.ax.text(
        0, 3,
        f'{colorbar_name}',
        fontsize=19,
        transform=cbar.ax.transAxes,
        va='bottom',
        ha='left',
        fontname='Times New Roman'
    )

    # 调整整个图形的边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 添加标题
    ax.set_title(title, fontsize=26, fontname='Times New Roman')

    # 不显示 Global Mean（与 map 函数不同）

    # 显示全球平均值（参考 picture.map 函数风格）
    if show_global_mean:
        # 计算全球平均值（排除NaN）
        global_avg_val = np.nanmean(data)

        # 设置显示文本格式
        text_str = format_global_mean_text(global_avg_val, global_mean_as_percentage)

        # 在地图上添加文本（位置在左下角）
        ax.text(
            0.5, 0.25,  # 相对坐标，水平居中
            text_str,
            transform=ax.transAxes,  # 使用轴坐标系
            fontsize=22,
            fontname='Times New Roman',
            fontweight='bold',  # 加粗
            color='black',
            ha='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)  # 半透明白色背景
        )

    # 保存地图
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    return output_path


# =============================================================================
# 对比地图绘制函数（3张独立的单图）
# =============================================================================

def draw_comparison_maps_from_tif(
    comparison_data, indicator, output_dir,
    reference_tif_path,
    shapefile_path="E:/USERS/Global/country.shp",
    vmin=None, vmax=None, vmin_change=None, vmax_change=None
):
    """
    从对比数据生成TIF文件并绘制高精度对比地图

    参数:
        comparison_data: DataFrame, 包含x, y, optimal, ck, change列
        indicator: 指标名称
        output_dir: 输出目录
        reference_tif_path: 参考TIF文件路径
        shapefile_path: 底图文件路径
        vmin, vmax: optimal/ck数据范围
        vmin_change, vmax_change: change数据范围

    返回:
        dict: 包含生成的文件路径
    """
    # 指标名称映射
    indicator_names = {
        'Y': ('Yield', 'kg/ha'),
        'ET': ('ET', 'mm'),
        'WUE': ('WUE', 'kg/m³'),
        'CH4': ('CH₄', 'kg/ha'),
        'N2O': ('N₂O', 'kg/ha'),
        'GWP': ('GWP', r'kg CO$_2$-eq/ha')
    }
    title_name = INDICATOR_TITLE_NAMES.get(indicator, indicator)
    colorbar_name = INDICATOR_COLORBAR_NAMES.get(indicator, indicator)

    # 比值地图配色配置
    RATIO_COLORBAR_CONFIG = {
        'common': {
            'cmap': 'RdYlGn',  # 用户指定使用红黄绿渐变
            'center': 1.0,      # 中心点为1.0（无变化）
            'colorbar_name': 'Ratio'
        }
    }

    # 创建TIF输出目录
    tif_dir = Path(output_dir) / 'tif'
    maps_dir = Path(output_dir) / 'maps'
    tif_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    # 准备数据
    data_optimal = comparison_data[['x', 'y', 'optimal']].copy()
    data_ck = comparison_data[['x', 'y', 'ck']].copy()
    data_change = comparison_data[['x', 'y', 'change']].copy()

    # 准备比值数据
    if 'optimal_ck_ratio' in comparison_data.columns:
        data_ratio = comparison_data[['x', 'y', 'optimal_ck_ratio']].copy()
    else:
        # 如果CSV中没有比值列，先计算
        comparison_data['optimal_ck_ratio'] = comparison_data['optimal'] / comparison_data['ck']
        data_ratio = comparison_data[['x', 'y', 'optimal_ck_ratio']].copy()

    # 计算统一的数据范围
    if vmin is None:
        vmin = min(data_optimal['optimal'].min(), data_ck['ck'].min())
    if vmax is None:
        vmax = max(data_optimal['optimal'].max(), data_ck['ck'].max())

    if vmin_change is None or vmax_change is None:
        vmax_change_val = max(abs(data_change['change'].min()), abs(data_change['change'].max()))
        vmin_change = -vmax_change_val
        vmax_change = vmax_change_val

    # 计算比值可视化范围（对称分布在1.0周围）
    if 'optimal_ck_ratio' in comparison_data.columns:
        ratio_values = comparison_data['optimal_ck_ratio'].dropna()

        # 使用百分位数避免极值影响
        ratio_p05 = ratio_values.quantile(0.05)
        ratio_p95 = ratio_values.quantile(0.95)

        # 确保对称性（以1.0为中心）
        max_deviation = max(abs(ratio_p05 - 1.0), abs(ratio_p95 - 1.0))
        vmin_ratio = 1.0 - max_deviation
        vmax_ratio = 1.0 + max_deviation

        # 设置最小范围限制
        vmin_ratio = max(0.5, min(0.85, vmin_ratio))
        vmax_ratio = min(1.5, max(1.15, vmax_ratio))
    else:
        vmin_ratio = 0.85
        vmax_ratio = 1.15

    # Step 1: 生成TIF文件
    print(f"  [Step 1] 生成TIF文件...")

    tif_optimal = csv_to_tif_advanced(
        data_optimal, 'optimal',
        str(tif_dir / f'{indicator}_optimal.tif'),
        reference_tif_path
    )

    tif_ck = csv_to_tif_advanced(
        data_ck, 'ck',
        str(tif_dir / f'{indicator}_ck.tif'),
        reference_tif_path
    )

    tif_change = csv_to_tif_advanced(
        data_change, 'change',
        str(tif_dir / f'{indicator}_change.tif'),
        reference_tif_path
    )

    tif_ratio = csv_to_tif_advanced(
        data_ratio, 'optimal_ck_ratio',
        str(tif_dir / f'{indicator}_ratio.tif'),
        reference_tif_path
    )

    # Step 2: 绘制单张地图
    print(f"  [Step 2] 绑制单张地图...")

    # 最优情景图
    map_optimal = draw_single_map_from_tif(
        tif_optimal,
        title=title_name,
        colorbar_name=colorbar_name,
        output_path=str(maps_dir / f'{indicator}_optimal.png'),
        vmin=vmin, vmax=vmax, cmap='RdYlGn',
        shapefile_path=shapefile_path
    )

    # 对照组图
    map_ck = draw_single_map_from_tif(
        tif_ck,
        title=title_name,
        colorbar_name=colorbar_name,
        output_path=str(maps_dir / f'{indicator}_ck.png'),
        vmin=vmin, vmax=vmax, cmap='RdYlGn',
        shapefile_path=shapefile_path
    )

    # 变化图
    map_change = draw_single_map_from_tif(
        tif_change,
        title=title_name,
        colorbar_name=colorbar_name,
        output_path=str(maps_dir / f'{indicator}_change.png'),
        vmin=vmin_change, vmax=vmax_change, cmap='RdBu_r',
        shapefile_path=shapefile_path
    )

    # 比值地图（独立单张）
    map_ratio = draw_single_map_from_tif(
        tif_ratio,
        title=title_name,
        colorbar_name='Ratio',
        output_path=str(maps_dir / f'{indicator}_ratio.png'),
        vmin=vmin_ratio, vmax=vmax_ratio, cmap='RdYlGn',
        shapefile_path=shapefile_path,
        global_mean_as_percentage=True
    )

    # Step 3: 生成拼图（3x1纵向）
    print(f"  [Step 3] 生成拼图...")

    combined_map = create_combined_map_3x1(
        map_optimal, map_ck, map_change,
        indicator, title_name,
        output_path=str(maps_dir / f'{indicator}_comparison_combined.png')
    )

    return {
        'tif_optimal': tif_optimal,
        'tif_ck': tif_ck,
        'tif_change': tif_change,
        'tif_ratio': tif_ratio,  # 新增
        'map_optimal': map_optimal,
        'map_ck': map_ck,
        'map_change': map_change,
        'map_ratio': map_ratio,  # 新增
        'map_combined': combined_map
    }


# =============================================================================
# 拼图函数
# =============================================================================

def create_combined_map_3x1(map_optimal_path, map_ck_path, map_change_path,
                            indicator, title_name, output_path):
    """
    将3张单图拼接成一张纵向布局的对比图

    参数:
        map_optimal_path: 最优情景图路径
        map_ck_path: 对照组图路径
        map_change_path: 变化图路径
        indicator: 指标名称
        title_name: 指标标题名
        output_path: 输出路径
    """
    from PIL import Image, ImageDraw, ImageFont

    # 打开图片
    img_optimal = Image.open(map_optimal_path)
    img_ck = Image.open(map_ck_path)
    img_change = Image.open(map_change_path)

    # 获取尺寸
    width = img_optimal.width
    height = img_optimal.height

    # 创建新图片（3张纵向排列）
    padding = 20
    new_height = height * 3 + padding * 2
    new_image = Image.new('RGB', (width, new_height), color='white')

    # 粘贴图片
    new_image.paste(img_optimal, (0, 0))
    new_image.paste(img_ck, (0, height + padding))
    new_image.paste(img_change, (0, height * 2 + padding * 2))

    # 保存
    new_image.save(output_path, quality=95)

    print(f"    拼图已保存: {output_path}")
    return output_path


# =============================================================================
# 批量处理函数
# =============================================================================

def generate_all_comparison_maps(optimal_results, crop, outday, output_dir,
                                  indicators=['Y', 'ET', 'WUE', 'CH4', 'N2O', 'GWP']):
    """
    为所有指标生成对比地图

    参数:
        optimal_results: DataFrame, 最优结果
        crop: 作物类型
        outday: 输出日期
        output_dir: 输出目录
        indicators: 指标列表
    """
    from .W_comparison import load_ck_data
    from .W_path_manager import get_path_manager

    pm = get_path_manager()

    # 获取参考TIF路径
    reference_tif_path = pm.base_dir / crop / outday / 'x_data_mask.tif'
    if not reference_tif_path.exists():
        reference_tif_path = Path(f'F:/WaterML/{crop}/{outday}/x_data_mask.tif')

    if not reference_tif_path.exists():
        raise FileNotFoundError(f"找不到参考TIF文件: {reference_tif_path}")

    results = {}

    for indicator in indicators:
        if indicator not in optimal_results.columns:
            print(f"  跳过 {indicator}: 最优结果中不存在")
            continue

        # 加载对照组数据
        ck_data = load_ck_data(crop, outday, indicator)
        if ck_data is None:
            print(f"  跳过 {indicator}: 对照组数据不存在")
            continue

        # 合并数据
        merged = optimal_results[['x', 'y', indicator]].copy()
        merged = merged.rename(columns={indicator: 'optimal'})
        merged = merged.merge(ck_data[['x', 'y', 'Ypred']], on=['x', 'y'], how='inner')
        merged = merged.rename(columns={'Ypred': 'ck'})

        # 计算变化
        merged['change'] = merged['optimal'] - merged['ck']

        # 创建指标输出目录
        indicator_output_dir = Path(output_dir) / indicator

        # 生成对比地图
        print(f"\n  处理指标: {indicator}")

        try:
            result = draw_comparison_maps_from_tif(
                comparison_data=merged,
                indicator=indicator,
                output_dir=str(indicator_output_dir),
                reference_tif_path=str(reference_tif_path)
            )
            results[indicator] = result
            print(f"    完成: {indicator}")

        except Exception as e:
            print(f"    错误: {indicator} - {e}")

    return results


if __name__ == "__main__":
    print("对比地图绑制模块 - 高精度版本")

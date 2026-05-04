"""
大洲尺度分析模块
功能：按七大洲进行精确统计分析

创建时间：2025-03-24
最后修改：2025-04-08

改进内容：
1. 使用动态地理编码（reverse_geocoder + pycountry_convert）
2. 添加缓存机制，提高性能
3. 更精确的大洲分类（基于国家边界）
"""

import numpy as np
import pandas as pd
from pathlib import Path
import reverse_geocoder as rg
import pycountry_convert as pc
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='reverse_geocoder')


# =============================================================================
# 大洲代码映射（基于 pycountry_convert）
# =============================================================================

# 大洲代码到大洲名称映射
CONTINENT_CODE_MAP = {
    'AF': 'Africa',
    'AS': 'Asia',
    'EU': 'Europe',
    'NA': 'North America',
    'SA': 'South America',
    'OC': 'Oceania',
    'AN': 'Antarctica'  # 南极洲（研究范围通常不包括）
}

# 大洲显示顺序（按网格数从大到小排序）
CONTINENT_ORDER = ['Asia', 'Europe', 'Africa', 'North America', 'South America', 'Oceania']

# 大洲中文名称映射
CONTINENT_NAME_CN = {
    'Africa': '非洲',
    'Asia': '亚洲',
    'Europe': '欧洲',
    'North America': '北美洲',
    'South America': '南美洲',
    'Oceania': '大洋洲',
    'Antarctica': '南极洲',
    'Global': '全球',
    'Unknown': '未知'
}

# 全局地理编码缓存
_GEO_CACHE = {}


# =============================================================================
# 动态地理编码核心函数
# =============================================================================

def get_geo_cache():
    """获取全局地理编码缓存"""
    return _GEO_CACHE


def clear_geo_cache():
    """清空地理编码缓存"""
    global _GEO_CACHE
    _GEO_CACHE = {}


def assign_continent_dynamic(coords_batch, geo_cache=None):
    """
    使用动态地理编码批量查询大洲（高性能版本）

    参数:
        coords_batch: list of tuples, [(lat, lon), ...] 格式的坐标列表
        geo_cache: dict, 地理编码缓存（可选，默认使用全局缓存）

    返回:
        continents: list, 大洲名称列表
        geo_cache: dict, 更新后的缓存
    """
    if geo_cache is None:
        geo_cache = _GEO_CACHE

    # 坐标降维（保留1位小数，减少缓存体积，提高命中率）
    coords_keys = [f"{round(lat, 1)},{round(lon, 1)}" for lat, lon in coords_batch]

    # 查询未缓存的坐标
    new_keys = list(set(key for key in coords_keys if key not in geo_cache))

    if new_keys:
        new_coords = [tuple(map(float, k.split(','))) for k in new_keys]

        # 分块查询（避免内存溢出，每块5000个坐标）
        chunk_size = 5000
        results = []

        print(f"  正在查询 {len(new_coords)} 个新坐标的地理信息...")
        for i in range(0, len(new_coords), chunk_size):
            chunk = new_coords[i:i+chunk_size]
            chunk_results = rg.search(chunk, mode=1)  # mode=1: 单线程，稳定
            results.extend(chunk_results)

        # 更新缓存：国家代码 -> 大洲代码
        for i, key in enumerate(new_keys):
            try:
                country_code = results[i]['cc']
                continent_code = pc.country_alpha2_to_continent_code(country_code)
                geo_cache[key] = CONTINENT_CODE_MAP.get(continent_code, 'Unknown')
            except Exception as e:
                geo_cache[key] = 'Unknown'

        print(f"  缓存已更新，当前缓存大小: {len(geo_cache)} 个坐标")

    # 返回大洲名称列表
    continents = [geo_cache.get(k, 'Unknown') for k in coords_keys]

    return continents, geo_cache


def assign_continent(row):
    """
    根据经纬度判断所属大洲（单点查询，兼容旧代码）

    参数:
        row: 包含x, y的行或dict

    返回:
        continent: 大洲名称，如果无法识别返回'Unknown'
    """
    lat, lon = row['y'], row['x']
    coords = [(lat, lon)]
    continents, _ = assign_continent_dynamic(coords)
    return continents[0]


def add_continent_column(df, use_batch=True):
    """
    为DataFrame添加大洲列（高性能版本）

    参数:
        df: 包含x, y列的DataFrame
        use_batch: bool, 是否使用批量查询（默认True，大幅提升性能）

    返回:
        df: 添加了continent列的DataFrame
    """
    df = df.copy()

    if use_batch:
        # 批量查询（性能优化：避免逐行查询）
        coords_batch = list(zip(df['y'], df['x']))
        continents, _ = assign_continent_dynamic(coords_batch)
        df['continent'] = continents
    else:
        # 逐行查询（兼容模式，性能较慢）
        df['continent'] = df.apply(assign_continent, axis=1)

    return df


def filter_by_continent(df, continent):
    """
    筛选指定大洲的数据

    参数:
        df: DataFrame
        continent: 大洲名称

    返回:
        filtered_df: 筛选后的DataFrame
    """
    if 'continent' not in df.columns:
        df = add_continent_column(df)

    return df[df['continent'] == continent].copy()


def compute_continent_statistics(df, indicators, value_prefix=''):
    """
    计算每个大洲各指标的统计量

    参数:
        df: 包含x, y和各指标列的DataFrame
        indicators: 指标列表
        value_prefix: 值列前缀（用于区分optimal和ck）

    返回:
        stats_df: 统计结果DataFrame
    """
    # 添加大洲列
    if 'continent' not in df.columns:
        df = add_continent_column(df)

    results = []

    for continent in CONTINENT_ORDER:
        continent_data = df[df['continent'] == continent]

        if len(continent_data) == 0:
            continue

        row = {
            'continent': continent,
            'n_grids': len(continent_data),
            'coverage_pct': len(continent_data) / len(df) * 100
        }

        for indicator in indicators:
            col_name = f"{value_prefix}{indicator}" if value_prefix else indicator
            if col_name in continent_data.columns:
                values = continent_data[col_name].dropna()
                row[f'{indicator}_mean'] = values.mean()
                row[f'{indicator}_std'] = values.std()
                row[f'{indicator}_median'] = values.median()
                row[f'{indicator}_q25'] = values.quantile(0.25)
                row[f'{indicator}_q75'] = values.quantile(0.75)
                row[f'{indicator}_min'] = values.min()
                row[f'{indicator}_max'] = values.max()
                row[f'{indicator}_cv'] = (values.std() / values.mean() * 100) if values.mean() != 0 else 0

        results.append(row)

    # 添加全球统计
    global_row = {
        'continent': 'Global',
        'n_grids': len(df),
        'coverage_pct': 100.0
    }

    for indicator in indicators:
        col_name = f"{value_prefix}{indicator}" if value_prefix else indicator
        if col_name in df.columns:
            values = df[col_name].dropna()
            global_row[f'{indicator}_mean'] = values.mean()
            global_row[f'{indicator}_std'] = values.std()
            global_row[f'{indicator}_median'] = values.median()
            global_row[f'{indicator}_q25'] = values.quantile(0.25)
            global_row[f'{indicator}_q75'] = values.quantile(0.75)
            global_row[f'{indicator}_min'] = values.min()
            global_row[f'{indicator}_max'] = values.max()
            global_row[f'{indicator}_cv'] = (values.std() / values.mean() * 100) if values.mean() != 0 else 0

    results.append(global_row)

    return pd.DataFrame(results)


def compute_continent_comparison(optimal_df, ck_df, indicators):
    """
    计算每个大洲的优化vs对照对比

    参数:
        optimal_df: 最优结果DataFrame（包含x, y和各指标）
        ck_df: 对照组DataFrame（包含x, y和Ypred列，需要按指标分组）
        indicators: 指标列表

    返回:
        comparison_dict: 每个大洲的对比结果字典
    """
    # 添加大洲列
    if 'continent' not in optimal_df.columns:
        optimal_df = add_continent_column(optimal_df)

    if 'continent' not in ck_df.columns:
        ck_df = add_continent_column(ck_df)

    comparison_dict = {}

    for continent in CONTINENT_ORDER + ['Global']:
        opt_data = optimal_df[optimal_df['continent'] == continent] if continent != 'Global' else optimal_df
        ck_data = ck_df[ck_df['continent'] == continent] if continent != 'Global' else ck_df

        if len(opt_data) == 0:
            continue

        rows = []

        for indicator in indicators:
            if indicator not in opt_data.columns:
                continue

            opt_values = opt_data[indicator].dropna()

            # 对于ck，需要从对应的ck文件读取
            # 这里假设ck_df已经按指标组织好
            if 'Ypred' in ck_data.columns:
                ck_values = ck_data['Ypred'].dropna()
            elif indicator in ck_data.columns:
                ck_values = ck_data[indicator].dropna()
            else:
                continue

            opt_mean = opt_values.mean()
            opt_std = opt_values.std()
            ck_mean = ck_values.mean()
            ck_std = ck_values.std()

            # 计算变化
            change = opt_mean - ck_mean
            change_pct = (change / abs(ck_mean) * 100) if ck_mean != 0 else 0

            row = {
                'indicator': indicator,
                'ck_mean': ck_mean,
                'ck_std': ck_std,
                'optimal_mean': opt_mean,
                'optimal_std': opt_std,
                'change': change,
                'change_pct': change_pct,
                'n_grids': len(opt_values)
            }

            rows.append(row)

        if rows:
            comparison_dict[continent] = pd.DataFrame(rows)

    return comparison_dict


def get_continent_summary(df):
    """
    获取各大洲的概要统计

    参数:
        df: DataFrame

    返回:
        summary_df: 概要统计DataFrame
    """
    if 'continent' not in df.columns:
        df = add_continent_column(df)

    results = []

    for continent in CONTINENT_ORDER:
        continent_data = df[df['continent'] == continent]

        if len(continent_data) == 0:
            continue

        results.append({
            'continent': continent,
            'name_cn': CONTINENT_NAME_CN.get(continent, continent),
            'n_grids': len(continent_data),
            'coverage_pct': len(continent_data) / len(df) * 100,
            'lon_range': f"[{continent_data['x'].min():.1f}, {continent_data['x'].max():.1f}]",
            'lat_range': f"[{continent_data['y'].min():.1f}, {continent_data['y'].max():.1f}]"
        })

    return pd.DataFrame(results)


# =============================================================================
# 批量处理函数
# =============================================================================

def analyze_all_continents(optimal_results, ck_data_dict, indicators):
    """
    对所有大洲进行完整分析

    参数:
        optimal_results: 最优结果DataFrame
        ck_data_dict: 对照组数据字典 {indicator: DataFrame}
        indicators: 指标列表

    返回:
        results: 完整分析结果字典
    """
    results = {}

    # 1. 大洲概要
    results['continent_summary'] = get_continent_summary(optimal_results)

    # 2. 各大洲最优比例统计
    results['optimal_ratio_stats'] = compute_continent_statistics(
        optimal_results,
        ['optimal_ratio', 'comprehensive_score']
    )

    # 3. 各大洲指标统计（最优情景）
    results['optimal_indicator_stats'] = compute_continent_statistics(
        optimal_results,
        indicators
    )

    # 4. 各大洲指标统计（对照组）
    # 需要合并各指标的ck数据
    ck_combined = None
    for indicator in indicators:
        if indicator in ck_data_dict:
            ck_df = ck_data_dict[indicator][['x', 'y', 'Ypred']].copy()
            ck_df = ck_df.rename(columns={'Ypred': indicator})
            if ck_combined is None:
                ck_combined = ck_df
            else:
                ck_combined = ck_combined.merge(ck_df, on=['x', 'y'], how='outer')

    if ck_combined is not None:
        results['ck_indicator_stats'] = compute_continent_statistics(
            ck_combined,
            indicators
        )

    return results


if __name__ == "__main__":
    print("大洲尺度分析模块 - 动态地理编码版本")
    print(f"支持 {len(CONTINENT_ORDER)} 个大洲: {CONTINENT_ORDER}")

    print("\n测试动态地理编码:")
    # 测试坐标
    test_coords = [
        (35.0, 120.0),  # 中国（亚洲）
        (50.0, 10.0),   # 德国（欧洲）
        (-30.0, 25.0),  # 南非（非洲）
        (40.0, -100.0), # 美国（北美洲）
        (-20.0, -60.0), # 巴西（南美洲）
        (-35.0, 150.0)  # 澳大利亚（大洋洲）
    ]

    continents, cache = assign_continent_dynamic(test_coords)

    print("\n坐标测试结果:")
    for i, (lat, lon) in enumerate(test_coords):
        print(f"  ({lat}, {lon}) -> {continents[i]} ({CONTINENT_NAME_CN.get(continents[i], '未知')})")

    print(f"\n缓存大小: {len(cache)} 个坐标")
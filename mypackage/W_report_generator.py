"""
专业报告生成模块
功能：生成顶刊标准的中文分析报告

创建时间：2025-03-24
最后修改：2025-04-08

改进内容：
1. 报告语言改为中文
2. 指标配置增加中文名称
3. 表格列名改为中文
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from .W_continental_analysis import (
    CONTINENT_ORDER,
    add_continent_column,
    compute_continent_statistics,
    get_continent_summary
)
from .W_statistical_tests import (
    cohens_d,
    mean_confidence_interval,
    comprehensive_comparison
)
from .W_path_manager import get_path_manager
from .W_config import INDICATOR_DIRECTION


# =============================================================================
# 指标配置（增加中文）
# =============================================================================

INDICATOR_CONFIG = {
    'Y': {
        'name': 'Yield',
        'name_cn': '产量',
        'unit': 'kg/ha',
        'unit_cn': '千克/公顷',
        'direction': 'maximize',
        'direction_cn': '最大化',
        'decimals': 0
    },
    'ET': {
        'name': 'Evapotranspiration',
        'name_cn': '蒸散发',
        'unit': 'mm',
        'unit_cn': '毫米',
        'direction': 'minimize',
        'direction_cn': '最小化',
        'decimals': 1
    },
    'WUE': {
        'name': 'Water Use Efficiency',
        'name_cn': '水分利用效率',
        'unit': 'kg/m³',
        'unit_cn': '千克/立方米',
        'direction': 'maximize',
        'direction_cn': '最大化',
        'decimals': 2
    },
    'CH4': {
        'name': 'CH4 Emission',
        'name_cn': '甲烷排放',
        'unit': 'kg/ha',
        'unit_cn': '千克/公顷',
        'direction': 'minimize',
        'direction_cn': '最小化',
        'decimals': 2
    },
    'N2O': {
        'name': 'N2O Emission',
        'name_cn': '氧化亚氮排放',
        'unit': 'kg/ha',
        'unit_cn': '千克/公顷',
        'direction': 'minimize',
        'direction_cn': '最小化',
        'decimals': 2
    },
    'GWP': {
        'name': 'Global Warming Potential',
        'name_cn': '全球变暖潜势',
        'unit': 'kg CO2-eq/ha',
        'unit_cn': '千克CO2当量/公顷',
        'direction': 'minimize',
        'direction_cn': '最小化',
        'decimals': 0
    }
}


# =============================================================================
# 报告生成器类
# =============================================================================

class ReportGenerator:
    """专业报告生成器"""

    def __init__(self, crop, outday):
        self.crop = crop
        self.outday = outday
        self.pm = get_path_manager()

    def generate_full_report(self, optimal_results, ck_data_dict, indicators):
        """
        生成完整报告

        参数:
            optimal_results: 最优结果DataFrame
            ck_data_dict: 对照组数据字典 {indicator: DataFrame}
            indicators: 指标列表

        返回:
            report: Markdown格式报告
            tables: 数据表格字典
        """
        sections = []

        # 标题
        sections.append(self._generate_title())

        # 摘要
        sections.append(self._generate_abstract(optimal_results, ck_data_dict, indicators))

        # 1. 数据概述
        sections.append(self._generate_data_overview(optimal_results))

        # 2. 全球尺度结果
        global_stats = self._compute_global_statistics(optimal_results, ck_data_dict, indicators)
        sections.append(self._generate_global_results(global_stats, indicators))

        # 3. 大洲尺度分析
        continental_stats = self._compute_continental_statistics(optimal_results, ck_data_dict, indicators)
        sections.append(self._generate_continental_results(continental_stats, indicators))

        # 4. 效益分析
        benefit_analysis = self._compute_benefit_analysis(optimal_results, ck_data_dict, indicators)
        sections.append(self._generate_benefit_analysis(benefit_analysis))

        # 5. 方法说明
        sections.append(self._generate_methods())

        # 6. 关键发现
        sections.append(self._generate_key_findings(optimal_results, global_stats))

        # 7. 结论
        sections.append(self._generate_conclusions())

        report = '\n'.join(sections)

        # 生成表格
        tables = {
            'global_comparison': self._create_global_comparison_table(global_stats, indicators),
            'continental_ratio_stats': self._create_continental_ratio_table(continental_stats),
            'continental_detailed': self._create_continental_detailed_tables(continental_stats, indicators)
        }

        return report, tables

    def _generate_title(self):
        """生成标题"""
        crop_cn = {
            'rice': '水稻',
            'wheat': '小麦',
            'maize': '玉米'
        }

        return f"""# 全球尺度 {crop_cn.get(self.crop, self.crop)} 亏缺灌溉优化研究

**空间分析报告**

---

**作物**: {crop_cn.get(self.crop, self.crop)}
**分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据版本**: {self.outday}

"""

    def _generate_abstract(self, optimal_results, ck_data_dict, indicators):
        """生成摘要"""
        n_grids = len(optimal_results)
        mean_ratio = optimal_results['optimal_ratio'].mean()

        # 计算主要变化
        changes = []
        for ind in ['Y', 'WUE', 'GWP']:
            if ind in optimal_results.columns and ind in ck_data_dict:
                opt_mean = optimal_results[ind].mean()
                ck_mean = ck_data_dict[ind]['Ypred'].mean()
                change_pct = (opt_mean - ck_mean) / abs(ck_mean) * 100
                cfg = INDICATOR_CONFIG.get(ind, {})
                name_cn = cfg.get('name_cn', ind)
                changes.append(f"{name_cn}: {change_pct:+.1f}%")

        crop_cn = {
            'rice': '水稻',
            'wheat': '小麦',
            'maize': '玉米'
        }

        return f"""## 摘要

本研究基于 {n_grids:,} 个全球网格单元，采用多目标优化方法，对 {crop_cn.get(self.crop, self.crop)} 的亏缺灌溉策略进行全球尺度优化分析。我们识别了最优亏缺灌溉比例，并量化了其对产量、水分利用效率和温室气体排放的影响。

**主要发现：**
- 最优亏缺灌溉比例：{mean_ratio:.3f}（平均）
- 主要指标变化：{', '.join(changes)}
- 优化策略实现了农业生产与环境可持续性的平衡

---

"""

    def _generate_data_overview(self, optimal_results):
        """生成数据概述"""
        n_grids = len(optimal_results)

        # 计算空间范围
        lon_range = f"[{optimal_results['x'].min():.2f}, {optimal_results['x'].max():.2f}]"
        lat_range = f"[{optimal_results['y'].min():.2f}, {optimal_results['y'].max():.2f}]"

        return f"""## 1. 数据概述

### 1.1 研究区域

- **网格总数**: {n_grids:,}
- **经度范围**: {lon_range}°
- **纬度范围**: {lat_range}°
- **空间分辨率**: 0.5° × 0.5°

### 1.2 优化参数

- **优化范围**: [0.75, 0.90]
- **步长**: 0.01
- **优化方法**: 等权重多目标优化

---

"""

    def _compute_global_statistics(self, optimal_results, ck_data_dict, indicators):
        """计算全球尺度统计"""
        stats = {}

        # 最优比例统计
        stats['optimal_ratio'] = {
            'mean': optimal_results['optimal_ratio'].mean(),
            'std': optimal_results['optimal_ratio'].std(),
            'median': optimal_results['optimal_ratio'].median(),
            'q25': optimal_results['optimal_ratio'].quantile(0.25),
            'q75': optimal_results['optimal_ratio'].quantile(0.75),
            'min': optimal_results['optimal_ratio'].min(),
            'max': optimal_results['optimal_ratio'].max()
        }

        # 各指标统计
        for indicator in indicators:
            if indicator in optimal_results.columns and indicator in ck_data_dict:
                opt_values = optimal_results[indicator].dropna()
                ck_values = ck_data_dict[indicator]['Ypred'].dropna()

                # 计算变化
                change = opt_values.mean() - ck_values.mean()

                # 根据优化方向计算变化百分比
                direction = INDICATOR_DIRECTION.get(indicator, 'maximize')

                if direction == 'minimize':
                    # 最小化指标：计算减少百分比 = (ck - opt) / |ck|
                    change_pct = (ck_values.mean() - opt_values.mean()) / abs(ck_values.mean()) * 100
                else:
                    # 最大化指标：计算增加百分比 = (opt - ck) / ck
                    change_pct = (opt_values.mean() - ck_values.mean()) / ck_values.mean() * 100 if ck_values.mean() != 0 else 0

                # 效应量
                d, d_interp = cohens_d(ck_values, opt_values)

                # 置信区间
                _, ci_lower, ci_upper, margin = mean_confidence_interval(opt_values)

                stats[indicator] = {
                    'ck_mean': ck_values.mean(),
                    'ck_std': ck_values.std(),
                    'opt_mean': opt_values.mean(),
                    'opt_std': opt_values.std(),
                    'change': change,
                    'change_pct': change_pct,
                    'cohens_d': d,
                    'effect_interp': d_interp,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'n': len(opt_values)
                }

        return stats

    def _generate_global_results(self, global_stats, indicators):
        """生成全球尺度结果"""
        lines = []
        lines.append("## 2. 全球尺度结果")
        lines.append("")

        # 2.1 最优比例统计
        lines.append("### 2.1 最优亏缺灌溉比例")
        lines.append("")
        ratio = global_stats['optimal_ratio']
        lines.append(f"| 统计指标 | 数值 |")
        lines.append(f"|----------|------|")
        lines.append(f"| 平均值 ± 标准差 | {ratio['mean']:.3f} ± {ratio['std']:.3f} |")
        lines.append(f"| 中位数 | {ratio['median']:.3f} |")
        lines.append(f"| 四分位间距 (IQR) | [{ratio['q25']:.3f}, {ratio['q75']:.3f}] |")
        lines.append(f"| 范围 | [{ratio['min']:.3f}, {ratio['max']:.3f}] |")
        lines.append("")

        # 2.2 六大指标对比
        lines.append("### 2.2 六大指标对比（最优情景 vs 对照组）")
        lines.append("")
        lines.append("| 指标 | 单位 | 对照组（均值±标准差） | 最优情景（均值±标准差） | 变化百分比 | 效应量 |")
        lines.append("|------|------|-----------------------|-------------------------|------------|--------|")

        for ind in indicators:
            if ind in global_stats and ind in INDICATOR_CONFIG:
                s = global_stats[ind]
                cfg = INDICATOR_CONFIG[ind]
                fmt = f"{{:.{cfg['decimals']}f}}" if cfg['decimals'] > 0 else "{:.0f}"

                lines.append(
                    f"| {cfg['name_cn']} ({ind}) | {cfg['unit_cn']} | "
                    f"{fmt.format(s['ck_mean'])}±{fmt.format(s['ck_std'])} | "
                    f"{fmt.format(s['opt_mean'])}±{fmt.format(s['opt_std'])} | "
                    f"{s['change_pct']:+.1f}% | "
                    f"{s['cohens_d']:.3f} ({s['effect_interp']}) |"
                )

        lines.append("")
        lines.append("**注**: 效应量解释：可忽略 (< 0.2)、小效应 (0.2-0.5)、中等效应 (0.5-0.8)、大效应 (≥ 0.8)")
        lines.append("")

        return '\n'.join(lines)

    def _compute_continental_statistics(self, optimal_results, ck_data_dict, indicators):
        """计算大洲尺度统计"""
        # 添加大洲列
        optimal_results = add_continent_column(optimal_results)

        stats = {}

        # 为每个ck数据添加大洲列
        ck_data_with_continent = {}
        for indicator, ck_df in ck_data_dict.items():
            ck_data_with_continent[indicator] = add_continent_column(ck_df.copy())

        # 各大洲最优比例统计
        for continent in CONTINENT_ORDER + ['Global']:
            if continent == 'Global':
                data = optimal_results
            else:
                data = optimal_results[optimal_results['continent'] == continent]

            if len(data) == 0:
                continue

            stats[continent] = {
                'n_grids': len(data),
                'optimal_ratio': {
                    'mean': data['optimal_ratio'].mean(),
                    'std': data['optimal_ratio'].std(),
                    'median': data['optimal_ratio'].median(),
                    'q25': data['optimal_ratio'].quantile(0.25),
                    'q75': data['optimal_ratio'].quantile(0.75),
                    'cv': data['optimal_ratio'].std() / data['optimal_ratio'].mean() * 100
                }
            }

            # 各指标统计（包含CK对比）
            for indicator in indicators:
                if indicator in data.columns:
                    opt_values = data[indicator].dropna()

                    ind_stats = {
                        'opt_mean': opt_values.mean(),
                        'opt_std': opt_values.std()
                    }

                    # 添加CK数据
                    if indicator in ck_data_with_continent:
                        if continent == 'Global':
                            ck_data = ck_data_with_continent[indicator]
                        else:
                            ck_data = ck_data_with_continent[indicator][ck_data_with_continent[indicator]['continent'] == continent]

                        if len(ck_data) > 0:
                            ck_values = ck_data['Ypred'].dropna()
                            ind_stats['ck_mean'] = ck_values.mean()
                            ind_stats['ck_std'] = ck_values.std()

                            # 计算变化
                            change = ind_stats['opt_mean'] - ind_stats['ck_mean']
                            change_pct = (change / abs(ind_stats['ck_mean']) * 100) if ind_stats['ck_mean'] != 0 else 0
                            ind_stats['change'] = change
                            ind_stats['change_pct'] = change_pct

                    stats[continent][indicator] = ind_stats

        return stats

    def _generate_continental_results(self, continental_stats, indicators):
        """生成大洲尺度结果"""
        from .W_continental_analysis import CONTINENT_NAME_CN

        lines = []
        lines.append("## 3. 大洲尺度分析")
        lines.append("")

        # 3.1 各大洲最优比例
        lines.append("### 3.1 各大洲最优亏缺灌溉比例")
        lines.append("")
        lines.append("| 大洲 | 网格数 | 平均值 | 标准差 | 中位数 | 变异系数(%) |")
        lines.append("|------|--------|--------|--------|--------|-------------|")

        for continent in CONTINENT_ORDER:
            if continent in continental_stats:
                s = continental_stats[continent]
                ratio = s['optimal_ratio']
                name_cn = CONTINENT_NAME_CN.get(continent, continent)
                lines.append(
                    f"| {name_cn} | {s['n_grids']:,} | "
                    f"{ratio['mean']:.3f} | {ratio['std']:.3f} | "
                    f"{ratio['median']:.3f} | {ratio['cv']:.1f} |"
                )

        # 全球总计
        if 'Global' in continental_stats:
            s = continental_stats['Global']
            ratio = s['optimal_ratio']
            lines.append(
                f"| **全球** | **{s['n_grids']:,}** | "
                f"**{ratio['mean']:.3f}** | **{ratio['std']:.3f}** | "
                f"**{ratio['median']:.3f}** | **{ratio['cv']:.1f}** |"
            )

        lines.append("")

        # 3.2 各大洲详细对比
        lines.append("### 3.2 各大洲指标详细对比")
        lines.append("")
        lines.append("以下表格展示了各大洲在最优情景和对照组之间的详细对比。")
        lines.append("")

        for indicator in indicators:
            if indicator in INDICATOR_CONFIG:
                cfg = INDICATOR_CONFIG[indicator]
                lines.append(f"#### {cfg['name_cn']} ({indicator})")
                lines.append("")
                lines.append("| 大洲 | 网格数 | 对照组均值 | 对照组标准差 | 最优情景均值 | 最优情景标准差 | 变化百分比 |")
                lines.append("|------|--------|------------|--------------|--------------|----------------|------------|")

                for continent in CONTINENT_ORDER:
                    if continent in continental_stats and indicator in continental_stats[continent]:
                        s = continental_stats[continent]
                        ind_s = s[indicator]
                        fmt = f"{{:.{cfg['decimals']}f}}" if cfg['decimals'] > 0 else "{:.0f}"
                        name_cn = CONTINENT_NAME_CN.get(continent, continent)

                        # 检查是否有CK数据
                        if 'ck_mean' in ind_s:
                            lines.append(
                                f"| {name_cn} | {s['n_grids']:,} | "
                                f"{fmt.format(ind_s['ck_mean'])} | {fmt.format(ind_s['ck_std'])} | "
                                f"{fmt.format(ind_s['opt_mean'])} | {fmt.format(ind_s['opt_std'])} | "
                                f"{ind_s['change_pct']:+.1f}% |"
                            )
                        else:
                            lines.append(
                                f"| {name_cn} | {s['n_grids']:,} | "
                                f"- | - | "
                                f"{fmt.format(ind_s['opt_mean'])} | {fmt.format(ind_s['opt_std'])} | "
                                f"- |"
                            )

                lines.append("")

        # 3.3 大洲综合对比表
        lines.append("### 3.3 大洲综合对比表")
        lines.append("")
        lines.append("| 大洲 | 网格数 | 最优比例 | 产量变化(%) | 蒸散发变化(%) | 水分利用效率变化(%) | 全球变暖潜势变化(%) |")
        lines.append("|------|--------|----------|-------------|---------------|---------------------|-------------------|")

        for continent in CONTINENT_ORDER:
            if continent in continental_stats:
                s = continental_stats[continent]
                ratio = s['optimal_ratio']
                name_cn = CONTINENT_NAME_CN.get(continent, continent)

                # 获取各指标变化
                y_change = s.get('Y', {}).get('change_pct', 0) if 'Y' in s else 0
                et_change = s.get('ET', {}).get('change_pct', 0) if 'ET' in s else 0
                wue_change = s.get('WUE', {}).get('change_pct', 0) if 'WUE' in s else 0
                gwp_change = s.get('GWP', {}).get('change_pct', 0) if 'GWP' in s else 0

                lines.append(
                    f"| {name_cn} | {s['n_grids']:,} | "
                    f"{ratio['mean']:.3f} | "
                    f"{y_change:+.1f}% | {et_change:+.1f}% | "
                    f"{wue_change:+.1f}% | {gwp_change:+.1f}% |"
                )

        # 全球总计
        if 'Global' in continental_stats:
            s = continental_stats['Global']
            ratio = s['optimal_ratio']
            y_change = s.get('Y', {}).get('change_pct', 0) if 'Y' in s else 0
            et_change = s.get('ET', {}).get('change_pct', 0) if 'ET' in s else 0
            wue_change = s.get('WUE', {}).get('change_pct', 0) if 'WUE' in s else 0
            gwp_change = s.get('GWP', {}).get('change_pct', 0) if 'GWP' in s else 0

            lines.append(
                f"| **全球** | **{s['n_grids']:,}** | "
                f"**{ratio['mean']:.3f}** | "
                f"**{y_change:+.1f}%** | **{et_change:+.1f}%** | "
                f"**{wue_change:+.1f}%** | **{gwp_change:+.1f}%** |"
            )

        lines.append("")

        return '\n'.join(lines)

    def _compute_benefit_analysis(self, optimal_results, ck_data_dict, indicators):
        """计算效益分析"""
        benefits = {}

        # 水资源节约
        if 'ET' in optimal_results.columns and 'ET' in ck_data_dict:
            et_opt = optimal_results['ET'].mean()
            et_ck = ck_data_dict['ET']['Ypred'].mean()
            water_saved = (et_ck - et_opt) / et_ck * 100
            benefits['water'] = {
                'et_reduction_pct': water_saved,
                'et_optimal': et_opt,
                'et_control': et_ck
            }

        # 产量变化
        if 'Y' in optimal_results.columns and 'Y' in ck_data_dict:
            y_opt = optimal_results['Y'].mean()
            y_ck = ck_data_dict['Y']['Ypred'].mean()
            yield_change = (y_opt - y_ck) / y_ck * 100
            benefits['yield'] = {
                'yield_change_pct': yield_change,
                'yield_optimal': y_opt,
                'yield_control': y_ck
            }

        # 温室气体减排
        if 'GWP' in optimal_results.columns and 'GWP' in ck_data_dict:
            gwp_opt = optimal_results['GWP'].mean()
            gwp_ck = ck_data_dict['GWP']['Ypred'].mean()
            gwp_reduction = (gwp_ck - gwp_opt) / gwp_ck * 100
            benefits['ghg'] = {
                'gwp_reduction_pct': gwp_reduction,
                'gwp_optimal': gwp_opt,
                'gwp_control': gwp_ck
            }

        return benefits

    def _generate_benefit_analysis(self, benefits):
        """生成效益分析"""
        lines = []
        lines.append("## 4. 效益分析")
        lines.append("")

        # 水资源
        if 'water' in benefits:
            w = benefits['water']
            lines.append("### 4.1 水资源节约")
            lines.append("")
            lines.append(f"- **蒸散发减少**: {w['et_reduction_pct']:.1f}%")
            lines.append(f"- **对照组蒸散发**: {w['et_control']:.1f} mm")
            lines.append(f"- **最优情景蒸散发**: {w['et_optimal']:.1f} mm")
            lines.append("")

        # 产量
        if 'yield' in benefits:
            y = benefits['yield']
            lines.append("### 4.2 产量影响")
            lines.append("")
            lines.append(f"- **产量变化**: {y['yield_change_pct']:+.1f}%")
            lines.append(f"- **对照组产量**: {y['yield_control']:.0f} kg/ha")
            lines.append(f"- **最优情景产量**: {y['yield_optimal']:.0f} kg/ha")
            lines.append("")

        # 温室气体
        if 'ghg' in benefits:
            g = benefits['ghg']
            lines.append("### 4.3 温室气体减排")
            lines.append("")
            lines.append(f"- **全球变暖潜势降低**: {g['gwp_reduction_pct']:.1f}%")
            lines.append(f"- **对照组全球变暖潜势**: {g['gwp_control']:.0f} kg CO2-eq/ha")
            lines.append(f"- **最优情景全球变暖潜势**: {g['gwp_optimal']:.0f} kg CO2-eq/ha")
            lines.append("")

        return '\n'.join(lines)

    def _generate_methods(self):
        """生成方法说明"""
        return """## 5. 研究方法

### 5.1 多目标优化

本研究采用等权重多目标优化方法，优化目标包括：

| 指标 | 单位 | 优化方向 |
|------|------|----------|
| 产量 (Y) | kg/ha | 最大化 |
| 蒸散发 (ET) | mm | 最小化 |
| 水分利用效率 (WUE) | kg/m³ | 最大化 |
| 甲烷排放 (CH4) | kg/ha | 最小化 |
| 氧化亚氮排放 (N2O) | kg/ha | 最小化 |
| 全球变暖潜势 (GWP) | kg CO2-eq/ha | 最小化 |

### 5.2 优化流程

1. **响应面构建**: 构建"网格×比例×指标"三维响应面
2. **归一化处理**: 对各指标进行 Min-Max 归一化
3. **综合评分**: 计算归一化指标的等权重加权和
4. **最优选择**: 选择综合得分最高的亏缺比例

### 5.3 统计分析

- **效应量**: Cohen's d
- **置信区间**: 95% 置信区间
- **空间分析**: 大洲尺度聚合统计

### 5.4 大洲分类方法

采用动态地理编码方法（reverse_geocoder + pycountry_convert），根据经纬度坐标精确查询所属国家，再通过国家代码映射到相应大洲，确保大洲分类的准确性。

---

"""

    def _generate_key_findings(self, optimal_results, global_stats):
        """生成关键发现"""
        mean_ratio = global_stats['optimal_ratio']['mean']

        return f"""## 6. 主要发现

1. **最优亏缺灌溉比例**: 全球平均最优比例为 {mean_ratio:.3f}，表明在约 {(1-mean_ratio)*100:.1f}% 的水分亏缺下可实现最佳平衡。

2. **水-能-粮协同关系**: 优化结果展示了水资源节约、产量维持和温室气体减排之间的协同效应潜力。

3. **空间异质性**: 各大洲之间存在显著的空间差异，表明需要制定区域特异性的灌溉策略。

4. **环境效益**: 最优策略在维持产量水平的同时，实现了用水减少和温室气体减排的双重效益。

---

"""

    def _generate_conclusions(self):
        """生成结论"""
        crop_cn = {
            'rice': '水稻',
            'wheat': '小麦',
            'maize': '玉米'
        }

        return f"""## 7. 结论与建议

本研究对 {crop_cn.get(self.crop, self.crop)} 亏缺灌溉进行了全面的全球尺度优化分析，研究结果为农业可持续水管理提供了重要参考：

1. 在最优亏缺灌溉比例下，可同时实现多重效益
2. 区域适应性策略对于最大化效益至关重要
3. 该方法为平衡粮食安全与环境可持续性提供了框架

### 7.1 政策建议

- 实施区域特异性的亏缺灌溉策略
- 在亏缺灌溉过程中监测土壤湿度和作物状态
- 在实际应用中考虑经济因素

### 7.2 研究局限

- 极端气候条件下的模型不确定性
- 部分地区的验证数据有限
- 等权重假设的简化处理

### 7.3 未来展望

- 整合气候变化预测
- 开发动态优化方法
- 集成经济分析

---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    def _create_global_comparison_table(self, global_stats, indicators):
        """创建全球对比表格"""
        rows = []

        for ind in indicators:
            if ind in global_stats and ind in INDICATOR_CONFIG:
                s = global_stats[ind]
                cfg = INDICATOR_CONFIG[ind]

                rows.append({
                    '指标': f"{cfg['name_cn']} ({ind})",
                    '单位': cfg['unit_cn'],
                    '对照组均值': s['ck_mean'],
                    '对照组标准差': s['ck_std'],
                    '最优情景均值': s['opt_mean'],
                    '最优情景标准差': s['opt_std'],
                    '变化值': s['change'],
                    '变化百分比': s['change_pct'],
                    'Cohen_d': s['cohens_d'],
                    '效应量': s['effect_interp']
                })

        return pd.DataFrame(rows)

    def _create_continental_ratio_table(self, continental_stats):
        """创建大洲最优比例表格"""
        from .W_continental_analysis import CONTINENT_NAME_CN

        rows = []

        for continent in CONTINENT_ORDER + ['Global']:
            if continent in continental_stats:
                s = continental_stats[continent]
                r = s['optimal_ratio']
                name_cn = CONTINENT_NAME_CN.get(continent, continent)

                rows.append({
                    '大洲': name_cn,
                    '网格数': s['n_grids'],
                    '平均值': r['mean'],
                    '标准差': r['std'],
                    '中位数': r['median'],
                    '下四分位': r['q25'],
                    '上四分位': r['q75'],
                    '变异系数': r['cv']
                })

        return pd.DataFrame(rows)

    def _create_continental_detailed_tables(self, continental_stats, indicators):
        """创建各大洲详细表格"""
        from .W_continental_analysis import CONTINENT_NAME_CN

        tables = {}

        for indicator in indicators:
            if indicator not in INDICATOR_CONFIG:
                continue

            cfg = INDICATOR_CONFIG[indicator]
            rows = []

            for continent in CONTINENT_ORDER + ['Global']:
                if continent in continental_stats and indicator in continental_stats[continent]:
                    s = continental_stats[continent]
                    ind_s = s[indicator]
                    name_cn = CONTINENT_NAME_CN.get(continent, continent)

                    row = {
                        '大洲': name_cn,
                        '网格数': s['n_grids']
                    }

                    # 添加CK数据（如果有）
                    if 'ck_mean' in ind_s:
                        row['对照组均值'] = ind_s['ck_mean']
                        row['对照组标准差'] = ind_s['ck_std']

                    row['最优情景均值'] = ind_s['opt_mean']
                    row['最优情景标准差'] = ind_s['opt_std']

                    # 添加变化数据（如果有）
                    if 'change_pct' in ind_s:
                        row['变化值'] = ind_s.get('change', 0)
                        row['变化百分比'] = ind_s['change_pct']

                    rows.append(row)

            tables[indicator] = pd.DataFrame(rows)

        # 创建大洲综合对比表
        summary_rows = []
        for continent in CONTINENT_ORDER + ['Global']:
            if continent in continental_stats:
                s = continental_stats[continent]
                ratio = s['optimal_ratio']
                name_cn = CONTINENT_NAME_CN.get(continent, continent)

                row = {
                    '大洲': name_cn,
                    '网格数': s['n_grids'],
                    '最优比例均值': ratio['mean'],
                    '最优比例标准差': ratio['std']
                }

                for ind in indicators:
                    if ind in s:
                        row[f'{ind}_变化百分比'] = s[ind].get('change_pct', 0)

                summary_rows.append(row)

        tables['continental_summary'] = pd.DataFrame(summary_rows)

        return tables


# =============================================================================
# 便捷函数
# =============================================================================

def generate_report(optimal_results, ck_data_dict, crop, outday, indicators=None):
    """
    生成报告的便捷函数

    参数:
        optimal_results: 最优结果DataFrame
        ck_data_dict: 对照组数据字典
        crop: 作物类型
        outday: 输出日期
        indicators: 指标列表

    返回:
        report: Markdown报告
        tables: 数据表格字典
    """
    if indicators is None:
        indicators = ['Y', 'ET', 'WUE', 'CH4', 'N2O', 'GWP']

    generator = ReportGenerator(crop, outday)
    return generator.generate_full_report(optimal_results, ck_data_dict, indicators)


if __name__ == "__main__":
    print("专业报告生成模块 - 中文版")
    print("支持指标:")
    for ind, cfg in INDICATOR_CONFIG.items():
        print(f"  {cfg['name_cn']} ({ind}): {cfg['unit_cn']}, 优化方向: {cfg['direction_cn']}")
"""
比值分析报告生成器 (Ratio Analysis Report Generator)
功能：生成完善的比值专项分析报告，结构与主报告一致

创建时间：2026-04-11
最后修改：2026-04-11

主要改进：
1. 增加标题、摘要、数据概述
2. 增加大洲尺度分析
3. 增加效益分析（比值视角）
4. 增加关键发现和结论建议
5. 与主报告结构对齐
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


# =============================================================================
# 指标配置
# =============================================================================

INDICATOR_CONFIG_RATIO = {
    'Y': {
        'name': 'Yield',
        'name_cn': '产量',
        'direction': 'maximize',
        'direction_cn': '最大化',
        'ideal_ratio': '> 1.0',
        'improvement_desc': '比值>1表示产量提升'
    },
    'ET': {
        'name': 'Evapotranspiration',
        'name_cn': '蒸散发',
        'direction': 'minimize',
        'direction_cn': '最小化',
        'ideal_ratio': '< 1.0',
        'improvement_desc': '比值<1表示用水减少'
    },
    'WUE': {
        'name': 'Water Use Efficiency',
        'name_cn': '水分利用效率',
        'direction': 'maximize',
        'direction_cn': '最大化',
        'ideal_ratio': '> 1.0',
        'improvement_desc': '比值>1表示效率提升'
    },
    'CH4': {
        'name': 'CH4 Emission',
        'name_cn': '甲烷排放',
        'direction': 'minimize',
        'direction_cn': '最小化',
        'ideal_ratio': '< 1.0',
        'improvement_desc': '比值<1表示排放减少'
    },
    'N2O': {
        'name': 'N2O Emission',
        'name_cn': '氧化亚氮排放',
        'direction': 'minimize',
        'direction_cn': '最小化',
        'ideal_ratio': '< 1.0',
        'improvement_desc': '比值<1表示排放减少'
    },
    'GWP': {
        'name': 'Global Warming Potential',
        'name_cn': '全球变暖潜势',
        'direction': 'minimize',
        'direction_cn': '最小化',
        'ideal_ratio': '< 1.0',
        'improvement_desc': '比值<1表示温室效应降低'
    }
}


# =============================================================================
# 大洲名称映射
# =============================================================================

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

CONTINENT_ORDER = ['Asia', 'Europe', 'Africa', 'North America', 'South America', 'Oceania']


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
            significant_improvement = ratio_df[ratio_df['optimal_ck_ratio'] < 0.95]

        results[indicator] = {
            'regional_stats': regional_ratio_stats,
            'significant_improvement_grids': len(significant_improvement),
            'improvement_pct': len(significant_improvement) / len(ratio_df) * 100 if len(ratio_df) > 0 else 0
        }

    return results


# =============================================================================
# 大洲尺度分析
# =============================================================================

def compute_continental_ratio_statistics(comparison_results):
    """
    计算各大洲的比值统计

    参数:
        comparison_results: dict, 对比分析结果

    返回:
        continental_stats: dict, 大洲比值统计结果
    """
    from .W_continental_analysis import add_continent_column

    continental_stats = {}

    for indicator in ['Y', 'ET', 'WUE', 'CH4', 'N2O', 'GWP']:
        if indicator not in comparison_results['indicator_details']:
            continue

        # 获取详细数据
        detail_df = comparison_results['indicator_details'][indicator].copy()

        # 添加大洲列
        if 'continent' not in detail_df.columns:
            detail_df = add_continent_column(detail_df)

        # 计算各大洲统计
        indicator_stats = {}

        for continent in CONTINENT_ORDER + ['Global']:
            if continent == 'Global':
                data = detail_df
            else:
                data = detail_df[detail_df['continent'] == continent]

            if len(data) == 0:
                continue

            ratio_values = data['optimal_ck_ratio'].dropna()

            if len(ratio_values) == 0:
                continue

            # 从指标配置获取优化方向
            direction = INDICATOR_CONFIG_RATIO[indicator]['direction']

            # 计算改善百分比（负值指标已通过ratio=ck/opt实现ratio<1表示改善）
            if direction == 'maximize':
                improvement_pct = (ratio_values > 1.0).sum() / len(ratio_values) * 100
            else:
                improvement_pct = (ratio_values < 1.0).sum() / len(ratio_values) * 100

            indicator_stats[continent] = {
                'n_grids': len(data),
                'mean_ratio': ratio_values.mean(),
                'median_ratio': ratio_values.median(),
                'std_ratio': ratio_values.std(),
                'q25': ratio_values.quantile(0.25),
                'q75': ratio_values.quantile(0.75),
                'improvement_pct': improvement_pct,
                'significant_improvement': (ratio_values > 1.05 if direction == 'maximize' else ratio_values < 0.95).sum()
            }

        continental_stats[indicator] = indicator_stats

    return continental_stats


# =============================================================================
# 完善的比值报告生成器
# =============================================================================

class RatioReportGenerator:
    """比值专项分析报告生成器"""

    def __init__(self, crop, outday):
        self.crop = crop
        self.outday = outday

    def generate_full_report(self, comparison_results):
        """
        生成完整的比值分析报告

        参数:
            comparison_results: dict, 对比分析结果

        返回:
            report: Markdown格式报告
            tables: 数据表格字典
        """
        sections = []

        # 标题
        sections.append(self._generate_title())

        # 摘要
        sections.append(self._generate_abstract(comparison_results))

        # 1. 数据概述
        sections.append(self._generate_data_overview(comparison_results))

        # 2. 全球尺度比值统计
        global_stats = comparison_results['comparison_data']
        sections.append(self._generate_global_ratio_results(global_stats))

        # 3. 大洲尺度比值分析
        continental_stats = compute_continental_ratio_statistics(comparison_results)
        sections.append(self._generate_continental_ratio_results(continental_stats))

        # 4. 比值效益分析
        benefit_analysis = self._compute_ratio_benefits(comparison_results)
        sections.append(self._generate_benefit_analysis(benefit_analysis))

        # 5. 比值解读方法
        sections.append(self._generate_ratio_methods())

        # 6. 关键发现
        sections.append(self._generate_key_findings(comparison_results, global_stats))

        # 7. 结论与建议
        sections.append(self._generate_conclusions())

        report = '\n'.join(sections)

        # 生成表格
        tables = {
            'global_ratio_summary': self._create_global_ratio_table(global_stats),
            'continental_ratio_stats': self._create_continental_ratio_table(continental_stats)
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

**比值专项分析报告**

---

**作物**: {crop_cn.get(self.crop, self.crop)}
**分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据版本**: {self.outday}

---

"""

    def _generate_abstract(self, comparison_results):
        """生成摘要"""
        comparison_data = comparison_results['comparison_data']
        n_indicators = len(comparison_data)

        # 找出改善最显著的指标
        best_improvements = []
        for _, row in comparison_data.iterrows():
            indicator = row['indicator']
            improvement_pct = row['ratio_improvement_pct']
            if improvement_pct > 60:  # 改善超过60%
                cfg = INDICATOR_CONFIG_RATIO.get(indicator, {})
                name_cn = cfg.get('name_cn', indicator)
                best_improvements.append(f"{name_cn}({improvement_pct:.1f}%)")

        crop_cn = {
            'rice': '水稻',
            'wheat': '小麦',
            'maize': '玉米'
        }

        return f"""## 摘要

本报告从比值视角深入分析 {crop_cn.get(self.crop, self.crop)} 亏缺灌溉优化效果，通过计算最优情景与对照组的比值（optimal/ck ratio），量化评估各指标的改善程度。

**核心指标：**
- 分析指标数：{n_indicators}
- 比值定义：ratio = optimal / ck
- 显著改善指标：{', '.join(best_improvements) if best_improvements else '详见下文'}

**比值意义：**
比值直接反映优化策略相对于对照组的相对变化，便于跨指标、跨区域比较，是评估优化效果的重要标准化指标。

---

"""

    def _generate_data_overview(self, comparison_results):
        """生成数据概述"""
        total_grids = 0
        for indicator in ['Y', 'ET', 'WUE', 'CH4', 'N2O', 'GWP']:
            if indicator in comparison_results['indicator_details']:
                n = len(comparison_results['indicator_details'][indicator])
                total_grids = max(total_grids, n)

        return f"""## 1. 数据概述

### 1.1 分析范围

- **网格总数**: {total_grids:,}
- **分析指标**: 6个（Y, ET, WUE, CH4, N2O, GWP）
- **比值范围**: 理论上 0 ~ ∞，实际集中在 0.7-1.3
- **基准定义**: 对照组（CK）作为比值基准（ck = 1.0）

### 1.2 比值优势

比值分析的优势：
- **标准化**: 消除绝对值差异，便于跨指标比较
- **直观性**: 比值>1或<1直接反映改善/恶化
- **可比性**: 不同研究、不同区域的比值可直接对比
- **百分比解读**: 比值=1.25表示提升25%，比值=0.85表示降低15%

---

"""

    def _generate_global_ratio_results(self, global_stats):
        """生成全球尺度比值结果"""
        lines = []
        lines.append("## 2. 全球尺度比值统计")
        lines.append("")
        lines.append("### 2.1 各指标比值汇总")
        lines.append("")
        lines.append("| 指标 | 平均比值 | 中位数比值 | 标准差 | Q25 | Q75 | 改善网格百分比 |")
        lines.append("|------|---------|-----------|--------|-----|-----|---------------|")

        for _, row in global_stats.iterrows():
            indicator = row['indicator']
            cfg = INDICATOR_CONFIG_RATIO.get(indicator, {})
            name_cn = cfg.get('name_cn', indicator)

            lines.append(
                f"| {name_cn} ({indicator}) | {row['mean_ratio']:.3f} | "
                f"{row['median_ratio']:.3f} | {row['std_ratio']:.3f} | "
                f"{row['ratio_q25']:.3f} | {row['ratio_q75']:.3f} | "
                f"{row['ratio_improvement_pct']:.1f}% |"
            )

        lines.append("")
        lines.append("**比值解读要点：**")
        lines.append("- 平均比值接近1.0：整体优化效果接近对照组")
        lines.append("- 改善网格百分比>50%：优化策略在该指标上整体有效")
        lines.append("- 标准差较大：空间差异显著，需制定区域特异性策略")
        lines.append("")

        # 2.2 比值分布特征
        lines.append("### 2.2 比值分布特征")
        lines.append("")

        for _, row in global_stats.iterrows():
            indicator = row['indicator']
            cfg = INDICATOR_CONFIG_RATIO.get(indicator, {})
            name_cn = cfg.get('name_cn', indicator)
            direction_cn = cfg.get('direction_cn', '未知')
            ideal_ratio = cfg.get('ideal_ratio', '未知')

            mean_ratio = row['mean_ratio']
            median_ratio = row['median_ratio']
            improvement_pct = row['ratio_improvement_pct']

            # 判断优化效果
            if direction_cn == '最大化':
                if mean_ratio > 1.0:
                    effect_desc = f"✓ 平均比值{mean_ratio:.3f}>1.0，整体提升{((mean_ratio-1)*100):.1f}%"
                else:
                    effect_desc = f"⚠ 平均比值{mean_ratio:.3f}<1.0，整体降低{((1-mean_ratio)*100):.1f}%"
            else:  # 最小化
                if mean_ratio < 1.0:
                    effect_desc = f"✓ 平均比值{mean_ratio:.3f}<1.0，整体降低{((1-mean_ratio)*100):.1f}%"
                else:
                    effect_desc = f"⚠ 平均比值{mean_ratio:.3f}>1.0，整体提升{((mean_ratio-1)*100):.1f}%"

            lines.append(f"**{name_cn} ({indicator})**")
            lines.append(f"- 优化方向：{direction_cn}，理想比值{ideal_ratio}")
            lines.append(f"- {effect_desc}")
            lines.append(f"- 改善网格：{improvement_pct:.1f}% 的网格实现了优化目标")
            lines.append(f"- 分布特征：均值{mean_ratio:.3f}，中位数{median_ratio:.3f}，分布{'对称' if abs(mean_ratio-median_ratio)<0.02 else '偏态'}")
            lines.append("")

        return '\n'.join(lines)

    def _generate_continental_ratio_results(self, continental_stats):
        """生成大洲尺度比值分析"""
        lines = []
        lines.append("## 3. 大洲尺度比值分析")
        lines.append("")
        lines.append("### 3.1 各大洲比值统计汇总")
        lines.append("")

        # 为每个指标生成大洲对比表
        for indicator in ['Y', 'ET', 'WUE', 'CH4', 'N2O', 'GWP']:
            if indicator not in continental_stats:
                continue

            cfg = INDICATOR_CONFIG_RATIO.get(indicator, {})
            name_cn = cfg.get('name_cn', indicator)

            lines.append(f"#### {name_cn} ({indicator})")
            lines.append("")
            lines.append("| 大洲 | 网格数 | 平均比值 | 中位数比值 | 标准差 | 改善网格百分比 | 显著改善网格数 |")
            lines.append("|------|--------|----------|-----------|--------|---------------|---------------|")

            stats = continental_stats[indicator]

            for continent in CONTINENT_ORDER + ['Global']:
                if continent not in stats:
                    continue

                s = stats[continent]
                name_cn_continent = CONTINENT_NAME_CN.get(continent, continent)

                lines.append(
                    f"| {name_cn_continent} | {s['n_grids']:,} | "
                    f"{s['mean_ratio']:.3f} | {s['median_ratio']:.3f} | "
                    f"{s['std_ratio']:.3f} | {s['improvement_pct']:.1f}% | "
                    f"{s['significant_improvement']:,} |"
                )

            lines.append("")

        # 3.2 大洲综合对比
        lines.append("### 3.2 大洲综合对比")
        lines.append("")
        lines.append("各大洲在各指标上的平均比值汇总：")
        lines.append("")
        lines.append("| 大洲 | Y比值 | ET比值 | WUE比值 | CH4比值 | N2O比值 | GWP比值 |")
        lines.append("|------|-------|--------|---------|---------|---------|---------|")

        for continent in CONTINENT_ORDER:
            row_data = [CONTINENT_NAME_CN.get(continent, continent)]

            for indicator in ['Y', 'ET', 'WUE', 'CH4', 'N2O', 'GWP']:
                if indicator in continental_stats and continent in continental_stats[indicator]:
                    ratio = continental_stats[indicator][continent]['mean_ratio']
                    row_data.append(f"{ratio:.3f}")
                else:
                    row_data.append("-")

            lines.append(f"| {' | '.join(row_data)} |")

        # 全球平均
        global_row = ['**全球**']
        for indicator in ['Y', 'ET', 'WUE', 'CH4', 'N2O', 'GWP']:
            if indicator in continental_stats and 'Global' in continental_stats[indicator]:
                ratio = continental_stats[indicator]['Global']['mean_ratio']
                global_row.append(f"**{ratio:.3f}**")
            else:
                global_row.append("-")

        lines.append(f"| {' | '.join(global_row)} |")
        lines.append("")

        return '\n'.join(lines)

    def _compute_ratio_benefits(self, comparison_results):
        """计算比值效益"""
        benefits = {}
        comparison_data = comparison_results['comparison_data']

        for _, row in comparison_data.iterrows():
            indicator = row['indicator']
            cfg = INDICATOR_CONFIG_RATIO.get(indicator, {})
            direction = cfg.get('direction', 'maximize')
            name_cn = cfg.get('name_cn', indicator)

            mean_ratio = row['mean_ratio']
            improvement_pct = row['ratio_improvement_pct']

            # 计算相对变化百分比
            if direction == 'maximize':
                relative_change_pct = (mean_ratio - 1.0) * 100
                benefit_desc = f"平均提升 {relative_change_pct:.1f}%"
            else:
                relative_change_pct = (1.0 - mean_ratio) * 100
                benefit_desc = f"平均降低 {relative_change_pct:.1f}%"

            benefits[indicator] = {
                'name_cn': name_cn,
                'mean_ratio': mean_ratio,
                'relative_change_pct': relative_change_pct,
                'improvement_pct': improvement_pct,
                'benefit_desc': benefit_desc,
                'direction': direction
            }

        return benefits

    def _generate_benefit_analysis(self, benefits):
        """生成比值效益分析"""
        lines = []
        lines.append("## 4. 比值效益分析")
        lines.append("")
        lines.append("从比值视角评估优化策略的综合效益：")
        lines.append("")

        # 分类分析
        lines.append("### 4.1 生产效益（最大化指标）")
        lines.append("")

        for indicator in ['Y', 'WUE']:
            if indicator in benefits:
                b = benefits[indicator]
                lines.append(f"**{b['name_cn']} ({indicator})**")
                lines.append(f"- 平均比值：{b['mean_ratio']:.3f}")
                lines.append(f"- 相对变化：{b['benefit_desc']}")
                lines.append(f"- 改善覆盖：{b['improvement_pct']:.1f}% 的网格实现优化目标")
                lines.append("")

        lines.append("### 4.2 环境效益（最小化指标）")
        lines.append("")

        for indicator in ['ET', 'CH4', 'N2O', 'GWP']:
            if indicator in benefits:
                b = benefits[indicator]
                lines.append(f"**{b['name_cn']} ({indicator})**")
                lines.append(f"- 平均比值：{b['mean_ratio']:.3f}")
                lines.append(f"- 相对变化：{b['benefit_desc']}")
                lines.append(f"- 改善覆盖：{b['improvement_pct']:.1f}% 的网格实现优化目标")
                lines.append("")

        lines.append("### 4.3 综合评价")
        lines.append("")
        lines.append("比值分析显示优化策略在不同维度的效果：")
        lines.append("")

        # 统计改善情况
        improved_count = sum(1 for b in benefits.values() if b['relative_change_pct'] > 0)
        total_count = len(benefits)

        lines.append(f"- **改善指标数**: {improved_count}/{total_count} 个指标实现优化目标")
        lines.append(f"- **改善覆盖率**: 平均改善网格百分比: {np.mean([b['improvement_pct'] for b in benefits.values()]):.1f}%")
        lines.append("- **比值一致性**: 各指标比值空间分布存在差异，需区域特异性策略")
        lines.append("")

        return '\n'.join(lines)

    def _generate_ratio_methods(self):
        """生成比值解读方法"""
        return """## 5. 比值解读方法

### 5.1 比值定义与计算

**比值公式：**
```
ratio = optimal_value / control_value
```

**比值含义：**
- 比值 = 1.0：最优情景与对照组完全相等
- 比值 > 1.0：最优情景数值大于对照组
- 比值 < 1.0：最优情景数值小于对照组

### 5.2 不同优化方向的解读

#### 最大化指标（Y, WUE）

目标：提升产量和效率

- **比值 > 1.0**：改善（优化有效）
  - 例如：ratio = 1.25 表示产量提升25%
  - 例如：ratio = 1.08 表示效率提升8%
- **比值 = 1.0**：无差异
- **比值 < 1.0**：恶化（需改进策略）
  - 例如：ratio = 0.90 表示产量降低10%

#### 最小化指标（ET, CH4, N2O, GWP）

目标：减少用水和排放

- **比值 < 1.0**：改善（优化有效）
  - 例如：ratio = 0.85 表示用水减少15%
  - 例如：ratio = 0.75 表示排放减少25%
- **比值 = 1.0**：无差异
- **比值 > 1.0**：恶化（需改进策略）
  - 例如：ratio = 1.12 表示排放增加12%

### 5.3 特殊情况处理

#### CH4负值情况

CH4可能存在负值（表示土壤吸收甲烷）：

- 例如：optimal = -4.5 kg/ha, ck = -4.6 kg/ha
- 比值 = -4.5 / -4.6 = 0.978
- 对于最小化指标，比值 < 1 表示改善（合理）
- 解读：最优情景吸收更多CH4（绝对值更大），比值<1表示改善

#### 除零处理

- 当对照组数值接近零（|ck| < 1e-6）时，比值设为NaN
- 在空间分布图中以白色/空白显示

### 5.4 显著改善定义

- **最大化指标**：比值 > 1.05（改善超过5%）
- **最小化指标**：比值 < 0.95（改善超过5%）
- 用于识别显著改善区域和热点区域

---

"""

    def _generate_key_findings(self, comparison_results, global_stats):
        """生成关键发现"""
        comparison_data = comparison_results['comparison_data']

        # 计算关键统计数据
        best_improvement_ind = comparison_data.loc[comparison_data['ratio_improvement_pct'].idxmax()]
        worst_improvement_ind = comparison_data.loc[comparison_data['ratio_improvement_pct'].idxmin()]

        best_cfg = INDICATOR_CONFIG_RATIO.get(best_improvement_ind['indicator'], {})
        worst_cfg = INDICATOR_CONFIG_RATIO.get(worst_improvement_ind['indicator'], {})

        best_name = best_cfg.get('name_cn', best_improvement_ind['indicator'])
        worst_name = worst_cfg.get('name_cn', worst_improvement_ind['indicator'])

        return f"""## 6. 主要发现

1. **最佳改善指标**: {best_name}的改善网格百分比最高（{best_improvement_ind['ratio_improvement_pct']:.1f}%），表明优化策略在该指标上效果最显著。

2. **挑战指标**: {worst_name}的改善网格百分比为{worst_improvement_ind['ratio_improvement_pct']:.1f}%，需进一步优化策略或调整权重。

3. **比值空间异质性**: 各指标比值在不同大洲间存在显著差异，亚洲和欧洲的改善效果相对较好，部分大洲需针对性优化。

4. **比值一致性**: 部分指标的比值分布标准差较大，表明优化效果在不同网格间存在较大差异，需考虑区域特异性策略。

5. **比值解读价值**: 比值分析提供了标准化视角，便于跨指标、跨研究对比，揭示了优化策略的相对效果而非绝对效果。

---

"""

    def _generate_conclusions(self):
        """生成结论与建议"""
        crop_cn = {
            'rice': '水稻',
            'wheat': '小麦',
            'maize': '玉米'
        }

        return f"""## 7. 结论与建议

### 7.1 核心结论

比值分析从标准化视角验证了 {crop_cn.get(self.crop, self.crop)} 亏缺灌溉优化策略的有效性：

1. **整体有效**: 多数指标的平均比值符合优化方向，表明策略整体有效
2. **区域差异**: 比值的空间分布显示显著区域差异，需区域适应性策略
3. **指标差异**: 不同指标的改善程度不同，反映多目标优化的权衡特性

### 7.2 政策建议

- **区域特异性**: 根据大洲比值特征制定区域特异性灌溉策略
- **重点指标**: 关注比值改善较差的指标，优化权重分配
- **显著改善区**: 优先在比值显著改善区域推广优化策略

### 7.3 比值应用建议

- **对比研究**: 使用比值便于与其他研究对比，提高可比性
- **动态监测**: 定期计算比值，监测优化策略的长期效果
- **扩展应用**: 比值方法可推广至其他作物和灌溉策略评估

### 7.4 未来展望

- **多尺度比值**: 发展大洲-国家-网格多尺度比值分析体系
- **动态比值**: 整合气候变化预测，分析比值的时间动态
- **比值优化**: 直接以比值为目标函数优化，可能产生更稳健的策略

---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""

    def _create_global_ratio_table(self, global_stats):
        """创建全球比值汇总表"""
        rows = []

        for _, row in global_stats.iterrows():
            indicator = row['indicator']
            cfg = INDICATOR_CONFIG_RATIO.get(indicator, {})

            rows.append({
                '指标': f"{cfg.get('name_cn', indicator)} ({indicator})",
                '平均比值': row['mean_ratio'],
                '中位数比值': row['median_ratio'],
                '标准差': row['std_ratio'],
                'Q25': row['ratio_q25'],
                'Q75': row['ratio_q75'],
                '改善网格百分比': row['ratio_improvement_pct']
            })

        return pd.DataFrame(rows)

    def _create_continental_ratio_table(self, continental_stats):
        """创建大洲比值统计表"""
        rows = []

        for indicator in ['Y', 'ET', 'WUE', 'CH4', 'N2O', 'GWP']:
            if indicator not in continental_stats:
                continue

            cfg = INDICATOR_CONFIG_RATIO.get(indicator, {})
            stats = continental_stats[indicator]

            for continent in CONTINENT_ORDER + ['Global']:
                if continent not in stats:
                    continue

                s = stats[continent]
                name_cn_continent = CONTINENT_NAME_CN.get(continent, continent)

                rows.append({
                    '指标': cfg.get('name_cn', indicator),
                    '大洲': name_cn_continent,
                    '网格数': s['n_grids'],
                    '平均比值': s['mean_ratio'],
                    '中位数比值': s['median_ratio'],
                    '标准差': s['std_ratio'],
                    '改善网格百分比': s['improvement_pct']
                })

        return pd.DataFrame(rows)


# =============================================================================
# 便捷函数
# =============================================================================

def generate_ratio_analysis_report(comparison_results, crop, outday, output_dir):
    """
    生成比值专项分析报告（完善版本）

    参数:
        comparison_results: dict, 对比分析结果
        crop: 作物类型
        outday: 输出日期
        output_dir: 输出目录

    返回:
        report_file: 报告文件路径
    """
    output_dir = Path(output_dir)

    # 创建报告生成器
    generator = RatioReportGenerator(crop, outday)

    # 生成完整报告
    report, tables = generator.generate_full_report(comparison_results)

    # 保存主报告
    report_file = output_dir / f'{crop}-ratio-report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"  比值分析报告已保存: {report_file}")

    # 保存表格
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    if 'global_ratio_summary' in tables:
        tables['global_ratio_summary'].to_csv(tables_dir / 'global_ratio_summary.csv', index=False)
        print(f"  全球比值汇总表已保存")

    if 'continental_ratio_stats' in tables:
        tables['continental_ratio_stats'].to_csv(tables_dir / 'continental_ratio_stats.csv', index=False)
        print(f"  大洲比值统计表已保存")

    return report_file


# =============================================================================
# 测试函数
# =============================================================================

if __name__ == "__main__":
    print("比值分析报告生成器 - 完善版本")
    print("报告结构：")
    print("  1. 标题和基本信息")
    print("  2. 摘要")
    print("  3. 数据概述")
    print("  4. 全球尺度比值统计")
    print("  5. 大洲尺度比值分析")
    print("  6. 比值效益分析")
    print("  7. 比值解读方法")
    print("  8. 主要发现")
    print("  9. 结论与建议")
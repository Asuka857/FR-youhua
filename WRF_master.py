"""
WRF Master Script - 最优亏缺灌溉分析系统主控脚本

功能：
1. 集成机器学习预测流程
2. 集成最优亏缺比例选择
3. 集成空间分析和可视化

创建时间：2025-03-20
最后修改：2025-03-20

使用方法：
    python WRF_master.py --crop rice --mode full
    python WRF_master.py --crop rice --mode response_surface
    python WRF_master.py --crop rice --mode optimize
    python WRF_master.py --crop rice --mode spatial
"""

import os
import sys
import time
import argparse
from datetime import datetime

# 导入原有模块
import mypackage.W_glob_tif
import mypackage.W_select_better
import mypackage.W_glob_mac

# 导入新模块
import mypackage.W_response_surface
import mypackage.W_optimization
import mypackage.W_spatial_analysis

# 导入配置
from mypackage.W_config import get_model_params, MODEL_PARAMS


# =============================================================================
# 配置参数
# =============================================================================

# 默认参数
DEFAULT_CONFIG = {
    'readday': '20251027',
    'outday': '20251028',
    'crop': 'wheat',
    'lowdata': 0.75,      # 响应面下限
    'updata': 0.90,       # 响应面上限
    'foot': 0.01,
    'opt_lowdata': 0.75,  # 优化下限（None表示等于响应面下限）
    'opt_updata': 0.90    # 优化上限（None表示等于响应面上限）
}

# 指标配置（从WRF_master.py原版提取）
INDICATOR_CONFIG = {
    'rice': [
        ('Y', 'Y', 8, 1, 2, 150, 147, 0.01, 1),
        ('WUE', 'WUE', 14, 1, 6, 150,47, 0.01, 1),
        ('N2O', 'N2O', 4, 1, 2, 80, 179, 0.01, 1),
        ('CH4', 'CH4', 6, 1, 4, 150, 158, 0.01, 1),
        ('ET', 'ET', 8, 1, 2, 60, 79, 0.01, 1)
    ],
    'wheat': [
        ('Y', 'Y', 12, 1, 2, 150, 41, 0.01, 1),
        ('WUE', 'WUE', 14, 1, 2, 120, 192, 0.01, 1),
        ('N2O', 'N2O', 4, 1, 8, 60, 244, 0.01, 1),
        ('CH4', 'CH4', 6, 5, 2, 60, 284, 0.01, 1),
        ('ET', 'ET', 14, 1, 2, 150, 168, 0.01, 1)
    ],
    'maize': [
        ('Y', 'Y', 14, 1, 2, 150, 206, 0.01, 1),
        ('WUE', 'WUE', 14, 1, 2, 80, 291, 0.01, 1),
        ('N2O', 'N2O', 4, 5, 2, 150, 204, 0.01, 1),
        ('CH4', 'CH4', 14, 1, 2, 120, 1, 0.01, 1),
        ('ET', 'ET', 14, 1, 4, 60, 88, 0.01, 1)
    ]
}


# =============================================================================
# 主流程函数
# =============================================================================

def run_original_pipeline(config):
    """
    运行原有的机器学习预测流程

    参数：
        config: 配置字典
    """
    print(f"\n{'='*70}")
    print(f"  运行原有机器学习预测流程")
    print(f"{'='*70}\n")

    crop = config['crop']
    readday = config['readday']
    outday = config['outday']
    lowdata = config['lowdata']
    updata = config['updata']

    name_pred = INDICATOR_CONFIG.get(crop, INDICATOR_CONFIG['rice'])

    # 1. 指标预测流水线
    print(f"\n[1/4] 指标预测流水线...")
    for pred in name_pred:
        file, yname, crop_n, max_depth, min_samples_leaf, min_samples_split, n_random, i, foot, *rest = pred
        mypackage.W_glob_mac.global_machine_pipeline(
            file, yname, crop_n, max_depth, min_samples_leaf, min_samples_split,
            n_random, i, foot, readday, outday, lowdata, updata
        )

    # 2. GHG计算
    print(f"\n[2/4] GHG计算...")
    mypackage.W_glob_mac.GHG_pipeline(crop, outday, lowdata, updata, 0.01)
    mypackage.W_glob_mac.global_machine_vs_GHG_1(crop, outday)

    # 3. Ratio计算
    print(f"\n[3/4] Ratio计算...")
    for file, *rest in name_pred:
        if file == "CH4":
            mypackage.W_glob_mac.global_machine_vs_CH4_1(file, '0.65', crop, 'Ypred', outday)
            mypackage.W_glob_mac.global_machine_vs_CH4_1(file, '0.80', crop, 'Ypred', outday)
        else:
            mypackage.W_glob_mac.global_machine_vs_1(file, '0.65', crop, 'Ypred', outday)
            mypackage.W_glob_mac.global_machine_vs_1(file, '0.80', crop, 'Ypred', outday)

    # 4. TIF生成
    print(f"\n[4/4] TIF生成...")
    for file, *rest in name_pred:
        mypackage.W_glob_tif.csv_to_tif_r(f'{file}_0.65', 'ratio', '0.65', crop, 2, outday)
        mypackage.W_glob_tif.csv_to_tif_r(f'{file}_0.80', 'ratio', '0.80', crop, 2, outday)

    mypackage.W_glob_tif.csv_to_tif_r('GHG_0.65', 'ratio', '0.65', crop, 2, outday)
    mypackage.W_glob_tif.csv_to_tif_r('GHG_0.80', 'ratio', '0.80', crop, 2, outday)

    for file, *rest in name_pred:
        mypackage.W_glob_tif.csv_to_tif(f'{file}_ck', 'Ypred', crop, 1, outday)

    mypackage.W_glob_tif.csv_to_tif('GHG_ck', 'Ypred', crop, 1, outday)

    print(f"\n  原有流程完成!")


def run_response_surface_pipeline(config):
    """
    第2步：构建响应面

    参数：
        config: 配置字典
    """
    print(f"\n{'='*70}")
    print(f"  第2步：构建响应面")
    print(f"{'='*70}\n")

    crop = config['crop']
    readday = config['readday']
    outday = config['outday']
    lowdata = config['lowdata']
    updata = config['updata']
    foot = config['foot']

    # 获取模型参数
    model_params = get_model_params(crop)

    # 指标列表
    indicators = ['Y', 'ET', 'WUE', 'CH4', 'N2O', 'GWP']

    # 构建响应面
    surface = mypackage.W_response_surface.build_response_surface(
        crop=crop,
        readday=readday,
        outday=outday,
        indicators=indicators,
        lowdata=lowdata,
        updata=updata,
        foot=foot,
        model_params=model_params,
        save_intermediate=True,
        force_rebuild=config.get('force_rebuild', False)
    )

    return surface


def run_optimization_pipeline(config):
    """
    第3步：选择最优亏缺比例

    参数：
        config: 配置字典
    """
    print(f"\n{'='*70}")
    print(f"  第3步：选择最优亏缺比例")
    print(f"{'='*70}\n")

    crop = config['crop']
    outday = config['outday']
    foot = config['foot']

    # 优化范围（可与响应面范围不同）
    opt_lowdata = config['opt_lowdata']
    opt_updata = config['opt_updata']

    print(f"  优化范围: [{opt_lowdata:.2f}, {opt_updata:.2f}]")

    # 指标列表（优化只用Y、WUE、GWP三个指标）
    indicators = ['Y', 'WUE', 'GWP']

    # 权重配置（等权重）
    weights = None  # 使用默认等权重

    # 运行优化（使用优化范围参数）
    optimal_results = mypackage.W_optimization.optimize_deficit_ratio(
        crop=crop,
        outday=outday,
        lowdata=opt_lowdata,
        updata=opt_updata,
        foot=foot,
        indicators=indicators,
        weights=weights,
        validate_results=True,
        compare_with_ck=True
    )

    return optimal_results


def run_spatial_analysis_pipeline(config):
    """
    第4步：空间分析和可视化

    参数：
        config: 配置字典
    """
    print(f"\n{'='*70}")
    print(f"  第4步：空间分析和可视化")
    print(f"{'='*70}\n")

    crop = config['crop']
    outday = config['outday']

    # 运行空间分析
    analysis_results = mypackage.W_spatial_analysis.generate_spatial_analysis(
        crop=crop,
        outday=outday,
        generate_tif=True,
        generate_maps=True,
        compare_with_ck=True
    )

    return analysis_results


# =============================================================================
# 完整流程
# =============================================================================

def run_full_pipeline(config):
    """
    运行完整的分析流程

    参数：
        config: 配置字典
    """
    print(f"\n{'#'*70}")
    print(f"# 最优亏缺灌溉分析系统 - 完整流程")
    print(f"{'#'*70}")
    print(f"作物: {config['crop']}")
    print(f"输入日期: {config['readday']}")
    print(f"输出日期: {config['outday']}")
    print(f"Ratio区间: [{config['lowdata']:.2f}, {config['updata']:.2f}]")
    print(f"{'#'*70}\n")

    start_time = time.time()

    # 第1步：原有预测流程（可选）
    if config.get('run_original', False):
        run_original_pipeline(config)

    # 第2步：构建响应面
    surface = run_response_surface_pipeline(config)

    # 第3步：选择最优亏缺比例
    optimal_results = run_optimization_pipeline(config)

    # 第4步：空间分析和可视化
    analysis_results = run_spatial_analysis_pipeline(config)

    # 总结
    elapsed_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"  分析完成!")
    print(f"{'='*70}")
    print(f"总耗时: {elapsed_time/60:.1f} 分钟")
    print(f"输出目录: F:/WaterML/{config['crop']}/{config['outday']}/")
    print(f"{'='*70}\n")

    return {
        'surface': surface,
        'optimal_results': optimal_results,
        'analysis_results': analysis_results
    }


# =============================================================================
# 命令行接口
# =============================================================================

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='最优亏缺灌溉分析系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 完整流程（响应面0.60-0.90，优化0.60-0.90）
    python WRF_master.py --crop rice --mode full

    # 响应面0.60-0.90，优化仅在0.75-0.90区间
    python WRF_master.py --crop rice --mode optimize --opt-lowdata 0.75 --opt-updata 0.90

    # 自定义响应面范围
    python WRF_master.py --crop rice --mode response_surface --lowdata 0.50 --updata 1.00

    # 运行其他作物
    python WRF_master.py --crop wheat --mode full
    python WRF_master.py --crop maize --mode full
        """
    )

    parser.add_argument(
        '--crop',
        type=str,
        default='rice',
        choices=['rice', 'wheat', 'maize'],
        help='作物类型 (default: rice)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['full', 'original', 'response_surface', 'optimize', 'spatial'],
        help='运行模式 (default: full)'
    )

    parser.add_argument(
        '--readday',
        type=str,
        default=DEFAULT_CONFIG['readday'],
        help=f'输入数据日期标识 (default: {DEFAULT_CONFIG["readday"]})'
    )

    parser.add_argument(
        '--outday',
        type=str,
        default=DEFAULT_CONFIG['outday'],
        help=f'输出数据日期标识 (default: {DEFAULT_CONFIG["outday"]})'
    )

    parser.add_argument(
        '--lowdata',
        type=float,
        default=DEFAULT_CONFIG['lowdata'],
        help=f'ratio下限 (default: {DEFAULT_CONFIG["lowdata"]})'
    )

    parser.add_argument(
        '--updata',
        type=float,
        default=DEFAULT_CONFIG['updata'],
        help=f'ratio上限 (default: {DEFAULT_CONFIG["updata"]})'
    )

    parser.add_argument(
        '--foot',
        type=float,
        default=DEFAULT_CONFIG['foot'],
        help=f'ratio步长 (default: {DEFAULT_CONFIG["foot"]})'
    )

    parser.add_argument(
        '--opt-lowdata',
        type=float,
        default=None,
        help=f'优化ratio下限（默认等于响应面下限）'
    )

    parser.add_argument(
        '--opt-updata',
        type=float,
        default=None,
        help=f'优化ratio上限（默认等于响应面上限）'
    )

    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='强制重新构建响应面'
    )

    parser.add_argument(
        '--run-original',
        action='store_true',
        help='同时运行原有预测流程'
    )

    return parser.parse_args()


def main():
    """
    主函数

    使用方式：
    1. 任务列表模式：编辑 TASK_LIST，直接运行 python WRF_master.py
    2. 命令行模式：python WRF_master.py --crop rice --mode full
    """
    # 检查是否有预设任务
    if TASK_LIST:
        # 任务列表模式
        run_task_list()
        return

    # 命令行模式
    args = parse_arguments()

    # 构建配置
    config = {
        'crop': args.crop,
        'readday': args.readday,
        'outday': args.outday,
        'lowdata': args.lowdata,
        'updata': args.updata,
        'foot': args.foot,
        'opt_lowdata': args.opt_lowdata if args.opt_lowdata is not None else args.lowdata,
        'opt_updata': args.opt_updata if args.opt_updata is not None else args.updata,
        'force_rebuild': args.force_rebuild,
        'run_original': args.run_original
    }

    # 打印配置
    print(f"\n{'='*70}")
    print(f"  最优亏缺灌溉分析系统")
    print(f"{'='*70}")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"运行模式: {args.mode}")
    print(f"配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'='*70}\n")

    # 根据模式运行
    try:
        if args.mode == 'full':
            run_full_pipeline(config)
        elif args.mode == 'original':
            run_original_pipeline(config)
        elif args.mode == 'response_surface':
            run_response_surface_pipeline(config)
        elif args.mode == 'optimize':
            run_optimization_pipeline(config)
        elif args.mode == 'spatial':
            run_spatial_analysis_pipeline(config)
        else:
            raise ValueError(f"未知的运行模式: {args.mode}")

        print(f"\n{'='*70}")
        print(f"  任务完成!")
        print(f"{'='*70}")
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"  错误!")
        print(f"{'='*70}")
        print(f"错误信息: {e}")
        print(f"{'='*70}\n")
        raise


# =============================================================================
# 预设任务函数
# =============================================================================

def run_rice_analysis():
    """运行水稻分析"""
    config = DEFAULT_CONFIG.copy()
    config['crop'] = 'rice'
    run_full_pipeline(config)


def run_wheat_analysis():
    """运行小麦分析"""
    config = DEFAULT_CONFIG.copy()
    config['crop'] = 'wheat'
    run_full_pipeline(config)


def run_maize_analysis():
    """运行玉米分析"""
    config = DEFAULT_CONFIG.copy()
    config['crop'] = 'maize'
    run_full_pipeline(config)


# =============================================================================
# 任务配置 - 在这里修改要运行的任务
# =============================================================================

# 任务列表：每个任务是一个字典，定义作物和参数
# 将此列表留空 [] 则使用命令行参数
TASK_LIST = [
    # ========== 任务示例 ==========

    #示例1：Rice - 完整流程
    # {
    #     'crop': 'maize',
    #     'mode': 'full',
    #     'lowdata': 0.60,        # 响应面范围
    #     'updata': 0.86,
    #     'opt_lowdata': 0.60,    # 优化范围
    #     'opt_updata': 0.85,
    #     'force_rebuild': True,  # 强制重新构建响应面
    # },
    {
        'crop': 'rice',
        'mode': 'full',
        'lowdata': 0.75,  # 响应面范围
        'updata': 0.90,
        'opt_lowdata': 0.75,  # 优化范围
        'opt_updata': 0.90,
        'force_rebuild': True,  # 强制重新构建响应面
    },
    # ========== 添加更多任务 ==========
    # 复制上面的模板，修改参数即可
]


def run_task_list():
    """
    运行任务列表中的所有任务
    """
    if not TASK_LIST:
        print("任务列表为空，请使用命令行参数或编辑 TASK_LIST")
        return

    total_tasks = len(TASK_LIST)
    print(f"\n{'#'*70}")
    print(f"#  批量任务模式")
    print(f"#  共 {total_tasks} 个任务")
    print(f"{'#'*70}\n")

    start_time = time.time()
    results = []

    for i, task in enumerate(TASK_LIST, 1):
        print(f"\n{'='*70}")
        print(f"  任务 {i}/{total_tasks}: {task.get('crop', 'unknown')} - {task.get('mode', 'full')}")
        print(f"{'='*70}")

        # 构建完整配置
        config = DEFAULT_CONFIG.copy()
        config.update(task)

        # 设置优化范围默认值
        if 'opt_lowdata' not in config or config['opt_lowdata'] is None:
            config['opt_lowdata'] = config['lowdata']
        if 'opt_updata' not in config or config['opt_updata'] is None:
            config['opt_updata'] = config['updata']

        # 打印任务配置
        print(f"配置参数:")
        print(f"  作物: {config['crop']}")
        print(f"  模式: {config.get('mode', 'full')}")
        print(f"  响应面范围: [{config['lowdata']:.2f}, {config['updata']:.2f}]")
        print(f"  优化范围: [{config['opt_lowdata']:.2f}, {config['opt_updata']:.2f}]")

        # 运行任务
        try:
            mode = config.get('mode', 'full')
            if mode == 'full':
                run_full_pipeline(config)
            elif mode == 'response_surface':
                run_response_surface_pipeline(config)
            elif mode == 'optimize':
                run_optimization_pipeline(config)
            elif mode == 'spatial':
                run_spatial_analysis_pipeline(config)

            results.append({'task': i, 'crop': config['crop'], 'status': 'success'})
            print(f"\n  [成功] 任务 {i} 完成")

        except Exception as e:
            results.append({'task': i, 'crop': config['crop'], 'status': 'failed', 'error': str(e)})
            print(f"\n  [失败] 任务 {i}: {e}")

    # 打印汇总
    elapsed_time = time.time() - start_time
    print(f"\n{'#'*70}")
    print(f"#  批量任务完成")
    print(f"{'#'*70}")
    print(f"总耗时: {elapsed_time/60:.1f} 分钟")
    print(f"\n任务结果汇总:")
    for r in results:
        status = "成功" if r['status'] == 'success' else f"失败: {r.get('error', 'unknown')}"
        print(f"  任务{r['task']} ({r['crop']}): {status}")
    print(f"{'#'*70}\n")


def run_all_crops():
    """运行所有作物分析"""
    for crop in ['rice', 'wheat', 'maize']:
        config = DEFAULT_CONFIG.copy()
        config['crop'] = crop
        run_full_pipeline(config)


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    main()
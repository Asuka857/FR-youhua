"""
配置文件：模型参数和其他设置

创建时间：2025-03-20
最后修改：2025-03-20

改进内容：
1. 集成路径管理器
2. 集中化管理所有配置
"""

# 导入路径管理器
from .W_path_manager import path_manager, get_path_manager, set_base_dir

# =============================================================================
# 作物模型参数配置
# =============================================================================

MODEL_PARAMS = {
    'rice': {
        'Y': {
            'max_depth': 12,
            'min_samples_leaf': 1,
            'min_samples_split': 4,
            'n_random': 150,
            'random_state': 199
        },
        'WUE': {
            'max_depth': 8,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_random': 60,
            'random_state': 79
        },
        'N2O': {
            'max_depth': 4,
            'min_samples_leaf': 1,
            'min_samples_split': 4,
            'n_random': 150,
            'random_state': 53
        },
        'CH4': {
            'max_depth': 8,
            'min_samples_leaf': 1,
            'min_samples_split': 4,
            'n_random': 60,
            'random_state': 165
        },
        'ET': {
            'max_depth': 4,
            'min_samples_leaf': 1,
            'min_samples_split': 4,
            'n_random': 150,
            'random_state': 53
        }
    },
    'wheat': {
        'Y': {
            'max_depth': 12,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_random': 150,
            'random_state': 41
        },
        'WUE': {
            'max_depth': 14,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_random': 150,
            'random_state': 168
        },
        'N2O': {
            'max_depth': 14,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_random': 120,
            'random_state': 192
        },
        'CH4': {
            'max_depth': 6,
            'min_samples_leaf': 5,
            'min_samples_split': 2,
            'n_random': 60,
            'random_state': 284
        },
        'ET': {
            'max_depth': 4,
            'min_samples_leaf': 1,
            'min_samples_split': 8,
            'n_random': 60,
            'random_state': 244
        }
    },
    'maize': {
        'Y': {
            'max_depth': 14,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_random': 150,
            'random_state': 206
        },
        'WUE': {
            'max_depth': 14,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_random': 60,
            'random_state': 88
        },
        'N2O': {
            'max_depth': 14,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_random': 80,
            'random_state': 291
        },
        'CH4': {
            'max_depth': 14,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_random': 120,
            'random_state': 1
        },
        'ET': {
            'max_depth': 4,
            'min_samples_leaf': 5,
            'min_samples_split': 2,
            'n_random': 150,
            'random_state': 204
        }
    }
}


# =============================================================================
# 指标优化方向配置
# =============================================================================

INDICATOR_DIRECTION = {
    'Y': 'maximize',    # Yield 越大越好
    'ET': 'minimize',   # ET 越小越好
    'WUE': 'maximize',  # WUE 越大越好
    'CH4': 'minimize',  # CH4 越小越好
    'N2O': 'minimize',  # N2O 越小越好
    'GWP': 'minimize'   # GWP 越小越好
}


# =============================================================================
# GWP计算系数 (Global Warming Potential)
# =============================================================================

GWP_CH4_FACTOR = 27   # CH4 -> CO2-eq
GWP_N2O_FACTOR = 273  # N2O -> CO2-eq


# =============================================================================
# 可视化参数配置
# =============================================================================

VISUALIZATION_PARAMS = {
    'rice': {
        'optimal_ratio': {'vmin': 0.70, 'vmax': 0.90, 'cmap': 'RdYlGn'},
        'Y': {'vmin': 0, 'vmax': 12000, 'cmap': 'YlOrRd'},
        'ET': {'vmin': 600, 'vmax': 1500, 'cmap': 'Blues'},
        'WUE': {'vmin': 0, 'vmax': 25, 'cmap': 'Greens'},
        'CH4': {'vmin': 50, 'vmax': 500, 'cmap': 'Oranges'},
        'N2O': {'vmin': 0, 'vmax': 8, 'cmap': 'Purples'},
        'GWP': {'vmin': 0, 'vmax': 15000, 'cmap': 'Reds'}
    },
    'wheat': {
        'optimal_ratio': {'vmin': 0.70, 'vmax': 0.90, 'cmap': 'RdYlGn'},
        'Y': {'vmin': 0, 'vmax': 12000, 'cmap': 'YlOrRd'},
        'ET': {'vmin': 200, 'vmax': 600, 'cmap': 'Blues'},
        'WUE': {'vmin': 0, 'vmax': 25, 'cmap': 'Greens'},
        'CH4': {'vmin': -10, 'vmax': 0, 'cmap': 'Oranges'},
        'N2O': {'vmin': 0, 'vmax': 6, 'cmap': 'Purples'},
        'GWP': {'vmin': 0, 'vmax': 1500, 'cmap': 'Reds'}
    },
    'maize': {
        'optimal_ratio': {'vmin': 0.70, 'vmax': 0.90, 'cmap': 'RdYlGn'},
        'Y': {'vmin': 0, 'vmax': 18000, 'cmap': 'YlOrRd'},
        'ET': {'vmin': 200, 'vmax': 800, 'cmap': 'Blues'},
        'WUE': {'vmin': 0, 'vmax': 25, 'cmap': 'Greens'},
        'CH4': {'vmin': -10, 'vmax': 0, 'cmap': 'Oranges'},
        'N2O': {'vmin': 0, 'vmax': 6, 'cmap': 'Purples'},
        'GWP': {'vmin': 0, 'vmax': 1500, 'cmap': 'Reds'}
    }
}


# =============================================================================
# 文件路径配置
# =============================================================================

DATA_PATHS = {
    'base_dir': 'F:/WaterML',
    'train_data_pattern': '{base_dir}/{crop}/{readday}/D{indicator}.csv',
    'global_data_pattern': '{base_dir}/{crop}/{readday}/x_data_point_with_TotalN.{crop}.csv',
    'output_pattern': '{base_dir}/{crop}/{outday}'
}


# =============================================================================
# 辅助函数
# =============================================================================

def get_model_params(crop):
    """
    获取指定作物的模型参数

    参数：
        crop: 作物类型

    返回：
        params: 模型参数字典
    """
    return MODEL_PARAMS.get(crop, MODEL_PARAMS['rice'])


def get_visualization_params(crop, indicator):
    """
    获取指定作物和指标的可视化参数

    参数：
        crop: 作物类型
        indicator: 指标名称

    返回：
        params: 可视化参数字典
    """
    crop_params = VISUALIZATION_PARAMS.get(crop, VISUALIZATION_PARAMS['rice'])
    return crop_params.get(indicator, {})


def get_data_path(crop, readday, outday, data_type='train', indicator=None):
    """
    获取数据文件路径

    参数：
        crop: 作物类型
        readday: 输入日期
        outday: 输出日期
        data_type: 数据类型 ('train', 'global', 'output')
        indicator: 指标名称 (仅用于train类型)

    返回：
        path: 文件路径
    """
    base = DATA_PATHS['base_dir']

    if data_type == 'train':
        return DATA_PATHS['train_data_pattern'].format(
            base_dir=base, crop=crop, readday=readday, indicator=indicator
        )
    elif data_type == 'global':
        return DATA_PATHS['global_data_pattern'].format(
            base_dir=base, crop=crop, readday=readday
        )
    elif data_type == 'output':
        return DATA_PATHS['output_pattern'].format(
            base_dir=base, crop=crop, outday=outday
        )
    else:
        raise ValueError(f"未知的data_type: {data_type}")
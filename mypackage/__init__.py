"""
mypackage - 最优亏缺灌溉分析系统

包含模块：
- W_glob_mac: 机器学习核心模块
- W_glob_tif: 空间转换模块
- W_select_better: 数据选择模块
- picture: 可视化模块
- W_config: 配置模块
- W_response_surface: 响应面构建模块
- W_optimization: 最优比例选择模块
- W_spatial_analysis: 空间分析模块
- W_regional_analysis: 区域分析模块
- W_soil_manager: 土壤特征管理模块
- W_comparison: 对比分析模块（新增）
"""

from .W_glob_mac import (
    data_processed_one,
    data_processed_two,
    machine_RF,
    global_data_processed_one,
    global_machine_pipeline,
    GHG_pipeline,
    read_csv_cached,
    DTYPE_FLOAT
)

from .W_glob_tif import (
    csv_to_tif,
    csv_to_tif_r
)

from .W_config import (
    MODEL_PARAMS,
    INDICATOR_DIRECTION,
    VISUALIZATION_PARAMS,
    get_model_params,
    get_visualization_params,
    get_data_path
)

from .W_response_surface import (
    build_response_surface,
    load_response_surface,
    query_response_surface,
    get_response_surface_status,
    build_response_surface_batch
)

from .W_optimization import (
    optimize_deficit_ratio,
    validate_optimal_results,
    analyze_optimal_results,
    optimize_deficit_ratio_batch,
    optimize_with_multiple_weights
)

from .W_spatial_analysis import (
    generate_spatial_analysis,
    generate_statistics_summary
)

from .W_regional_analysis import (
    calculate_regional_statistics,
    detailed_regional_analysis,
    compare_regions,
    analyze_by_country_simple
)

from .W_soil_manager import (
    SoilFeatureManager,
    get_all_features,
    get_texture_features,
    get_base_features,
    align_features,
    encode_training_soil_texture,
    encode_global_soil_texture,
    ALL_FEATURES,
    ALL_TEXTURE_FEATURES,
    SOIL_TEXTURES
)

from .W_comparison import (
    calculate_gwp_ck,
    calculate_change_pct,
    compare_optimal_vs_ck,
    generate_comparison_analysis,
    COMPARISON_INDICATORS,
    INDICATOR_INFO,
    GWP_CH4_FACTOR,
    GWP_N2O_FACTOR
)

__version__ = '1.0.0'
__author__ = 'Water Management Research Team'

__all__ = [
    # 数据处理
    'data_processed_one',
    'data_processed_two',
    'machine_RF',
    'global_data_processed_one',
    'global_machine_pipeline',
    'GHG_pipeline',
    'read_csv_cached',
    'DTYPE_FLOAT',

    # 空间转换
    'csv_to_tif',
    'csv_to_tif_r',

    # 配置
    'MODEL_PARAMS',
    'INDICATOR_DIRECTION',
    'VISUALIZATION_PARAMS',
    'get_model_params',
    'get_visualization_params',
    'get_data_path',

    # 响应面
    'build_response_surface',
    'load_response_surface',
    'query_response_surface',
    'get_response_surface_status',
    'build_response_surface_batch',

    # 优化
    'optimize_deficit_ratio',
    'validate_optimal_results',
    'compare_optimal_vs_ck',
    'analyze_optimal_results',
    'optimize_deficit_ratio_batch',
    'optimize_with_multiple_weights',

    # 空间分析
    'generate_spatial_analysis',
    'generate_statistics_summary',

    # 区域分析
    'calculate_regional_statistics',
    'detailed_regional_analysis',
    'compare_regions',
    'analyze_by_country_simple',

    # 土壤特征管理
    'SoilFeatureManager',
    'get_all_features',
    'get_texture_features',
    'get_base_features',
    'align_features',
    'encode_training_soil_texture',
    'encode_global_soil_texture',
    'ALL_FEATURES',
    'ALL_TEXTURE_FEATURES',
    'SOIL_TEXTURES',

    # 对比分析
    'calculate_gwp_ck',
    'calculate_change_pct',
    'generate_comparison_analysis',
    'COMPARISON_INDICATORS',
    'INDICATOR_INFO',
    'GWP_CH4_FACTOR',
    'GWP_N2O_FACTOR'
]
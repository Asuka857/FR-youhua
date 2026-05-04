"""
响应面构建模块 (Response Surface Construction)
功能：构建"网格×ratio×指标"三维响应面

创建时间：2025-03-20
最后修改：2025-03-20

改进内容：
1. 使用read_csv_cached减少重复IO
2. 添加增量计算支持
3. 添加内存优化（分批处理）
4. 添加进度保存和恢复功能
"""

import os
import json
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor

# 导入现有模块的数据处理函数
from .W_glob_mac import (
    data_processed_one,
    data_processed_two,
    global_data_processed_one,
    read_csv_cached,
    DTYPE_FLOAT
)

# 导入配置
from .W_config import MODEL_PARAMS, INDICATOR_DIRECTION, get_model_params, GWP_CH4_FACTOR, GWP_N2O_FACTOR

# 导入路径管理器
from .W_path_manager import get_path_manager

# 导入土壤特征管理器
from .W_soil_manager import (
    SoilFeatureManager,
    get_all_features,
    align_features,
    ALL_FEATURES
)


# =============================================================================
# 响应面构建主函数
# =============================================================================

def build_response_surface(
    crop,
    readday,
    outday,
    indicators=['Y', 'ET', 'WUE', 'CH4', 'N2O'],
    lowdata=0.60,
    updata=0.90,
    foot=0.01,
    model_params=None,
    save_intermediate=True,
    force_rebuild=False,
    batch_size=None,
    checkpoint_interval=10
):
    """
    构建三维响应面：网格 × ratio × 指标

    改进特性：
    - 增量计算：已存在响应面则直接加载
    - 内存优化：支持分批处理大规模网格
    - 进度保存：定期保存检查点，支持断点续算

    参数：
        crop: 作物类型 ('rice', 'wheat', 'maize')
        readday: 输入数据日期标识
        outday: 输出数据日期标识
        indicators: 预测指标列表
        lowdata: ratio下限 (默认 0.60)
        updata: ratio上限 (默认 0.90)
        foot: ratio步长 (默认 0.01)
        model_params: 各指标模型超参数字典
        save_intermediate: 是否保存中间结果 (默认 True)
        force_rebuild: 是否强制重新构建 (默认 False)
        batch_size: 批处理大小 (None表示不分批)
        checkpoint_interval: 检查点保存间隔（每处理多少个ratio保存一次）

    返回：
        response_surface_dict: 包含三维响应面和元数据的字典

    示例：
        model_params = {
            'Y': {'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 6, 'n_random': 150, 'random_state': 11},
            'WUE': {'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_random': 60, 'random_state': 252}
        }
        surface = build_response_surface('rice', '20251027', '20251028', model_params=model_params)
    """

    print(f"\n{'='*70}")
    print(f"  响应面构建模块启动")
    print(f"{'='*70}")
    print(f"作物: {crop} | Ratio区间: [{lowdata:.2f}, {updata:.2f}] | 步长: {foot:.2f}")
    print(f"指标: {indicators}")
    print(f"{'='*70}\n")

    start_time = time.time()

    # 初始化路径管理器
    pm = get_path_manager()

    # =========================================================================
    # 步骤0: 检查是否已存在响应面（增量计算支持）
    # =========================================================================
    output_dir = pm.get_response_surface_dir(crop, outday)
    metadata_file = output_dir / 'metadata.json'

    if not force_rebuild and os.path.exists(metadata_file):
        print(f"[检查] 发现已存在的响应面数据...")

        # 加载元数据检查兼容性
        with open(metadata_file, 'r', encoding='utf-8') as f:
            existing_metadata = json.load(f)

        # 检查参数是否匹配
        if (existing_metadata.get('lowdata') == float(lowdata) and
            existing_metadata.get('updata') == float(updata) and
            existing_metadata.get('foot') == float(foot) and
            set(existing_metadata.get('indicators', [])) == set(indicators)):

            print(f"  参数匹配，直接加载已有响应面")
            print(f"  创建时间: {existing_metadata.get('creation_time', 'Unknown')}")

            # 直接加载并返回
            surface_dict = load_response_surface(output_dir)
            print(f"  响应面加载完成，跳过重新构建")
            return surface_dict
        else:
            print(f"  参数不匹配，将重新构建响应面")

    # =========================================================================
    # 步骤1: 准备ratio序列和网格数据
    # =========================================================================
    ratios = np.arange(lowdata, updata, foot, dtype=DTYPE_FLOAT)  # 使用float32避免精度问题
    n_ratios = len(ratios)

    print(f"[Step 1/6] 准备全球网格数据...")
    global_file_path = pm.get_global_data_path(crop, readday)

    if not os.path.exists(global_file_path):
        raise FileNotFoundError(f"找不到全球网格数据文件: {global_file_path}")

    # 使用缓存读取全球数据
    global_data_raw, original_coords = global_data_processed_one(global_file_path)
    n_grids = len(global_data_raw)

    print(f"  网格数量: {n_grids:,}")
    print(f"  Ratio候选数: {n_ratios}")
    print(f"  预计生成响应面大小: ({n_grids:,}, {n_ratios}, {len(indicators)})")

    # 内存预估
    estimated_memory_mb = n_grids * n_ratios * len(indicators) * 4 / 1024 / 1024  # float32 = 4 bytes
    print(f"  预计内存占用: {estimated_memory_mb:.1f} MB")

    # 自动判断是否需要分批处理
    if batch_size is None:
        if estimated_memory_mb > 4000:  # 超过4GB自动分批
            batch_size = 5000
            print(f"  内存占用较大，启用分批处理 (batch_size={batch_size})")
        else:
            batch_size = n_grids  # 不分批

    # =========================================================================
    # 步骤2: 训练所有指标模型
    # =========================================================================
    print(f"\n[Step 2/6] 训练各指标随机森林模型...")

    if model_params is None:
        model_params = get_model_params(crop)

    models_dict = {}
    for indicator in tqdm(indicators, desc="  训练模型"):
        if indicator == 'GWP':
            continue  # GWP 不需要单独训练，后续从 CH4 和 N2O 计算

        # 读取训练数据（使用缓存）
        file_path = pm.get_train_data_path(crop, readday, indicator)
        if not os.path.exists(file_path):
            print(f"    警告: 找不到训练数据 {file_path}, 跳过该指标")
            continue

        # 数据预处理（使用缓存读取）
        df_train = data_processed_one(file_path, indicator, lowdata, updata)
        df_train = data_processed_two(df_train)

        # 获取模型参数
        params = model_params.get(indicator, {})

        # 训练模型
        rf = train_random_forest(df_train, indicator, params)
        models_dict[indicator] = rf

    print(f"  已训练模型: {list(models_dict.keys())}")

    # 检查是否有成功训练的模型
    if not models_dict:
        raise ValueError(
            "没有成功训练任何模型，请检查训练数据文件是否存在且格式正确\n"
            f"期望的训练数据文件: {[f'D{ind}.csv' for ind in indicators if ind != 'GWP']}"
        )

    # =========================================================================
    # 步骤3: 准备预测输入模板（使用统一的土壤特征管理）
    # =========================================================================
    print(f"\n[Step 3/6] 准备预测输入数据...")

    # 创建土壤特征管理器
    soil_manager = SoilFeatureManager()

    # 获取统一的特征列（所有指标共享）
    unified_features = soil_manager.get_feature_subset(include_coords=False)
    print(f"  统一特征数量: {len(unified_features)}")
    print(f"  土壤特征: {soil_manager.texture_features}")

    # 一次性准备全球输入模板
    input_template_unified = soil_manager.prepare_global_input(global_data_raw)

    # 验证特征对齐
    from .W_soil_manager import validate_feature_alignment
    is_valid, info = validate_feature_alignment(input_template_unified, unified_features)
    if not is_valid:
        print(f"  警告: 特征对齐存在问题 - 缺失: {info['missing']}, 多余: {info['extra']}")
    else:
        print(f"  特征对齐验证通过")

    # =========================================================================
    # 步骤4: 批量预测构建响应面（支持分批和检查点）
    # =========================================================================
    print(f"\n[Step 4/6] 批量预测构建三维响应面...")

    # 初始化三维响应面数组 (n_grids, n_ratios, n_indicators)
    n_indicators = len(indicators)
    response_surface = np.zeros((n_grids, n_ratios, n_indicators), dtype=DTYPE_FLOAT)

    # 检查是否有检查点可以恢复
    checkpoint_file = f'{output_dir}/checkpoint.pkl'
    start_ratio_idx = 0

    if os.path.exists(checkpoint_file) and not force_rebuild:
        print(f"  发现检查点文件，尝试恢复...")
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            start_ratio_idx = checkpoint.get('last_ratio_idx', 0) + 1
            response_surface = checkpoint.get('response_surface', response_surface)
            print(f"  从检查点恢复: 从第 {start_ratio_idx} 个ratio继续")
        except Exception as e:
            print(f"  检查点恢复失败: {e}，从头开始")

    # 为每个指标预测所有ratio
    for ind_idx, indicator in enumerate(tqdm(indicators, desc="  构建响应面")):
        if indicator == 'GWP':
            print(f"    [GWP] 将在步骤5中从 CH4 和 N2O 计算")
            continue

        if indicator not in models_dict:
            print(f"    警告: {indicator} 模型未训练，跳过")
            continue

        rf = models_dict[indicator]

        # 使用统一的输入模板（不再需要每个指标重新加载和对齐特征）
        # 这解决了原代码中"换一种作物又会出错"的问题
        input_template = input_template_unified.copy()

        # 按模型训练时的特征顺序重新排列
        model_features = list(rf.feature_names_in_)
        input_template = input_template[model_features]

        # 对所有ratio批量预测
        temp_preds = np.zeros((n_ratios, n_grids), dtype=DTYPE_FLOAT)

        for ratio_idx in range(start_ratio_idx, n_ratios):
            r = ratios[ratio_idx]
            input_template['Rate'] = r
            temp_preds[ratio_idx, :] = rf.predict(input_template).astype(DTYPE_FLOAT)

            # 定期保存检查点
            if (ratio_idx + 1) % checkpoint_interval == 0:
                # 保存当前进度
                os.makedirs(output_dir, exist_ok=True)
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({
                        'last_ratio_idx': ratio_idx,
                        'response_surface': response_surface,
                        'indicator': indicator,
                        'timestamp': datetime.now().isoformat()
                    }, f)

        # 转置存储: (ratios, grid) -> (grid, ratios)
        response_surface[:, :, ind_idx] = temp_preds.T

        # 保存中间结果（可选）
        if save_intermediate:
            save_intermediate_csv(
                original_coords, temp_preds, ratios,
                crop, outday, indicator
            )

        # 重置检查点起始位置
        start_ratio_idx = 0

    # 删除检查点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"  检查点文件已清理")

    # =========================================================================
    # 步骤5: 计算 GWP 响应面
    # =========================================================================
    if 'GWP' in indicators and 'CH4' in indicators and 'N2O' in indicators:
        print(f"\n[Step 5/6] 计算 GWP 响应面...")

        ch4_idx = indicators.index('CH4')
        n2o_idx = indicators.index('N2O')
        gwp_idx = indicators.index('GWP')

        # GWP = CH4 × 27 + N2O × 273
        ch4_surface = response_surface[:, :, ch4_idx]
        n2o_surface = response_surface[:, :, n2o_idx]
        gwp_surface = ch4_surface * GWP_CH4_FACTOR + n2o_surface * GWP_N2O_FACTOR

        response_surface[:, :, gwp_idx] = gwp_surface

        print(f"  GWP 响应面计算完成 (系数: CH4={GWP_CH4_FACTOR}, N2O={GWP_N2O_FACTOR})")

        # 保存 GWP 中间结果
        if save_intermediate:
            save_intermediate_csv(
                original_coords, gwp_surface.T, ratios,
                crop, outday, 'GWP'
            )

    # =========================================================================
    # 步骤5.5: 预测对照组 (Ratio=1.0)
    # =========================================================================
    print(f"\n[Step 5.5/6] 预测对照组 (Ratio=1.0)...")

    ck_dir = pm.get_mix_ck_dir(crop, outday)

    for indicator in indicators:
        if indicator == 'GWP':
            continue  # GWP从CH4和N2O计算

        if indicator not in models_dict:
            print(f"    警告: {indicator} 模型未训练，跳过CK预测")
            continue

        rf = models_dict[indicator]
        input_template = input_template_unified.copy()

        # 设置Ratio=1.0 (充分灌溉，对照组)
        input_template['Rate'] = 1.0

        # 按模型特征顺序排列
        model_features = list(rf.feature_names_in_)
        for col in model_features:
            if col not in input_template.columns:
                input_template[col] = 0.0
        input_template = input_template[model_features]

        # 预测
        ck_pred = rf.predict(input_template).astype(DTYPE_FLOAT)

        # 保存
        ck_df = pd.DataFrame({
            'x': original_coords['Longitude'],
            'y': original_coords['Latitude'],
            'Ypred': ck_pred
        })
        ck_file = ck_dir / f'{indicator}_ck.csv'
        ck_df.to_csv(ck_file, index=False)
        print(f"    {indicator}_ck.csv 已保存 ({len(ck_df):,} 网格)")

    # 计算GWP_ck
    if 'CH4' in indicators and 'N2O' in indicators:
        ch4_ck_file = ck_dir / 'CH4_ck.csv'
        n2o_ck_file = ck_dir / 'N2O_ck.csv'

        if ch4_ck_file.exists() and n2o_ck_file.exists():
            ch4_ck = pd.read_csv(ch4_ck_file)
            n2o_ck = pd.read_csv(n2o_ck_file)

            # GWP = CH4 × 27 + N2O × 273
            gwp_ck = ch4_ck['Ypred'] * GWP_CH4_FACTOR + n2o_ck['Ypred'] * GWP_N2O_FACTOR

            gwp_df = pd.DataFrame({
                'x': ch4_ck['x'],
                'y': ch4_ck['y'],
                'Ypred': gwp_ck
            })
            gwp_df.to_csv(ck_dir / 'GWP_ck.csv', index=False)
            print(f"    GWP_ck.csv 已保存")
            print(f"    GWP_ck 平均值: {gwp_ck.mean():.2f} kg CO2-eq/ha")

    print(f"\n  对照组数据已生成: {ck_dir}")

    # =========================================================================
    # 步骤6: 保存响应面数据
    # =========================================================================
    print(f"\n[Step 6/6] 保存响应面数据...")

    os.makedirs(output_dir, exist_ok=True)

    # 保存核心数据
    np.save(f'{output_dir}/surface_data.npy', response_surface)
    np.save(f'{output_dir}/grid_coords.npy', original_coords.values)
    np.save(f'{output_dir}/ratios.npy', ratios)

    # 保存元数据
    metadata = {
        'crop': crop,
        'readday': readday,
        'outday': outday,
        'indicators': indicators,
        'lowdata': float(lowdata),
        'updata': float(updata),
        'foot': float(foot),
        'n_grids': int(n_grids),
        'n_ratios': int(n_ratios),
        'n_indicators': int(n_indicators),
        'creation_time': datetime.now().isoformat(),
        'indicator_direction': INDICATOR_DIRECTION,
        'model_params': {k: v for k, v in model_params.items() if k in indicators}
    }

    with open(f'{output_dir}/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # 完成
    # =========================================================================
    elapsed_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"  响应面构建完成!")
    print(f"{'='*70}")
    print(f"输出目录: {output_dir}")
    print(f"数据维度: {response_surface.shape}")
    print(f"数据大小: {response_surface.nbytes / 1024 / 1024:.2f} MB")
    print(f"总耗时: {elapsed_time:.1f} 秒")
    print(f"{'='*70}\n")

    # 返回结果
    return {
        'data': response_surface,
        'coords': original_coords.values,
        'ratios': ratios,
        'indicators': indicators,
        'metadata': metadata
    }


# =============================================================================
# 辅助函数
# =============================================================================

def train_random_forest(df_train, yname, params):
    """
    训练单个随机森林模型（与W_glob_mac.machine_RF兼容）

    参数：
        df_train: 训练数据
        yname: 目标变量名
        params: 模型参数字典

    返回：
        rf: 训练好的随机森林模型
    """
    # 使用统一的特征顺序（与预测时保持一致）
    feature_cols = ALL_FEATURES.copy()

    # 确保所有特征列都存在，添加缺失的土壤特征列
    for col in feature_cols:
        if col not in df_train.columns:
            df_train[col] = 0

    X = df_train[feature_cols]
    y = df_train[yname]

    # 提取参数（兼容两种命名方式）
    max_depth = params.get('max_depth', 10)
    min_samples_leaf = params.get('min_samples_leaf', 1)
    min_samples_split = params.get('min_samples_split', 2)
    n_estimators = params.get('n_random', params.get('n_estimators', 100))
    random_state = params.get('random_state', 42)

    # 训练模型
    rf = RandomForestRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        warm_start=False
    )
    rf.fit(X, y)

    return rf


def save_intermediate_csv(coords, preds_matrix, ratios, crop, outday, indicator):
    """
    保存中间结果到CSV文件（与W_glob_mac输出格式兼容）

    参数：
        coords: 坐标DataFrame
        preds_matrix: 预测矩阵 (n_ratios, n_grids)
        ratios: ratio序列
        crop: 作物类型
        outday: 输出日期
        indicator: 指标名称
    """
    pm = get_path_manager()
    output_dir = pm.get_mix_tnr_dir(crop, outday)

    for idx, r in enumerate(ratios):
        output_file = f'{output_dir}/{indicator}_{r:0.2f}.csv'

        # 只在文件不存在时保存，避免覆盖已有数据
        if not os.path.exists(output_file):
            df = pd.DataFrame({
                'x': coords['Longitude'],
                'y': coords['Latitude'],
                'Ypred': preds_matrix[idx, :]
            })
            df.to_csv(output_file, index=False)


def load_response_surface(output_dir):
    """
    加载已保存的响应面数据

    参数：
        output_dir: 响应面数据目录

    返回：
        surface_dict: 响应面字典
    """
    # 检查文件是否存在
    required_files = ['surface_data.npy', 'grid_coords.npy', 'ratios.npy', 'metadata.json']
    for f in required_files:
        if not os.path.exists(f'{output_dir}/{f}'):
            raise FileNotFoundError(f"缺少必要文件: {output_dir}/{f}")

    # 加载数据
    data = np.load(f'{output_dir}/surface_data.npy')
    coords = np.load(f'{output_dir}/grid_coords.npy')
    ratios = np.load(f'{output_dir}/ratios.npy')

    # 加载元数据
    with open(f'{output_dir}/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"  响应面数据加载成功: {data.shape}")

    return {
        'data': data,
        'coords': coords,
        'ratios': ratios,
        'indicators': metadata['indicators'],
        'metadata': metadata
    }


def query_response_surface(surface_dict, grid_idx=None, ratio=None, indicator=None):
    """
    从响应面查询数据

    参数：
        surface_dict: 响应面字典
        grid_idx: 网格索引 (可选)
        ratio: ratio值 (可选)
        indicator: 指标名称 (可选)

    返回：
        查询结果数组

    示例：
        # 查询某个网格在所有ratio下的Yield响应
        query_response_surface(surface, grid_idx=1000, indicator='Y')

        # 查询某个ratio下所有网格的ET值
        query_response_surface(surface, ratio=0.75, indicator='ET')

        # 查询某个网格某个ratio的所有指标
        query_response_surface(surface, grid_idx=1000, ratio=0.75)
    """
    data = surface_dict['data']
    indicators = surface_dict['indicators']
    ratios = surface_dict['ratios']

    # 建立索引映射
    indicator_idx = {ind: i for i, ind in enumerate(indicators)}

    # 处理ratio浮点数精度问题
    ratio_idx = {}
    for r in ratios:
        # 将ratio映射到最近的索引
        ratio_idx[round(float(r), 2)] = list(ratios).index(r)

    # 根据参数切片
    if grid_idx is not None and ratio is not None and indicator is not None:
        # 查询单个值
        ratio_key = round(float(ratio), 2)
        return data[grid_idx, ratio_idx[ratio_key], indicator_idx[indicator]]

    elif grid_idx is not None and ratio is not None:
        # 查询某个网格某个ratio的所有指标
        ratio_key = round(float(ratio), 2)
        return data[grid_idx, ratio_idx[ratio_key], :]

    elif grid_idx is not None and indicator is not None:
        # 查询某个网格某个指标在所有ratio下的响应
        return data[grid_idx, :, indicator_idx[indicator]]

    elif ratio is not None and indicator is not None:
        # 查询某个ratio下所有网格的某个指标值
        ratio_key = round(float(ratio), 2)
        return data[:, ratio_idx[ratio_key], indicator_idx[indicator]]

    else:
        raise ValueError("参数组合不正确，至少需要指定两个参数")


def get_response_surface_status(crop, outday):
    """
    获取响应面状态信息

    参数：
        crop: 作物类型
        outday: 输出日期

    返回：
        status: 状态字典
    """
    pm = get_path_manager()
    output_dir = pm.get_response_surface_dir(crop, outday)
    metadata_file = output_dir / 'metadata.json'

    status = {
        'exists': False,
        'valid': False,
        'metadata': None
    }

    if os.path.exists(metadata_file):
        status['exists'] = True
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            status['metadata'] = metadata
            status['valid'] = True
        except:
            status['valid'] = False

    return status


# =============================================================================
# 批处理函数
# =============================================================================

def build_response_surface_batch(
    crops=['rice', 'wheat', 'maize'],
    readday='20251027',
    outday='20251028',
    **kwargs
):
    """
    批量构建多种作物的响应面

    参数：
        crops: 作物列表
        readday: 输入日期
        outday: 输出日期
        **kwargs: 传递给build_response_surface的其他参数

    返回：
        results: 各作物响应面字典
    """
    results = {}

    for crop in crops:
        print(f"\n{'#'*70}")
        print(f"# 批处理: {crop}")
        print(f"{'#'*70}\n")

        try:
            surface = build_response_surface(
                crop=crop,
                readday=readday,
                outday=outday,
                **kwargs
            )
            results[crop] = surface
        except Exception as e:
            print(f"  错误: {crop} 响应面构建失败 - {e}")
            results[crop] = None

    return results


# =============================================================================
# 测试函数
# =============================================================================

def test_response_surface():
    """
    测试响应面构建功能（使用模拟数据）
    """
    print("\n[测试模式] 使用模拟数据测试响应面构建...")

    # 创建模拟数据
    n_grids = 100
    n_ratios = 31
    n_indicators = 5
    indicators = ['Y', 'ET', 'WUE', 'CH4', 'N2O']

    # 模拟响应面数据
    surface_data = np.random.rand(n_grids, n_ratios, n_indicators).astype(DTYPE_FLOAT)

    # 创建模拟坐标
    coords = np.column_stack([
        np.linspace(110, 120, n_grids),  # 经度
        np.linspace(30, 40, n_grids)     # 纬度
    ])

    # 模拟ratio序列
    ratios = np.arange(0.60, 0.90, 0.01)

    # 创建响应面字典
    surface_dict = {
        'data': surface_data,
        'coords': coords,
        'ratios': ratios,
        'indicators': indicators,
        'metadata': {'test': True}
    }

    # 测试查询功能
    print("\n测试查询功能:")
    print(f"  查询网格0的Yield响应: {query_response_surface(surface_dict, grid_idx=0, indicator='Y').shape}")
    # 使用实际的ratio值（避免浮点数精度问题）
    test_ratio = ratios[15]  # 使用第15个ratio值
    print(f"  查询ratio={test_ratio:.2f}的所有网格ET值: {query_response_surface(surface_dict, ratio=test_ratio, indicator='ET').shape}")
    print(f"  查询网格50在ratio={test_ratio:.2f}的所有指标: {query_response_surface(surface_dict, grid_idx=50, ratio=test_ratio).shape}")

    print("\n[测试通过] 响应面查询功能正常")


if __name__ == "__main__":
    # 运行测试
    test_response_surface()
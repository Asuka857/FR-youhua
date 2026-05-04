"""
土壤类型管理模块 (Soil Texture Manager)
统一管理土壤类型的编码、映射和特征构建

解决的问题：
1. 训练数据中不同指标的土壤类型独热编码列不一致
2. 全球预测数据与训练数据特征列不对齐
3. 换作物后代码报错

创建时间：2025-03-20
最后修改：2025-03-20
"""

import numpy as np
import pandas as pd

# =============================================================================
# 土壤类型全集定义
# =============================================================================

SOIL_TEXTURES = {
    'silt loam': {'code': 7, 'display_name': '粉壤土'},
    'loam': {'code': 9, 'display_name': '壤土'},
    'clay loam': {'code': 5, 'display_name': '粘壤土'},
    'sandy loam': {'code': 11, 'display_name': '砂壤土'},
    'loamy sand': {'code': 12, 'display_name': '壤砂土'}
}

# 编码到名称的映射
CODE_TO_NAME = {v['code']: k for k, v in SOIL_TEXTURES.items()}

# 名称到编码的映射
NAME_TO_CODE = {k: v['code'] for k, v in SOIL_TEXTURES.items()}

# 所有土壤类型名称列表（固定顺序）
ALL_TEXTURE_NAMES = list(SOIL_TEXTURES.keys())

# 所有独热编码特征列名（固定顺序）
ALL_TEXTURE_FEATURES = [f'texture_{name}' for name in ALL_TEXTURE_NAMES]

# 基础特征列名
BASE_FEATURES = ['Ypre', 'Ytem', 'N.rate', 'Rate', 'SOC', 'PH', 'fertilizer', 'mulch']

# 完整特征列名
ALL_FEATURES = BASE_FEATURES + ALL_TEXTURE_FEATURES


# =============================================================================
# 核心函数
# =============================================================================

def get_texture_features():
    """获取完整的土壤质地特征列名列表"""
    return ALL_TEXTURE_FEATURES.copy()


def get_base_features():
    """获取基础特征列名列表（不含土壤质地）"""
    return BASE_FEATURES.copy()


def get_all_features():
    """获取完整的特征列名列表"""
    return ALL_FEATURES.copy()


def encode_global_soil_texture(global_data):
    """
    对全球数据的土壤质地进行编码
    确保生成所有5种土壤类型的独热编码列

    参数：
        global_data: 包含 soil_texture 列的DataFrame

    返回：
        encoded_data: 包含完整土壤质地编码的DataFrame
    """
    data = global_data.copy()

    # 将数字编码转换为名称
    if data['soil_texture'].dtype in ['int64', 'float64', 'int32', 'float32']:
        data['soil_texture'] = data['soil_texture'].map(CODE_TO_NAME)

    # 执行独热编码
    data = pd.get_dummies(data, columns=['soil_texture'], prefix='texture')

    # 添加缺失的土壤类型列（确保所有5种类型都有）
    for texture_name in ALL_TEXTURE_NAMES:
        col_name = f'texture_{texture_name}'
        if col_name not in data.columns:
            data[col_name] = 0

    return data


def encode_training_soil_texture(df):
    """
    对训练数据的土壤质地进行编码
    确保生成所有5种土壤类型的独热编码列

    参数：
        df: 包含 soil.texture 列的DataFrame

    返回：
        encoded_df: 包含完整土壤质地编码的DataFrame
    """
    data = df.copy()

    # 过滤非法土壤类型
    data['soil.texture'] = data['soil.texture'].apply(
        lambda x: x if x in ALL_TEXTURE_NAMES else np.nan
    )

    # 执行独热编码
    data = pd.get_dummies(data, columns=['soil.texture'], prefix='texture')

    # 添加缺失的土壤类型列（确保所有5种类型都有）
    for texture_name in ALL_TEXTURE_NAMES:
        col_name = f'texture_{texture_name}'
        if col_name not in data.columns:
            data[col_name] = 0

    return data


def align_features(df, feature_cols=None, keep_coords=True):
    """
    对齐特征列到统一的特征空间

    参数：
        df: 输入DataFrame
        feature_cols: 目标特征列（默认使用全集）
        keep_coords: 是否保留经纬度坐标列

    返回：
        aligned_df: 特征对齐后的DataFrame
    """
    if feature_cols is None:
        feature_cols = ALL_FEATURES

    result = df.copy()

    # 添加缺失的特征列
    for col in feature_cols:
        if col not in result.columns:
            result[col] = 0.0

    # 确保列顺序一致
    if keep_coords:
        # 保留坐标列
        coord_cols = [col for col in ['Longitude', 'Latitude', 'x', 'y'] if col in result.columns]
        final_cols = coord_cols + feature_cols
        # 只选择存在的列
        final_cols = [col for col in final_cols if col in result.columns]
        result = result[final_cols]
    else:
        result = result[feature_cols]

    return result


def validate_feature_alignment(df, expected_features=None):
    """
    验证特征对齐情况

    参数：
        df: 输入DataFrame
        expected_features: 期望的特征列

    返回：
        (is_valid, info): 是否对齐及详细信息
    """
    if expected_features is None:
        expected_features = ALL_FEATURES

    actual_features = [col for col in df.columns if col in expected_features]
    missing = set(expected_features) - set(actual_features)
    extra = set(df.columns) - set(expected_features) - {'Longitude', 'Latitude', 'x', 'y'}

    is_valid = len(missing) == 0 and len(extra) == 0

    info = {
        'expected_count': len(expected_features),
        'actual_count': len(actual_features),
        'missing': missing,
        'extra': extra,
        'is_aligned': is_valid
    }

    return is_valid, info


# =============================================================================
# 特征管理器类
# =============================================================================

class SoilFeatureManager:
    """
    土壤特征管理器
    集中管理土壤类型相关的所有操作

    使用示例：
        manager = SoilFeatureManager()

        # 训练时
        df_encoded = manager.encode_training_data(df_raw)
        X = df_encoded[manager.all_features]

        # 预测时
        input_template = manager.prepare_global_input(global_data_raw)
        predictions = model.predict(input_template)
    """

    def __init__(self):
        self.textures = SOIL_TEXTURES
        self.all_features = ALL_FEATURES
        self.base_features = BASE_FEATURES
        self.texture_features = ALL_TEXTURE_FEATURES

    def encode_training_data(self, df):
        """
        编码训练数据

        参数：
            df: 原始训练数据

        返回：
            encoded_df: 编码后的数据
        """
        # 处理离散变量
        df = df.copy()
        df['fertilizer.with.IR'] = df['fertilizer.with.IR'].replace('YES', 'yes')
        df['mulch'] = df['mulch'].replace('YES', 'yes')
        df['fertilizer'] = df['fertilizer.with.IR'].replace({'yes': 1, 'no': 0}).fillna(0).astype(int)
        df['mulch'] = df['mulch'].replace({'yes': 1, 'no': 0}).fillna(0).astype(int)

        # 编码土壤质地
        return encode_training_soil_texture(df)

    def prepare_global_input(self, global_data_raw):
        """
        准备全球预测输入

        参数：
            global_data_raw: 原始全球数据

        返回：
            input_template: 准备好的输入模板（仅包含特征，不含坐标）
        """
        input_template = global_data_raw.copy()
        # 不保留坐标列，因为模型训练时不包含这些特征
        return align_features(input_template, self.all_features, keep_coords=False)

    def get_feature_subset(self, include_coords=False):
        """
        获取特征列表

        参数：
            include_coords: 是否包含坐标列

        返回：
            features: 特征列表
        """
        if include_coords:
            return ['Longitude', 'Latitude'] + self.all_features
        return self.all_features.copy()

    def print_info(self):
        """打印特征信息"""
        print("=" * 50)
        print("土壤特征管理器信息")
        print("=" * 50)
        print(f"支持的土壤类型数量: {len(self.textures)}")
        print(f"土壤类型列表: {list(self.textures.keys())}")
        print(f"基础特征数量: {len(self.base_features)}")
        print(f"土壤特征数量: {len(self.texture_features)}")
        print(f"总特征数量: {len(self.all_features)}")
        print("=" * 50)


# =============================================================================
# 便捷函数
# =============================================================================

def create_manager():
    """创建土壤特征管理器实例"""
    return SoilFeatureManager()


def quick_align(df, keep_coords=True):
    """快速对齐特征（便捷函数）"""
    return align_features(df, ALL_FEATURES, keep_coords)


# =============================================================================
# 测试函数
# =============================================================================

def test_soil_manager():
    """测试土壤管理模块"""
    print("\n[测试] 土壤管理模块...")

    # 创建管理器
    manager = SoilFeatureManager()
    manager.print_info()

    # 测试训练数据编码
    print("\n测试训练数据编码...")
    df_train = pd.DataFrame({
        'soil.texture': ['silt loam', 'loam', 'clay loam', 'silt loam', 'invalid_type'],
        'PH': [6.5, 7.0, 6.8, 6.2, 7.5],
        'SOC': [10, 15, 12, 8, 20]
    })
    df_encoded = encode_training_soil_texture(df_train)
    print(f"编码后列: {[col for col in df_encoded.columns if col.startswith('texture_')]}")

    # 测试全球数据编码
    print("\n测试全球数据编码...")
    df_global = pd.DataFrame({
        'soil_texture': [7, 9, 5, 11, 12],  # 全部5种类型
        'PH': [6.5, 7.0, 6.8, 6.2, 7.5],
        'Longitude': [110, 111, 112, 113, 114],
        'Latitude': [30, 31, 32, 33, 34]
    })
    df_global_encoded = encode_global_soil_texture(df_global)
    print(f"编码后列: {[col for col in df_global_encoded.columns if col.startswith('texture_')]}")

    # 测试特征对齐
    print("\n测试特征对齐...")
    is_valid, info = validate_feature_alignment(df_global_encoded, ALL_FEATURES)
    print(f"特征对齐状态: {'通过' if is_valid else '失败'}")
    print(f"缺失特征: {info['missing']}")

    print("\n[测试通过] 土壤管理模块工作正常")


if __name__ == "__main__":
    test_soil_manager()
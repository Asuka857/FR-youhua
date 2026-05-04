import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

# 性能优化：使用float32代替float64
DTYPE_FLOAT = np.float32

# CSV缓存字典
_csv_cache = {}

# 尝试导入土壤特征管理模块
USE_SOIL_MANAGER = False
try:
    from W_soil_manager import (
        encode_training_soil_texture,
        encode_global_soil_texture,
        ALL_TEXTURE_NAMES,
        ALL_FEATURES
    )
    USE_SOIL_MANAGER = True
except ImportError:
    # 尝试相对导入
    try:
        from .W_soil_manager import (
            encode_training_soil_texture,
            encode_global_soil_texture,
            ALL_TEXTURE_NAMES,
            ALL_FEATURES
        )
        USE_SOIL_MANAGER = True
    except ImportError:
        pass


def read_csv_cached(file_path, usecols=None):
    """带缓存的CSV读取"""
    cache_key = (file_path, tuple(usecols) if usecols else None)

    if cache_key in _csv_cache:
        return _csv_cache[cache_key].copy()

    # 默认使用float32
    dtype = {col: DTYPE_FLOAT for col in (usecols or ['x', 'y'])}
    df = pd.read_csv(file_path, usecols=usecols, dtype=dtype)
    _csv_cache[cache_key] = df.copy()

    return df


#处理函数
def data_processed_one(file_path,yname,lowdata,updata):#数据读取函数，数据整理部分
    # 添加dtype字典优化
    dtype_dict = {
        'ID': 'int32', 'VID': 'int32',
        'PH': DTYPE_FLOAT, 'SOC': DTYPE_FLOAT,
        'Ypre': DTYPE_FLOAT, 'Ytem': DTYPE_FLOAT,
        'N.rate': DTYPE_FLOAT, 'Rate': DTYPE_FLOAT
    }

    data = pd.read_csv(file_path, dtype=dtype_dict)#读取文件
    #读取实验组数据
    columns_n = ['ID','VID','soil.texture','PH','SOC','Ypre','Ytem','N.rate','Rate','mulch','fertilizer.with.IR','nir','nctrl',f'{yname}ir']#列名提取
    dfn = data[columns_n].copy()#获取列名数据
    dfn['source'] = 'ir'#标明来源
    dfn = dfn.dropna(subset=[f'{yname}ir'])
    dfn = dfn[(dfn['Rate']>=lowdata)&(dfn['Rate']<updata)]#提取最优区间
    #读取对照组数据
    columns_ck = ['ID','VID','soil.texture','PH','SOC','Ypre','Ytem','N.rate','Rate','mulch','fertilizer.with.IR','nir','nctrl',f'{yname}ctrl']
    dfck = data[columns_ck].copy()#获取列名数据
    dfck = dfck.drop_duplicates(subset=['ID',f'{yname}ctrl'])#选取不重复数据
    dfck = dfck.dropna(subset=[f'{yname}ctrl'])
    dfck['source'] = 'ctrl'
    dfck['Rate'] = 1 #认为对照组是1
    #统一数据
    dfn.rename(columns={f'{yname}ir': f'{yname}'},inplace=True)
    dfck.rename(columns={f'{yname}ctrl': f'{yname}'},inplace=True)
    df = pd.concat([dfn,dfck],ignore_index= True)
    return df

def data_processed_two(df):#数据处理函数，独热编码部分
    """
    数据处理函数，独热编码部分

    改进：使用统一的土壤类型全集，确保所有指标训练数据生成相同的特征列
    """
    #对离散型变量进行编码
    df['fertilizer.with.IR'] = df['fertilizer.with.IR'].replace('YES', 'yes')
    df['mulch'] = df['mulch'].replace('YES', 'yes')
    # fertilizer 列处理
    df['fertilizer'] = df['fertilizer.with.IR'].replace({'yes': 1, 'no': 0})
    df['fertilizer'] = df['fertilizer'].fillna(0).astype(int)
    # mulch 列处理
    df['mulch'] = df['mulch'].replace({'yes': 1,'no': 0})
    df['mulch'] = df['mulch'].fillna(0).astype(int)

    # 使用土壤管理模块进行编码（确保特征一致性）
    if USE_SOIL_MANAGER:
        df = encode_training_soil_texture(df)
    else:
        # 兼容旧逻辑
        textures =['silt loam','loam','clay loam','sandy loam','loamy sand']
        df['soil.texture']=df['soil.texture'].apply(lambda x: x if x in textures else np.nan)
        df = pd.get_dummies(df, columns=['soil.texture'], prefix='texture')

    return df    

def machine_RF(df,yname,max_depth,min_samples_leaf,min_samples_split,n_random,i):
    # 使用统一的特征顺序（确保与预测时一致）
    if USE_SOIL_MANAGER:
        feature_cols = ALL_FEATURES.copy()
        # 确保所有特征列都存在
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
    else:
        feature_cols = ['Ypre','Ytem','N.rate','Rate','SOC','PH','fertilizer','mulch']+ [col for col in df.columns if col.startswith('texture_')]

    X = df[feature_cols]
    y = df[f'{yname}']
    #训练模型
    rf = RandomForestRegressor(max_depth=max_depth,
                               min_samples_leaf=min_samples_leaf,
                               min_samples_split=min_samples_split,
                               n_estimators=n_random,
                               random_state=i,
                               n_jobs=-1,  # 启用底层并行
                               warm_start=False)
    rf.fit(X, y)
    return rf

def global_data_processed(df,global_file_path):


    # 读取全球文件
    global_data = pd.read_csv(global_file_path)

    # 保存原始经纬度
    original_coords = global_data[['Longitude', 'Latitude']]

    # 修改列名
    global_data.rename(columns={
         'Band_1': 'Ypre',
        'Band_2': 'Ytem',
        'Band_3': 'PH',
        'Band_4': 'SOC',
        'Band_5': 'soil_texture',
        'Total_N':'N.rate',

    }, inplace=True)

    # 将 Ypre 的值变为原来的 12 倍
    global_data['Ypre'] = global_data['Ypre'] * 12 #月均值变为年值
 #   global_data['SOC'] = global_data['SOC'] * 10 #因为全球数据中SOC的单位是%，而训练数据的单位是g/kg 10g/kg = 1 %

    # 土壤质地转换
    texture_mapping = {7: 'silt loam', 5: 'clay loam', 9: 'loam', 11: 'sandy loam', 12: 'loamy sand'}
    global_data['soil_texture'] = global_data['soil_texture'].map(texture_mapping)
    global_data = pd.get_dummies(global_data, columns=['soil_texture'], prefix='texture')

    # 新增 fertilizer 和 mulch 列，并设置固定值
    global_data['fertilizer'] = 1  
    global_data['mulch'] = 1  
    
    # 新增 Rate 列并设置为 NaN
    global_data['Rate'] = np.nan

    # 确保 global_data 的特征与训练时的 X 完全一致
    X = df[['Ypre','Ytem','N.rate','Rate','SOC','PH','fertilizer','mulch']+ [col for col in df.columns if col.startswith('texture_')]]
    missing_cols = [col for col in X.columns if col not in global_data.columns]
    for col in missing_cols:
        global_data[col] = False  # 填补缺失的列，数值列默认填充为 0
    extra_cols = [col for col in global_data.columns if col not in X.columns]
    global_data = global_data.drop(columns=extra_cols)  # 删除多余的列
    global_data = global_data[X.columns]  # 确保列的顺序与训练一致

    return global_data, original_coords

def global_data_processed_one(global_file_path):#全球数据加工
    """
    预处理全球数据并使其与训练特征对齐。

    改进：使用统一的土壤类型全集编码，确保与训练数据特征一致

    参数：
    - global_file_path: 全球数据文件路径
    返回：
    - global_data: 预处理后的全球数据，特征对齐
    - original_coords: 原始经纬度数据
    """
    # 读取全球文件
    global_data = pd.read_csv(global_file_path)

    # 保存原始经纬度
    original_coords = global_data[['Longitude', 'Latitude']]

    # 修改列名
    global_data.rename(columns={
         'Band_1': 'Ypre',
        'Band_2': 'Ytem',
        'Band_3': 'PH',
        'Band_4': 'SOC',
        'Band_5': 'soil_texture',
        'Total_N': 'N.rate',
        'Total_N_raw': 'N.rate',  # 兼容两种列名
    }, inplace=True)

    # 将 Ypre 的值变为原来的 12 倍
    global_data['Ypre'] = global_data['Ypre'] * 12
 #   global_data['SOC'] = global_data['SOC'] * 10 #因为全球数据中SOC的单位是%，而训练数据的单位是g/kg 10g/kg = 1 %

    # 使用统一的土壤编码
    if USE_SOIL_MANAGER:
        global_data = encode_global_soil_texture(global_data)
    else:
        # 兼容旧逻辑
        texture_mapping = {7: 'silt loam', 5: 'clay loam', 9: 'loam', 11: 'sandy loam', 12: 'loamy sand'}
        global_data['soil_texture'] = global_data['soil_texture'].map(texture_mapping)
        global_data = pd.get_dummies(global_data, columns=['soil_texture'], prefix='texture')

    # 新增 fertilizer 和 mulch 列，并设置固定值
    global_data['fertilizer'] = 1
    global_data['mulch'] = 1
    return global_data, original_coords

# =============================================================================
# 全流程流水线优化 (Pipeline)
# =============================================================================

def global_machine_pipeline(name, yname, crop, max_depth, min_samples_leaf, min_samples_split, n_random, i, foot, readday, outday, lowdata, updata):
    """
    全流程流水线优化：
    1. 训练一次模型，完成 Best, Last, CK, Normalization 所有任务。
    2. 减少 75% 的训练时间和 80% 的 IO 重复读取。
    """
    print(f"\n--- [Pipeline] 正在处理指标: {yname} ---")
    
    # 1. 准备路径
    file_path = f'F:/WaterML/{crop}/{readday}/D{name}.csv'
    global_file_path = f'F:/WaterML/{crop}/{readday}/x_data_point_with_TotalN.{crop}.csv'
    
    # 2. 训练模型 (仅一次)
    df_train = data_processed_one(file_path, yname, lowdata, updata)
    df_train = data_processed_two(df_train)
    rf = machine_RF(df_train, yname, max_depth, min_samples_leaf, min_samples_split, n_random, i)
    
    # 3. 预处理全球数据 (仅一次)
    global_data_raw, original_coords = global_data_processed_one(global_file_path)
    X_cols = df_train[['Ypre','Ytem','N.rate','Rate','SOC','PH','fertilizer','mulch']+ [col for col in df_train.columns if col.startswith('texture_')]].columns
    
    # 在循环外准备好基础矩阵
    input_template = global_data_raw.copy()
    missing_cols = [col for col in X_cols if col not in input_template.columns]
    for col in missing_cols: input_template[col] = 0.0
    input_template = input_template[X_cols] # 确保顺序

    # 4. 执行预测任务
    
    # A. 获取 Best (Max) 和 Last (Min)
    print(f"  [Task 1/3] 优化搜索 Best/Last...")
    rates = np.arange(lowdata, updata, foot)
    all_preds = np.zeros((len(rates), len(global_data_raw)), dtype=DTYPE_FLOAT)
    
    for idx, r in enumerate(tqdm(rates, desc="  Analyzing Rates", leave=False)):
        input_template['Rate'] = r  # 原地修改，极致加速
        all_preds[idx, :] = rf.predict(input_template).astype(DTYPE_FLOAT)
    
    # 计算 Max 和 Min
    best_idx = np.argmax(all_preds, axis=0)
    last_idx = np.argmin(all_preds, axis=0)
    
    y_best = all_preds[best_idx, np.arange(len(global_data_raw))]
    y_last = all_preds[last_idx, np.arange(len(global_data_raw))]
    
    # 保存 Best/Last
    output_dir_best = f'F:/WaterML/{crop}/{outday}/mix_best/W'
    os.makedirs(output_dir_best, exist_ok=True)
    
    res_best = pd.DataFrame({'x': original_coords['Longitude'], 'y': original_coords['Latitude'], 'Ypred': y_best, 'Rate': rates[best_idx]})
    res_last = pd.DataFrame({'x': original_coords['Longitude'], 'y': original_coords['Latitude'], 'Ypred': y_last, 'Rate': rates[last_idx]})
    
    res_best.to_csv(f'{output_dir_best}/W{yname}_1.csv', index=False)
    res_last.to_csv(f'{output_dir_best}/W{yname}_2.csv', index=False)
    
    # B. 获取 CK (Rate=1.0)
    print(f"  [Task 2/3] 计算 CK 模型...")
    output_dir_ck = f'F:/WaterML/{crop}/{outday}/mix_ck/W'
    os.makedirs(output_dir_ck, exist_ok=True)
    
    input_template['Rate'] = 1.0
    y_ck = rf.predict(input_template)
    res_ck = pd.DataFrame({'x': original_coords['Longitude'], 'y': original_coords['Latitude'], 'Ypred': y_ck})
    res_ck.to_csv(f'{output_dir_ck}/{name}_ck.csv', index=False)
    
    # C. 获取 Normalization (多 Rate 归一化)
    print(f"  [Task 3/3] 生成多 Rate 归一化数据...")
    output_dir_norm = f'F:/WaterML/{crop}/{outday}/mix_TNR/W'
    os.makedirs(output_dir_norm, exist_ok=True)
    
    for idx, r in enumerate(tqdm(rates, desc="  Normalizing", leave=False)):
        y_current = all_preds[idx, :]
        # 归一化逻辑
        norm_val = np.where((y_best == y_last), 1.0, (y_current - y_last) / (y_best - y_last))
        
        res_norm = pd.DataFrame({'x': original_coords['Longitude'], 'y': original_coords['Latitude'], 'Ypred': y_current, 'normalization': norm_val})
        res_norm.to_csv(f'{output_dir_norm}/{name}_{r:0.2f}.csv', index=False)
    
    print(f"  [Pipeline DONE] {yname} 处理完毕")

#基础预测，不做最优选择的预测
def global_machine_vs_1(name,rname,crop, value, outday):  # 用于比较值，得出比率
    # 用于得出比率（与ck相比）
    import pandas as pd
    # 优化文件路径
    file1 = f'F:/WaterML/{crop}/{outday}/mix_TNR/W/{name}_{rname}.csv'
    file2 = f'F:/WaterML/{crop}/{outday}/mix_ck/W/{name}_ck.csv'
    output_dir = f'F:/WaterML/{crop}/{outday}/ratio/W/{rname}'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{name}_{rname}.csv'
    output_file1 = f'{output_dir}/{name}_{rname}_mean.csv'

    # 1. 读取数据，只加载必要的列，设置数据类型
    print("开始读取数据...")
    usecols = ['x', 'y', value]
    dtype = {'x': 'float32', 'y': 'float32', value: 'float32'}
    df1 = pd.read_csv(file1, usecols=usecols, dtype=dtype)
    df2 = pd.read_csv(file2, usecols=usecols, dtype=dtype)

    # 2. 重命名列名，防止冲突
    df1 = df1.rename(columns={value: 'value1'})
    df2 = df2.rename(columns={value: 'value2'})

    # 3. 设置索引并合并，提高合并效率
    print("开始合并数据...")
    df1.set_index(['x', 'y'], inplace=True)
    df2.set_index(['x', 'y'], inplace=True)
    merged_df = df1.join(df2, how='inner').reset_index()

    # 4. 计算比值，处理除以0的情况
    print("开始计算 ratio...")
    merged_df['ratio'] = merged_df['value1'].div(merged_df['value2']).replace([float('inf'), -float('inf')], pd.NA)

    # 确保 ratio 列是数值类型 (pd.NA 可能导致 object 类型)
    merged_df['ratio'] = pd.to_numeric(merged_df['ratio'], errors='coerce')

    # 计算比率的平均值 (skipna=True 自动忽略 NaN)
    ratio_mean = merged_df['ratio'].mean(skipna=True)
    print(f"【统计结果】 {name} (VS CK) 比率 {rname} 的全球平均值: {ratio_mean}")
    pd.DataFrame({'mean_ratio': [ratio_mean]}).to_csv(output_file1, index=False)
    print(f"计算结果保存到：{output_file1}")
    # 5. 将结果保存到新的CSV文件
    merged_df.to_csv(output_file, index=False)
    print(f"结果已保存到: {output_file}")


def global_machine_vs_CH4_1(name, rname,crop, value, outday):  # 用于比较值，得出比率
    # 用于得出比率（与ck相比）
    import pandas as pd
    # 优化文件路径
    file1 = f'F:/WaterML/{crop}/{outday}/mix_TNR/W/{name}_{rname}.csv'
    file2 = f'F:/WaterML/{crop}/{outday}/mix_ck/W/{name}_ck.csv'
    output_dir = f'F:/WaterML/{crop}/{outday}/ratio/W/{rname}'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{name}_{rname}.csv'
    output_file1 = f'{output_dir}/{name}_{rname}_mean.csv'

    # 1. 读取数据，只加载必要的列，设置数据类型
    print("开始读取数据...")
    usecols = ['x', 'y', value]
    dtype = {'x': 'float32', 'y': 'float32', value: 'float32'}
    df1 = pd.read_csv(file1, usecols=usecols, dtype=dtype)
    df2 = pd.read_csv(file2, usecols=usecols, dtype=dtype)

    # 2. 重命名列名，防止冲突
    df1 = df1.rename(columns={value: 'value1'})
    df2 = df2.rename(columns={value: 'value2'})

    # 3. 设置索引并合并，提高合并效率
    print("开始合并数据...")
    df1.set_index(['x', 'y'], inplace=True)
    df2.set_index(['x', 'y'], inplace=True)
    merged_df = df1.join(df2, how='inner').reset_index()

    # **计算全局最小值**
    global_min = min(merged_df['value1'].min(), merged_df['value2'].min())

    # **数据调整（确保所有值大于 0）**
    shift_value = abs(global_min) + 1  # 计算偏移量
    merged_df['value1_adj'] = merged_df['value1'] + shift_value
    merged_df['value2_adj'] = merged_df['value2'] + shift_value

    # **计算比值**
    merged_df['ratio'] = merged_df['value1_adj'] / merged_df['value2_adj']

    # 确保 ratio 列是数值类型 (pd.NA 可能导致 object 类型)
    merged_df['ratio'] = pd.to_numeric(merged_df['ratio'], errors='coerce')

    # 计算比率的平均值 (skipna=True 自动忽略 NaN)
    ratio_mean = merged_df['ratio'].mean(skipna=True)
    print(f"【统计结果】 {name} (VS CK) 比率 {rname} 的全球平均值: {ratio_mean}")
    pd.DataFrame({'mean_ratio': [ratio_mean]}).to_csv(output_file1, index=False)
    print(f"计算结果保存到：{output_file1}")
    # 5. 将结果保存到新的CSV文件
    merged_df.to_csv(output_file, index=False)
    print(f"结果已保存到: {output_file}")


def GHG_1(crop, outday, sub_dir, suffix, outname):
    """
    基础 GHG 单文件计算 (保持兼容或用于特定合成)
    """
    ch4_prefix = "W" if sub_dir == "mix_best" else ""
    n2o_prefix = "W" if sub_dir == "mix_best" else ""
    
    file1 = f'F:/WaterML/{crop}/{outday}/{sub_dir}/W/{ch4_prefix}CH4_{suffix}.csv'
    file2 = f'F:/WaterML/{crop}/{outday}/{sub_dir}/W/{n2o_prefix}N2O_{suffix}.csv'
    
    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"找不到输入文件: {file1} 或 {file2}")
        return

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    Ypred_sum = df1['Ypred'] * 28 + df2['Ypred'] * 273
    result_df = df1[['x', 'y']].copy()
    result_df['Ypred'] = Ypred_sum
    output_dir = f'F:/WaterML/{crop}/{outday}/{sub_dir}/W'
    os.makedirs(output_dir, exist_ok=True)
    result_df.to_csv(f'{output_dir}/GHG_{outname}.csv', index=False)

def GHG_pipeline(crop, outday, lowdata, updata, foot):
    """
    GHG 逐步流水线计算逻辑：
    1. 遍历每个 Rate (0.01步长)，合成对应 Rate 的 GHG 数据。
    2. 计算基于所有 Rate 的 GHG 流水线最优值 (Best)、最小值 (Last)。
    3. 计算 GHG 的归一化 (Normalization) 和 CK (Rate=1.0)。
    """
    print(f"\n--- [GHG Pipeline] 开始 GHG 逐步流水线逻辑 ---")
    rates = np.arange(lowdata, updata, foot)
    
    # 获取坐标基准 (从 CK 中获取)
    base_file = f'F:/WaterML/{crop}/{outday}/mix_ck/W/CH4_ck.csv'
    if not os.path.exists(base_file):
        print("错误: 找不到 CH4_ck.csv 作为基准。")
        return
    base_df = pd.read_csv(base_file, usecols=['x', 'y'])
    n_points = len(base_df)
    
    # 存储所有 Rate 的预测值以进行寻优
    all_ghg_preds = np.zeros((len(rates), n_points), dtype=DTYPE_FLOAT)
    
    # 1. 逐步计算各 Rate 的 GHG
    output_dir_norm = f'F:/WaterML/{crop}/{outday}/mix_TNR/W'
    os.makedirs(output_dir_norm, exist_ok=True)
    
    for idx, r in enumerate(tqdm(rates, desc="  GHG Rate Processing", leave=False)):
        file_ch4 = f'{output_dir_norm}/CH4_{r:0.2f}.csv'
        file_n2o = f'{output_dir_norm}/N2O_{r:0.2f}.csv'
        
        if os.path.exists(file_ch4) and os.path.exists(file_n2o):
            df_ch4 = pd.read_csv(file_ch4, usecols=['Ypred'])
            df_n2o = pd.read_csv(file_n2o, usecols=['Ypred'])
            
            # GWP 合成
            ghg_val = df_ch4['Ypred'] * 28 + df_n2o['Ypred'] * 273
            all_ghg_preds[idx, :] = ghg_val.values.astype(DTYPE_FLOAT)
            
            # 保存当前 Rate 的 GHG CSV
            res_r = base_df.copy()
            res_r['Ypred'] = ghg_val
            res_r.to_csv(f'{output_dir_norm}/GHG_{r:0.2f}.csv', index=False)
    
    # 2. 寻优 (Best: Max GHG, Last: Min GHG)
    best_idx = np.argmax(all_ghg_preds, axis=0)
    last_idx = np.argmin(all_ghg_preds, axis=0)
    
    y_best = all_ghg_preds[best_idx, np.arange(n_points)]
    y_last = all_ghg_preds[last_idx, np.arange(n_points)]
    
    # 保存 Best/Last
    output_dir_best = f'F:/WaterML/{crop}/{outday}/mix_best/W'
    os.makedirs(output_dir_best, exist_ok=True)
    
    pd.DataFrame({'x': base_df['x'], 'y': base_df['y'], 'Ypred': y_best, 'Rate': rates[best_idx]}).to_csv(f'{output_dir_best}/GHG_1.csv', index=False)
    pd.DataFrame({'x': base_df['x'], 'y': base_df['y'], 'Ypred': y_last, 'Rate': rates[last_idx]}).to_csv(f'{output_dir_best}/GHG_2.csv', index=False)
    
    # 3. 计算 CK (Rate=1.0)
    print(f"  [Task] 计算 GHG CK...")
    file_ch4_ck = f'F:/WaterML/{crop}/{outday}/mix_ck/W/CH4_ck.csv'
    file_n2o_ck = f'F:/WaterML/{crop}/{outday}/mix_ck/W/N2O_ck.csv'
    if os.path.exists(file_ch4_ck) and os.path.exists(file_n2o_ck):
        df_ch4_ck = pd.read_csv(file_ch4_ck)
        df_n2o_ck = pd.read_csv(file_n2o_ck)
        y_ck = df_ch4_ck['Ypred'] * 28 + df_n2o_ck['Ypred'] * 273
        pd.DataFrame({'x': base_df['x'], 'y': base_df['y'], 'Ypred': y_ck}).to_csv(f'F:/WaterML/{crop}/{outday}/mix_ck/W/GHG_ck.csv', index=False)

    print(f"  [GHG Pipeline DONE] GHG 逐步流水线处理完毕")

def global_machine_vs_GHG_1(crop, outday):
    """
    专门针对 GHG 的 Ratio 计算 (兼容原有命名风格)
    """
    file_best = f'F:/WaterML/{crop}/{outday}/mix_best/W/GHG_1.csv'
    file_ck = f'F:/WaterML/{crop}/{outday}/mix_ck/W/GHG_ck.csv'
    output_dir = f'F:/WaterML/{crop}/{outday}/ratio/W/GHG'
    os.makedirs(output_dir, exist_ok=True)
    
    df_best = pd.read_csv(file_best)
    df_ck = pd.read_csv(file_ck)
    
    df_best.rename(columns={'Ypred': 'v_best'}, inplace=True)
    df_ck.rename(columns={'Ypred': 'v_ck'}, inplace=True)
    
    df_best.set_index(['x', 'y'], inplace=True)
    df_ck.set_index(['x', 'y'], inplace=True)
    res = df_best.join(df_ck, how='inner').reset_index()
    
    res['ratio'] = res['v_best'] / res['v_ck']
    ratio_mean = res['ratio'].mean()
    
    print(f"【统计结果】 GHG (Best/CK) 全球平均比率: {ratio_mean}")
    pd.DataFrame({'mean_ratio': [ratio_mean]}).to_csv(f'{output_dir}/GHG_ratio_mean.csv', index=False)
    res.to_csv(f'{output_dir}/GHG_ratio.csv', index=False)

def GHG_ratio_fixed(crop, outday, rname):
    """
    计算特定 Rate (如 0.75, 0.90) 的 GHG Ratio (Rate_X / CK)
    """
    file_rate = f'F:/WaterML/{crop}/{outday}/mix_TNR/W/GHG_{rname}.csv'
    file_ck = f'F:/WaterML/{crop}/{outday}/mix_ck/W/GHG_ck.csv'
    output_dir = f'F:/WaterML/{crop}/{outday}/ratio/W/{rname}'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f'{output_dir}/GHG_{rname}.csv'
    output_file_mean = f'{output_dir}/GHG_{rname}_mean.csv'

    if not os.path.exists(file_rate) or not os.path.exists(file_ck):
        print(f"找不到比例计算文件: {file_rate} 或 {file_ck}")
        return

    print(f"正在计算 GHG Ratio ({rname})...")
    df_r = pd.read_csv(file_rate)
    df_ck = pd.read_csv(file_ck)

    df_r.rename(columns={'Ypred': 'v_r'}, inplace=True)
    df_ck.rename(columns={'Ypred': 'v_ck'}, inplace=True)

    df_r.set_index(['x', 'y'], inplace=True)
    df_ck.set_index(['x', 'y'], inplace=True)
    res = df_r.join(df_ck, how='inner').reset_index()

    res['ratio'] = res['v_r'] / res['v_ck']
    ratio_mean = res['ratio'].mean()

    print(f"【统计结果】 GHG {rname} 全球平均比率: {ratio_mean}")
    pd.DataFrame({'mean_ratio': [ratio_mean]}).to_csv(output_file_mean, index=False)
    res.to_csv(output_file, index=False)
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    # global_machine_model_best('WY','Y',150,2676,0.01,1)
    # global_machine_model('WY','Y',150,2676,0.01)
    # global_machine_ckmodel('WY','Y',150,2676)
    # normalization_sum(0.01)
    pass


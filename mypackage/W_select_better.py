def  machine_select_random(readday,outday,crop,name,yname,num1,num2,lowdata,updata,parallel=True, n_jobs=None):#ck与W的混合模型
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from scipy.stats import mode  # 用于计算众数
    from joblib import Parallel, delayed
    import multiprocessing

    pd.set_option('future.no_silent_downcasting', True)
    file_path = f'F:/WaterML/{crop}/{readday}/D{name}.csv'
    out_path = f'F:/WaterML/{crop}/{outday}//Wmix/W_{yname}select.csv'
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        y_true = np.where(y_true == 0, 1e-8, y_true)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def data_processed_one(file_path,yname):#数据读取函数，数据整理部分
        data = pd.read_csv(file_path)#读取文件

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
        #对离散型变量进行编码
        df['fertilizer.with.IR'] = df['fertilizer.with.IR'].replace('YES', 'yes')
        df['mulch'] = df['mulch'].replace('YES', 'yes')
        # fertilizer 列处理
        df['fertilizer'] = df['fertilizer.with.IR'].replace({'yes': 1, 'no': 0})
        df['fertilizer'] = df['fertilizer'].fillna(0).astype(int)
        # mulch 列处理
        df['mulch'] = df['mulch'].replace({'yes': 1,'no': 0})
        df['mulch'] = df['mulch'].fillna(0).astype(int)
        #设置种类
        textures =['clay loam','silt loam','loam','sandy loam','loamy sand']
        #替换其他种类和空白值
        df['soil.texture']=df['soil.texture'].apply(lambda x: x if x in textures else np.nan)
        #加入独热编码
        df = pd.get_dummies(df, columns=['soil.texture'], prefix='texture')
        return df    

    df = data_processed_one(file_path,yname)
    df = data_processed_two(df)

    X = df[['Ypre','Ytem','N.rate','Rate','SOC','PH','fertilizer','mulch']+ [col for col in df.columns if col.startswith('texture_')]]
    y = df[f'{yname}']

    
    # **定义超参数网格**
    param_grid = {
        'n_estimators': [60, 80, 120, 150],
        'max_depth': [4, 6, 8, 10, 12, 14],
        'min_samples_split': [2, 4, 6, 8],  
        'min_samples_leaf': [1, 2, 3, 4, 5]
    }

    # **使用 5 个不同的 random_state 进行超参数调优**
    random_states_for_tuning = [40, 100, 200, 250, 300]
    grid_results = []

    for rs in random_states_for_tuning:
        grid_search = GridSearchCV(RandomForestRegressor(random_state=rs), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X, y)
        
        best_params = grid_search.best_params_
        grid_results.append(best_params)

    # **将数据转换为 DataFrame**
    param_df = pd.DataFrame(grid_results)

    # **计算众数，并在多个众数时选择最小值**
    final_best_params = {}
    for col in param_df.columns:
        mode_values, count = mode(param_df[col], keepdims=True)  # 计算众数

        # 确保 mode_values 是数组
        mode_values = np.atleast_1d(mode_values)

        # 取最小的众数
        final_best_params[col] = min(mode_values)


    # **转换为整数**
    final_best_params = {k: int(v) for k, v in final_best_params.items()}

    # **定义评估单个random_state的内部函数**
    def evaluate_single_random_state(rs, X_data, y_data, best_params):
        """评估单个random_state"""
        kf = KFold(n_splits=5, shuffle=True, random_state=rs)
        fold_results = []

        for train_index, test_index in kf.split(X_data):
            X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
            y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]

            rf = RandomForestRegressor(**best_params, random_state=rs)
            rf.fit(X_train, y_train)

            # 计算评估指标
            y_train_pred = rf.predict(X_train)
            y_test_pred = rf.predict(X_test)

            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
            test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

            fold_results.append([train_mse, test_mse, train_r2, test_r2,
                                train_mae, test_mae, train_mape, test_mape])

        avg_results = np.mean(fold_results, axis=0)
        return [rs] + list(avg_results)

    # **遍历 random_state=1~500，寻找最优 random_state**
    random_states = range(num1, num2)

    if parallel:
        # 确定并行数
        if n_jobs is None:
            n_jobs = min(multiprocessing.cpu_count(), 8)

        print(f"[{yname}] 使用并行模式遍历 {len(random_states)} 个random_state (n_jobs={n_jobs})...")

        # 使用joblib并行
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(evaluate_single_random_state)(rs, X, y, final_best_params)
            for rs in random_states
        )
    else:
        print(f"[{yname}] 使用顺序模式遍历 {len(random_states)} 个random_state...")
        results = []
        for rs in random_states:
            results.append(evaluate_single_random_state(rs, X, y, final_best_params))

    # **存储所有 random_state 结果**
    random_search_results_df = pd.DataFrame(results, columns=[
        'Random State', 'Train MSE', 'Test MSE', 'Train R²', 'Test R²',
        'Train MAE', 'Test MAE', 'Train MAPE', 'Test MAPE'
    ])

    # **创建一个 DataFrame 记录最优超参数**
    best_params_df = pd.DataFrame([final_best_params])
    best_params_df['description'] = 'Best Parameters (Averaged from 6 Random States)'

    # **合并最优超参数和 random_state 评估结果**
    final_results_df = pd.concat([best_params_df, random_search_results_df], ignore_index=True)

    # **保存到 CSV 文件**
    
    final_results_df.to_csv(out_path, index=False, sep=',', float_format='%.3f')


    print(f"已完成 `random_state` 遍历，所有结果已保存到 `{out_path}`，请自行评估最优 `random_state`！")

def  machine_select_random_CH4(readday,outday,name,yname,num1,num2,lowdata,updata):#ck与W的混合模型
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from scipy.stats import mode  # 用于计算众数


    file_path = f'D:/Waterreduce/wheat/{readday}/{name}.csv'

    out_path = f'D:/Waterreduce/wheat/{outday}/Wmix/W_{yname}select.csv'
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        y_true = np.where(y_true == 0, 1e-8, y_true)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def data_processed_one(file_path,yname):#数据读取函数，数据整理部分
        data = pd.read_csv(file_path)#读取文件

        #读取实验组数据
        columns_n = ['ID','VID','soil.texture','PH','SOC','Ypre','Ytem','N.rate','Rate','mulch','fertilizer.with.IR','nir','nctrl',f'{yname}ir']#列名提取
        dfn = data[columns_n].copy()#获取列名数据
        dfn['source'] = 'ir'#标明来源
        dfn = dfn.dropna(subset=[f'{yname}ir'])
        dfn = dfn[(dfn['Rate']>=0.75)&(dfn['Rate']<1)]#提取最优区间
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
        #对离散型变量进行编码
        df['fertilizer.with.IR'] = df['fertilizer.with.IR'].replace('YES', 'yes')
        df['mulch'] = df['mulch'].replace('YES', 'yes')
        # fertilizer 列处理
        df['fertilizer'] = df['fertilizer.with.IR'].replace({'yes': 1, 'no': 0})
        df['fertilizer'] = df['fertilizer'].fillna(0).astype(int)
        # mulch 列处理
        df['mulch'] = df['mulch'].replace({'mulch': 1,'B-mulch': 0,'straw-mulch': 0, 'yes': 1,'no': 0})
        df['mulch'] = df['mulch'].fillna(0).astype(int)
        #设置种类
        textures =['silt loam','loam','clay loam','sandy loam','loamy sand']
        #替换其他种类和空白值
        df['soil.texture']=df['soil.texture'].apply(lambda x: x if x in textures else np.nan)

        #加入独热编码
        df = pd.get_dummies(df, columns=['soil.texture'], prefix='texture')

        return df    

    df = data_processed_one(file_path,yname)
    df = data_processed_two(df)

    X = df[['Ypre','Ytem','N.rate','Rate','SOC','PH','fertilizer','mulch']+ [col for col in df.columns if col.startswith('texture_')]]
    y = df[f'{yname}']

    
    # **定义超参数网格**
    param_grid = {
        'n_estimators': [50, 80, 120, 150],  
        'max_depth': [4, 6, 8, 12, 14],
        'min_samples_split': [2, 4, 6, 8,10],
        'min_samples_leaf': [1, 2, 3,4,5]
    }

    # **使用 5 个不同的 random_state 进行超参数调优**
    random_states_for_tuning = [40, 100, 200]
    grid_results = []

    for rs in random_states_for_tuning:
        grid_search = GridSearchCV(RandomForestRegressor(random_state=rs), param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X, y)
        
        best_params = grid_search.best_params_
        grid_results.append(best_params)

    # **将数据转换为 DataFrame**
    param_df = pd.DataFrame(grid_results)

    # **计算众数，并在多个众数时选择最小值**
    final_best_params = {}
    for col in param_df.columns:
        mode_values, count = mode(param_df[col], keepdims=True)  # 计算众数

        # 确保 mode_values 是数组
        mode_values = np.atleast_1d(mode_values)

        # 取最小的众数
        final_best_params[col] = min(mode_values)


    # **转换为整数**
    final_best_params = {k: int(v) for k, v in final_best_params.items()}

    # **遍历 random_state=1~500，寻找最优 random_state**
    random_states = range(num1, num2)
    results = []


    for rs in random_states:
        # **7:3 训练-测试集划分**
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rs)

        # **使用最优超参数训练随机森林**
        best_rf = RandomForestRegressor(**final_best_params, random_state=rs)
        best_rf.fit(X_train, y_train)

        # **预测**
        y_train_pred = best_rf.predict(X_train)
        y_test_pred = best_rf.predict(X_test)

        # **计算评估指标**
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
        test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

        # **记录当前 random_state 下的结果**
        results.append([rs, train_mse, test_mse, train_r2, test_r2, train_mae, test_mae, train_mape, test_mape])


    # **存储所有 random_state 结果**
    random_search_results_df = pd.DataFrame(results, columns=[
        'Random State', 'Train MSE', 'Test MSE', 'Train R²', 'Test R²',
        'Train MAE', 'Test MAE', 'Train MAPE', 'Test MAPE'
    ])

    # **创建一个 DataFrame 记录最优超参数**
    best_params_df = pd.DataFrame([final_best_params])
    best_params_df['description'] = 'Best Parameters (Averaged from 6 Random States)'

    # **合并最优超参数和 random_state 评估结果**
    final_results_df = pd.concat([best_params_df, random_search_results_df], ignore_index=True)

    # **保存到 CSV 文件**
    
    final_results_df.to_csv(out_path, index=False, sep=',', float_format='%.3f')


    print(f"已完成 `random_state` 遍历，所有结果已保存到 `{out_path}`，请自行评估最优 `random_state`！")


def  machine_select_mix_someone(readday,name,yname,n_random,num):#ck与W的混合模型
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    from sklearn.metrics import mean_absolute_error

    file_path = f'Z:/Python code/data/W/re_data/{readday}/{name}.csv'

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        y_true = np.where(y_true == 0, 1e-8, y_true)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def data_processed_one(file_path,yname):#数据读取函数，数据整理部分
        data = pd.read_csv(file_path)#读取文件
        data=data[data['why'] != 'x']
        data = data.dropna(subset=['IR'])
        data = data.dropna(subset=['IRck'])
        data['Rate']=data['IR']/data['IRck']
        data['SOC']=data['OMC']/1.724
        #读取实验组数据
        columns_n = ['ID','VID','soil.texture','PH','SOC','Ypre','Ytem','N.rate','Rate','crop','mulch','fertilizer.with.IR','nir','nctrl',f'{yname}ir']#列名提取
        dfn = data[columns_n].copy()#获取列名数据
        dfn['source'] = 'ir'#标明来源
        dfn = dfn.dropna(subset=[f'{yname}ir'])
        dfn = dfn[(dfn['Rate']>=0.75)&(dfn['Rate']<1)]#提取最优区间
        #读取对照组数据
        columns_ck = ['ID','VID','soil.texture','PH','SOC','Ypre','Ytem','N.rate','Rate','crop','mulch','fertilizer.with.IR','nir','nctrl',f'{yname}ctrl']
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
        #对离散型变量进行编码
        df['fertilizer.with.IR'] = df['fertilizer.with.IR'].replace('YES', 'yes')
        df['mulch'] = df['mulch'].replace('YES', 'yes')
        # fertilizer 列处理
        df['fertilizer'] = df['fertilizer.with.IR'].replace({'yes': 1, 'no': 0})
        df['fertilizer'] = df['fertilizer'].fillna(0).astype(int)
        # mulch 列处理
        df['mulch'] = df['mulch'].replace({'mulch': 1,'B-mulch': 0,'straw-mulch': 0, 'yes': 1,'no': 0})
        df['mulch'] = df['mulch'].fillna(0).astype(int)
        #设置种类
        textures =['silty clay','silty clay loam', 'clay loam','silty loam','loam','sandy clay loam','sandy loam','loamy sand']
        textures_crop =['maize','wheat','cotton']
        #替换其他种类和空白值
        df['soil.texture']=df['soil.texture'].apply(lambda x: x if x in textures else np.nan)
        df.loc[:, 'crop'] = df['crop'].apply(lambda x: x if x in textures_crop else np.nan)
        #加入独热编码
        df = pd.get_dummies(df, columns=['soil.texture'], prefix='texture')
        df = pd.get_dummies(df, columns=['crop'], prefix='crop_kind')
        return df    

    df = data_processed_one(file_path,yname)
    df = data_processed_two(df)
    X = df[['Ypre','Ytem','N.rate','Rate','SOC','PH','fertilizer','mulch']+ [col for col in df.columns if col.startswith('texture_')]+[col for col in df.columns if col.startswith('crop_kind_')]]
    y = df[f'{yname}']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=num)

    rf = RandomForestRegressor(n_estimators=n_random, random_state=num)
    # 训练模型
    rf.fit(X_train, y_train)
    # 预测测试集
    y_pred = rf.predict(X_test)
    # 使用回归评估函数
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results_df = X_test.copy()  # 复制 X_test，保留测试特征
    results_df['y_test'] = y_test  # 添加真实 y
    results_df['y_pred'] = y_pred  # 添加预测 y
    results_df['mae']= mae
    results_df['mape']= mape
    results_df['mse']= mse
    results_df['r2']= r2
    results_df.to_csv(f'W_{yname}_{num}.csv', index=False)
    print(f'{name}已完成')

if __name__ == "__main__":
     machine_select_random("20251027","20251028",'wheat','N2O','N2O',1,1000,0.60,0.90)
 #   machine_select_random_CH4("2025_02_28",'WCH4', 'CH4',3001,6000)
    #machine_select_mix_someone('WWUE','WUE',3000,1)
 #   machine_select_random_CH4("2025_02_28", "2025_07_28", 'WCH4', 'CH4', 1, 301, 0.70, 1.0)


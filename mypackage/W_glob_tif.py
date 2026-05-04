def csv_to_tif_r(file_name, value_name,rname,crop,model, outday):
    #用于提取tif model=1-----ypred....model=2---------ratio
    import pandas as pd
    import numpy as np
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    from osgeo import gdal
    import os

    # 1. 读取参考 TIF 文件
    reference_tif_path = f'F:/WaterML/{crop}/{outday}/x_data_mask.tif'
    ref_ds = gdal.Open(reference_tif_path)
    geotransform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    x_min = geotransform[0]
    y_max = geotransform[3]
    x_res = geotransform[1]
    y_res = abs(geotransform[5])
    cols = ref_ds.RasterXSize
    rows = ref_ds.RasterYSize
    ref_ds = None
    print('读取参考 TIF 完成')

    # 2. 读取 CSV 文件
    if model == 1:
        csv_path = f'F:/WaterML/{crop}/{outday}/mix_ck/W/{rname}/{file_name}.csv'
    else:
        csv_path = f'F:/WaterML/{crop}/{outday}/ratio/W/{rname}/{file_name}.csv'
    data = pd.read_csv(csv_path)
    x_coords = data['x']  # 经度
    y_coords = data['y']  # 纬度
    values = data[value_name]  # 栅格值
    print('读取 CSV 完成')

    # 3. 初始化栅格数组
    raster = np.full((rows, cols), -9999, dtype=np.float32)

    # 4. 将点数据映射到栅格索引
    for x, y, value in zip(x_coords, y_coords, values):
        col = int(round((x - x_min) / x_res))
        row = int(round((y_max - y) / y_res))

        # 边界检查
        col = max(0, min(cols - 1, col))
        row = max(0, min(rows - 1, row))

        raster[row, col] = value

    # 5. 保存为 GeoTIFF 文件
    if model == 1 :
        output_dir = f'F:/WaterML/{crop}/{outday}/W/Ypred/'
        os.makedirs(output_dir, exist_ok=True)
        output_tif_path = f'{output_dir}/{file_name}_{value_name}.tif'
    else:
        output_dir = f'F:/WaterML/{crop}/{outday}/W/ratio/'
        os.makedirs(output_dir, exist_ok=True)
        output_tif_path = f'{output_dir}/{file_name}_{value_name}.tif'
    transform = Affine(x_res, 0, x_min, 0, -y_res, y_max)
    crs = CRS.from_wkt(projection)  # 提取 CRS

    with rasterio.open(
        output_tif_path,
        'w',
        driver='GTiff',
        height=raster.shape[0],
        width=raster.shape[1],
        count=1,
        dtype='float32',
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(raster, 1)
        dst.nodata = -9999

    print(f"生成的 TIF 文件已保存到: {output_tif_path}")

def csv_to_tif(file_name, value_name, crop,model, outday):
    #用于提取tif model=1-----ypred....model=2---------ratio
    import pandas as pd
    import numpy as np
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    from osgeo import gdal
    import os

    # 1. 读取参考 TIF 文件
    reference_tif_path = f'F:/WaterML/{crop}/{outday}/x_data_mask.tif'
    ref_ds = gdal.Open(reference_tif_path)
    geotransform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    x_min = geotransform[0]
    y_max = geotransform[3]
    x_res = geotransform[1]
    y_res = abs(geotransform[5])
    cols = ref_ds.RasterXSize
    rows = ref_ds.RasterYSize
    ref_ds = None
    print('读取参考 TIF 完成')

    # 2. 读取 CSV 文件
    if model == 1:
        csv_path = f'F:/WaterML/{crop}/{outday}/mix_ck/W/{file_name}.csv'
    else:
        csv_path = f'F:/WaterML/{crop}/{outday}/ratio/W/{file_name}.csv'
    data = pd.read_csv(csv_path)
    x_coords = data['x']  # 经度
    y_coords = data['y']  # 纬度
    values = data[value_name]  # 栅格值
    print('读取 CSV 完成')

    # 3. 初始化栅格数组
    raster = np.full((rows, cols), -9999, dtype=np.float32)

    # 4. 将点数据映射到栅格索引
    for x, y, value in zip(x_coords, y_coords, values):
        col = int(round((x - x_min) / x_res))
        row = int(round((y_max - y) / y_res))

        # 边界检查
        col = max(0, min(cols - 1, col))
        row = max(0, min(rows - 1, row))

        raster[row, col] = value

    # 5. 保存为 GeoTIFF 文件
    if model == 1 :
        output_dir = f'F:/WaterML/{crop}/{outday}/W/Ypred/'
        os.makedirs(output_dir, exist_ok=True)
        output_tif_path = f'{output_dir}/{file_name}_{value_name}.tif'
    else:
        output_dir = f'F:/WaterML/{crop}/{outday}/W/ratio/'
        os.makedirs(output_dir, exist_ok=True)
        output_tif_path = f'{output_dir}/{file_name}_{value_name}.tif'
    transform = Affine(x_res, 0, x_min, 0, -y_res, y_max)
    crs = CRS.from_wkt(projection)  # 提取 CRS

    with rasterio.open(
        output_tif_path,
        'w',
        driver='GTiff',
        height=raster.shape[0],
        width=raster.shape[1],
        count=1,
        dtype='float32',
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(raster, 1)
        dst.nodata = -9999

    print(f"生成的 TIF 文件已保存到: {output_tif_path}")

if __name__ == "__main__":
    csv_to_tif('all','max_column',2,'2025_07_28')

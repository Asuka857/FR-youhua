import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from matplotlib.ticker import FormatStrFormatter, LinearLocator
import matplotlib.font_manager as fm
from PIL import Image
from scipy.interpolate import griddata
import os

def crop_image(image, left):#裁剪图片

    # 使用 crop() 方法进行裁剪
    cropped_image = image.crop((left, 0, 1800, 1800))
    
    # 保存裁剪后的图片
    #cropped_image.save(output_path)

    return cropped_image

def map(tif_file, tif_name,crop, vmin, vmax, title, colorbar_name, save_file,outday,saveday,road,NW,model):
    """
    绘制栅格数据地图，并保存为图片。

    参数:
    tif_file (str): 栅格文件路径（.tif文件）- 支持绝对路径或相对路径
    tif_name (str): 栅格文件名
    vmin (float): 图例棒的下限
    vmax (float): 图例棒的上限
    title (str): 图型名称（标题）
    colorbar_name (str): 图例棒名称
    save_file (str): 保存图片的名称
    """
    # 自定义颜色映射，使用内置 'RdYlBu'
    cmap = 'RdYlGn'

    # 判断是否为绝对路径（包含盘符或以/开头）
    if os.path.isabs(tif_file) or (tif_file and tif_file[0] in '/\\'):
        # 绝对路径：直接使用
        tif_path = f'{tif_file}/{tif_name}.tif'
    else:
        # 相对路径：使用原有逻辑
        tif_path = f'F:/WaterML/{crop}/{outday}/{NW}/{tif_file}/{tif_name}.tif'
    
    # 1. 读取并检查栅格数据
    with rasterio.open(tif_path) as src:
        data = src.read(1)  # 读取第一个波段
        transform = src.transform  # 获取地理坐标信息
        crs = src.crs  # 获取投影信息
        nodata = src.nodata  # 获取 NoData 值
        bounds = src.bounds  # 栅格文件的地理范围

    # 手动设置 NoData 值（如果未正确读取）
    if nodata is None:
        nodata = -9999  # 假设无效值是 -9999

    # 替换 NoData 值为 np.nan
    data = np.where(data == nodata, np.nan, data)

    # 检查数据范围，打印统计信息
    print(tif_name, "Data range:", np.nanmin(data), "to", np.nanmax(data))

    # 如果没有传入vmin和vmax，自动设置
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)


    # 2. 读取底图文件
    shapefile = "E:/USERS/Global/country.shp"  # 替换为你的底图文件路径
    gdf = gpd.read_file(shapefile)  # 用 GeoPandas 读取 Shapefile

    # 3. 设置地图投影和范围
    fig, ax = plt.subplots(
        figsize=(9, 7),  # 设置图像大小
        subplot_kw={'projection': ccrs.PlateCarree()} # 强制布局  # 使用 WGS 1984 投影
    )
    # 确保地图范围与栅格数据匹配
    ax.set_extent([bounds.left, bounds.right, bounds.bottom, bounds.top], crs=ccrs.PlateCarree())

    # 4. 绘制栅格数据
    im = ax.imshow(
        data,
        cmap=cmap,  # 使用颜色映射
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin='upper',
        transform=ccrs.PlateCarree(),  # 使用 PlateCarree 投影
        vmin=vmin,  # 数据范围最小值
        vmax=vmax,  # 数据范围最大值
        interpolation='nearest'  # 最近邻插值，避免数据稀疏
    )

    # 5. 绘制底图（矢量边界）
    gdf.plot(
        ax=ax,
        edgecolor='black',  # 设置边界颜色
        facecolor='none',   # 透明填充
        linewidth=0.6       # 设置边界线宽
    )

    # 手动创建颜色条的轴
    cax = fig.add_axes([0.14, 0.26, 0.75, 0.035])  # [left, bottom, width, height]

    # 添加颜色条
    cbar = plt.colorbar(
        im,
        cax=cax,                # 使用手动创建的轴
        orientation='horizontal'
    )

    cbar.locator=LinearLocator(numticks=5)
    cbar.formatter=FormatStrFormatter('%.2f')
    cbar.update_ticks()
    # 设置颜色条边框的线宽
    cbar.outline.set_linewidth(0.9)  # 调整颜色条边框的粗细

    if model == 1:
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))  # 设置为两位小数

    # 设置全局字体为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    # 设置颜色条刻度属性，刻度朝上
    cbar.ax.tick_params(
        direction='out',    # 刻度线向外
        length=1.7,         # 刻度线的长度
        width=0.8,          # 刻度线的宽度
        labelsize=12,        # 刻度字体大小
        labeltop=True,      # 显示刻度标签在顶部
        labelbottom=False,  # 隐藏底部的刻度标签
        top=True,           # 在颜色条顶部显示刻度
        bottom=False        # 隐藏底部的刻度
    )
    # 获取颜色条的所有刻度标签
    labels = cbar.ax.get_xticklabels()

    # 设置最后一个标签右对齐
    labels[-1].set_ha('right')  # 右对齐

    # 设置第一个标签左对齐
    labels[0].set_ha('left')

    # 设置其余标签左对齐
    for label in labels[1:-1]:
        label.set_ha('center')  # 左对齐


    # 修改颜色条刻度标签字体
    for label in cbar.ax.get_xticklabels():
        label.set_fontname('Times New Roman')

    # 添加颜色条标题（在颜色条上方靠左）
    cbar.ax.text(
        0, 2,  # 设置相对位置
        f'{colorbar_name}', 
        fontsize=15,
        transform=cbar.ax.transAxes,  # 使用颜色条的相对坐标
        va='bottom',                 # 垂直对齐方式
        ha='left',                   # 水平对齐方式
        fontname='Times New Roman'
    )

    # 调整整个图形的边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 设置图形边框的粗细

    # 7. 添加标题
    ax.set_title(title, fontsize=18, fontname='Times New Roman')
    # 定义平均值文件的路径
    # 这里假设文件在同目录下，文件名是 {tif_name}.txt，里面只包含一个数字
    # 如果你的数据在一个总的 Excel/CSV 里，你需要用 pd.read_csv 读取并筛选
    if os.path.isabs(tif_file) or (tif_file and tif_file[0] in '/\\'):
        # 绝对路径：平均值文件在同目录
        avg_file_path = f'{tif_file}/{tif_name}_avg.txt'
    else:
        # 相对路径
        avg_file_path = f'F:/WaterML/{crop}/{outday}/{NW}/{tif_file}/{tif_name}_avg.txt'

    global_avg_val = None

    # 1. 尝试从文件读取
    if os.path.exists(avg_file_path):
        try:
            with open(avg_file_path, 'r') as f:
                content = f.read().strip()
                global_avg_val = float(content)
                print(f"Read global average from file: {global_avg_val}")
        except Exception as e:
            print(f"Error reading average file: {e}")

    # 2. 如果文件不存在，你也可以选择在这里临时计算（如果不需要可以注释掉下面两行）
    if global_avg_val is None:
        global_avg_val = np.nanmean(data)
        print(f"File not found. Calculated mean from raster: {global_avg_val}")

    # 3. 将文字绘制到地图上
    if global_avg_val is not None:
        # 设置显示文本格式
        text_str = f'Global Mean: {global_avg_val:.2f}'

        # 在地图上添加文本
        # x=0.02, y=0.08 是相对坐标 (0-1)，位置大概在左下角（避开南极洲）
        ax.text(
            0.02, 0.25,
            text_str,
            transform=ax.transAxes,  # 使用轴坐标系
            fontsize=10,
            fontname='Times New Roman',
            fontweight='bold',  # 加粗
            color='black',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)  # 添加半透明白色背景，防止看不清
        )
    # 8. 保存或显示地图
    if os.path.isabs(tif_file) or (tif_file and tif_file[0] in '/\\'):
        # 绝对路径：保存到tif_file同目录下的maps子目录
        output_dir = f'{tif_file}/../maps'
    else:
        # 相对路径
        output_dir = f'F:/WaterML/{crop}/{saveday}/{road}/'
    os.makedirs(output_dir, exist_ok=True)
    save_path = f'{output_dir}/{save_file}.png'

    plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 保存地图

def notitle_map(tif_file, tif_name,crop, vmin, vmax, colorbar_name, save_file,outday,saveday,NW):
    """
    绘制栅格数据地图，并保存为图片。
    参数:
    tif_file (str): 栅格文件路径（.tif文件）
    tif_name (str): 栅格文件名
    vmin (float): 图例棒的下限
    vmax (float): 图例棒的上限
    title (str): 图型名称（标题）
    colorbar_name (str): 图例棒名称
    save_file (str): 保存图片的名称
    """
    # 自定义颜色映射，使用内置 'RdYlBu'
    cmap = 'RdYlBu'
    tif_path = f'F:/WaterML/{crop}/{outday}/{NW}/{tif_file}/{tif_name}.tif'
    
    # 1. 读取并检查栅格数据
    with rasterio.open(tif_path) as src:
        data = src.read(1)  # 读取第一个波段
        transform = src.transform  # 获取地理坐标信息
        crs = src.crs  # 获取投影信息
        nodata = src.nodata  # 获取 NoData 值
        bounds = src.bounds  # 栅格文件的地理范围

    # 手动设置 NoData 值（如果未正确读取）
    if nodata is None:
        nodata = -9999  # 假设无效值是 -9999

    # 替换 NoData 值为 np.nan
    data = np.where(data == nodata, np.nan, data)

    # 检查数据范围，打印统计信息
    print("Data range:", np.nanmin(data), "to", np.nanmax(data))

    # 如果没有传入vmin和vmax，自动设置
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    print("Set vmin and vmax:", vmin, vmax)

    # 2. 读取底图文件
    shapefile = ("E:/USERS/Global/country.shp")  # 替换为你的底图文件路径
    gdf = gpd.read_file(shapefile)  # 用 GeoPandas 读取 Shapefile

    # 3. 设置地图投影和范围
    fig, ax = plt.subplots(
        figsize=(9, 7),  # 设置图像大小
        subplot_kw={'projection': ccrs.PlateCarree()} # 强制布局  # 使用 WGS 1984 投影
        
    )

    # 确保地图范围与栅格数据匹配
    ax.set_extent([bounds.left, bounds.right, bounds.bottom, bounds.top], crs=ccrs.PlateCarree())

    # 4. 绘制栅格数据
    im = ax.imshow(
        data,
        cmap=cmap,  # 使用颜色映射
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin='upper',
        transform=ccrs.PlateCarree(),  # 使用 PlateCarree 投影
        vmin=vmin,  # 数据范围最小值
        vmax=vmax,  # 数据范围最大值
        interpolation='nearest'  # 最近邻插值，避免数据稀疏
    )

    # 5. 绘制底图（矢量边界）
    gdf.plot(
        ax=ax,
        edgecolor='black',  # 设置边界颜色
        facecolor='none',   # 透明填充
        linewidth=0.4       # 设置边界线宽
    )

    # 手动创建颜色条的轴
    cax = fig.add_axes([0.14, 0.26, 0.75, 0.035])  # [left, bottom, width, height]

    # 添加颜色条
    cbar = plt.colorbar(
        im,
        cax=cax,                # 使用手动创建的轴
        orientation='horizontal'
    )
        # 自定义颜色条的刻度数量
    tick_count = 5  # 设置刻度数量
    ticks = np.linspace(vmin, vmax, tick_count)  # 根据数据范围生成刻度

    cbar.locator = LinearLocator(numticks=5)
    cbar.formatter = FormatStrFormatter('%.2f')
    cbar.update_ticks()
    # 设置颜色条边框的线宽
    cbar.outline.set_linewidth(0.9)  # 调整颜色条边框的粗细

    # 设置全局字体为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    # 设置颜色条刻度
    cbar.set_ticks(ticks)  # 应用刻度
    # 设置颜色条刻度属性，刻度朝上
    cbar.ax.tick_params(
        direction='out',    # 刻度线向外
        length=1.7,         # 刻度线的长度
        width=0.8,          # 刻度线的宽度
        labelsize=12,        # 刻度字体大小
        labeltop=True,      # 显示刻度标签在顶部
        labelbottom=False,  # 隐藏底部的刻度标签
        top=True,           # 在颜色条顶部显示刻度
        bottom=False        # 隐藏底部的刻度
    )

    # 获取颜色条的所有刻度标签
    labels = cbar.ax.get_xticklabels()

    # 设置最后一个标签右对齐
    labels[-1].set_ha('right')  # 右对齐

    labels[0].set_ha('left')  # 右对齐

    # 设置其余标签左对齐
    for label in labels[1:-1]:
        label.set_ha('center')  # 左对齐


    # 修改颜色条刻度标签字体
    for label in cbar.ax.get_xticklabels():
        label.set_fontname('Times New Roman')

    # 添加颜色条标题（在颜色条上方靠左）
    cbar.ax.text(
        0, 2,  # 设置相对位置
        f'{colorbar_name}', 
        fontsize=15,
        transform=cbar.ax.transAxes,  # 使用颜色条的相对坐标
        va='bottom',                 # 垂直对齐方式
        ha='left',                   # 水平对齐方式
        fontname='Times New Roman'
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 设置图形边框的粗细
    # 7. 添加标题
    #ax.set_title(title, fontsize=9, fontname='Times New Roman')

    # 8. 保存或显示地图
    output_dir = f'F:/Waterreduce/rice/{saveday}/Picture'
    os.makedirs(output_dir, exist_ok=True)
    save_path = f'{output_dir}/{save_file}.png'

    plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 保存地图

def best_ratio_two(saveday):
    # 加载两张图片
    image1_path = f'Z:/NW_global/Newest/pic_data/2025_03_12/Wrate.png'  # 假设图片1的路径
    image2_path = f'Z:/NW_global/Newest/pic_data/{saveday}/Nrate.png'  # 假设图片2的路径

    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # 获取图片的尺寸
    width1, height1 = image1.size
    width2, height2 = image2.size

    # 创建一个新的图片，大小为两张图片水平拼接后的尺寸
    new_width = width1 + width2
    new_height = max(height1, height2)

    # 创建一个空白图片
    new_image = Image.new('RGB', (new_width, new_height))

    # 将两张图片粘贴到新的图片上
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (width1, 0))

    # 使用 matplotlib 来添加文本
    fig, ax = plt.subplots(figsize=(new_width / 100, new_height / 100), dpi=100)

    # 将拼接图像显示到 matplotlib 图形中
    ax.imshow(new_image)
    ax.axis('off')  # 关闭坐标轴

    # 设置字体（使用 matplotlib 自带字体）
    font = fm.FontProperties(family='DejaVu Sans', weight='bold')  # 使用系统的字体

    # 在图片1的右上角添加 "a"
    ax.text(width1 - 175, 170, 'a', color='black', fontsize=80, fontproperties=font, weight='bold')

    # 在图片2的右上角添加 "B"
    ax.text(width1 + width2 - 175, 180, 'b', color='black', fontsize=80, fontproperties=font, weight='bold')

    # 保存拼接后的图片
    output_dir = f'Z:/NW_global/NW_all_pic/{saveday}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/Bestrate.png'

    # 保存图像
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)

    # 关闭图形以释放内存
    plt.close(fig)

def six_pic(saveday, NW, pathname, DW, xname,savename):#W全球拼图
    # 定义图像的文件名列表
    image_files = ['Y', 'ET', 'WUE', 'N2O', 'CH4', 'GHG']
    
    # 加载图片
    images = []
    for image_file in image_files:
        image_path = f'Z:/NW_global/Newest/pic_data/{saveday}/{NW}/{pathname}/{DW}_{image_file}_{xname}.png'
        image = Image.open(image_path)
        images.append(image)

    # 获取每张图片的宽度和高度
    widths, heights = zip(*(i.size for i in images))
    h = 150
    # 计算拼接后的新图像大小
    new_width = sum(widths[:3])  # 上面三张图的总宽度
    new_height = max(heights[:3]) + max(heights[4:])+h+150  # 上下两行的高度之和

    # 创建一个空白图像
    new_image = Image.new('RGB', (new_width, new_height),color='white')

    # 将图片粘贴到新的空白图像上
    x_offset = 0
    for i in range(3):
        new_image.paste(images[i], (x_offset, 100))  # 上面一行
        x_offset += widths[i]

    x_offset = 0  # 第二排开始从左侧对齐
    for i in range(3, 6):
        new_image.paste(images[i], (x_offset, heights[0]+h))  # 下面一行
        x_offset += widths[i]

    # 使用 matplotlib 来添加文本
    fig, ax = plt.subplots(figsize=(new_width / 100, new_height / 100), dpi=100)
    ax.imshow(new_image)
    ax.axis('off')  # 关闭坐标轴

    # 设置字体（使用 matplotlib 自带字体）
    font = fm.FontProperties(family='DejaVu Sans', weight='bold')  # 使用系统的字体

    # 添加标签
    labels = ['a', 'b', 'c', 'd', 'e', 'f']
    x = 4100
    y = 350+100
    f = new_width/4
    e = 100
    x_offsets = [x, widths[0] + x, widths[0] + widths[1] + x,
                 x, widths[0] + x, widths[0] + widths[1] + x]
    y_offsets = [y, y, y, heights[0] +y+h-e , heights[0] +y+h-e , heights[0] + y+h-e ]

    # 添加文本标签
    for i in range(6):
        ax.text(x_offsets[i], y_offsets[i], labels[i], color='black', fontsize=80, fontproperties=font, weight='bold')

    # 保存拼接后的图片
    output_dir = f'Z:/NW_global/NW_all_pic/{saveday}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/{NW}_{savename}.png'
    
    # 保存图像
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)

    # 关闭图形以释放内存
    plt.close(fig)

def seven_pic(saveday, NW, pathname, DW, xname, savename):  # N全球拼图
    # 定义图像的文件名列表，删除 CO2，改为新的排列
    image_files = ['Y', 'ET', 'WUE','NUE', 'N2O', 'CH4', 'GHG']
    
    # 加载图片
    images = []
    for image_file in image_files:
        image_path = f'Z:/NW_global/Newest/pic_data/{saveday}/{NW}/{pathname}/{DW}_{image_file}_{xname}.png'
        image = Image.open(image_path)
        images.append(image)

    # 获取每张图片的宽度和高度
    widths, heights = zip(*(i.size for i in images))
    h = 150  # 设置上下排间距
    # 计算拼接后的新图像大小
    new_width = sum(widths[:3])  # 上面三张图的总宽度
    new_height = max(heights[:3]) + max(heights[3:6]) + heights[6] + h * 2 + 150  # 三行的高度

    # 创建一个空白图像
    new_image = Image.new('RGB', (new_width, new_height), color='white')

    # 将图片粘贴到新的空白图像上
    x_offset = 0
    # 第一排：Yield, ET, WUE
    for i in range(3):
        new_image.paste(images[i], (x_offset, 100))  # 上面一行
        x_offset += widths[i]

    x_offset = 0  # 第二排开始从左侧对齐
    # 第二排：GHG, N2O, CH4
    for i in range(3, 6):
        new_image.paste(images[i], (x_offset, heights[0] + h))  # 下面一行
        x_offset += widths[i]

    # 第三排：NUE
    new_image.paste(images[6], ((new_width - widths[3]) // 2, heights[0] + heights[1] + h * 2))  # 中间对齐

    # 使用 matplotlib 来添加文本
    fig, ax = plt.subplots(figsize=(new_width / 100, new_height / 100), dpi=100)
    ax.imshow(new_image)
    ax.axis('off')  # 关闭坐标轴

    # 设置字体（使用 matplotlib 自带字体）
    font = fm.FontProperties(family='DejaVu Sans', weight='bold')  # 使用系统的字体

    # 添加标签
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']  # 新标签去掉 'h'
    x = 4100
    y = 350 + 100
    e = 100
    # x_offsets的计算，保证每一行的标签位置正确
    x_offsets = [x, widths[0] + x, widths[0] + widths[1] + x,  # 第一排
                 x, widths[0] + x, widths[0] + widths[1] + x,  # 第二排
                 (new_width - widths[3]) // 2 + x]  # 第三排，NUE在中间

    # y_offsets的计算，第二排和第三排的标签加上一个适当的间距
    y_offsets = [y, y, y,  # 第一排
                 heights[0] + y + h - e, heights[0] + y + h - e, heights[0] + y + h - e,  # 第二排
                 heights[0] + heights[1] + h + y + h - e]  # 第三排（NUE）

    # 添加文本标签
    for i in range(7):  # 只有7个标签
        ax.text(x_offsets[i], y_offsets[i], labels[i], color='black', fontsize=80, fontproperties=font, weight='bold')

    # 保存拼接后的图片
    output_dir = f'Z:/NW_global/NW_all_pic/{saveday}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/{NW}_{savename}.png'
    
    # 保存图像
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)

    # 关闭图形以释放内存
    plt.close(fig)

def five_bili(saveday, NW, pathname, xname, savename):#W比例拼图 
    # 定义图像的文件名列表
    image_files = ['Y', 'ET', 'WUE', 'N2O', 'CH4']
    
    # 加载图片
    images = []
    for image_file in image_files:
        image_path = f'Z:/NW_global/Newest/pic_data/{saveday}/{NW}/{pathname}/{image_file}_{xname}.png'
        image = Image.open(image_path)

        if image_file not in ['N2O', 'Y']:
            image = crop_image(image, 420)  # 修改此值来裁剪不同的区域
        
        images.append(image)

    bai = 40#图形右侧留白
    # 获取每张图片的宽度和高度
    widths, heights = zip(*(i.size for i in images))
    h = 20
    # 计算拼接后的新图像大小
    new_width = sum(widths[:3])+bai  # 上面三张图的总宽度
    new_height = max(heights[:3]) + max(heights[3:]) + h + 75  # 上下两行的高度之和

    # 创建一个空白图像
    new_image = Image.new('RGB', (new_width, new_height), color='white')

    # 将图片粘贴到新的空白图像上
    x_offset = 0
    for i in range(3):
        new_image.paste(images[i], (x_offset, 50))  # 上面一行
        x_offset += widths[i]

    x_offset = (new_width - sum(widths[3:])) // 2  # 使下面两张图居中
    for i in range(3, 5):
        new_image.paste(images[i], (x_offset, heights[0] + h))  # 下面一行
        x_offset += widths[i]

    # 使用 matplotlib 来添加文本
    fig, ax = plt.subplots(figsize=(new_width / 100, new_height / 100), dpi=100)
    ax.imshow(new_image)
    ax.axis('off')  # 关闭坐标轴

    # 设置字体（使用 matplotlib 自带字体）
    font = fm.FontProperties(family='DejaVu Sans', weight='bold')  # 使用系统的字体

    # 添加标签
    labels = ['a', 'b', 'c', 'd', 'e']
    x = 1550
    y = 220 + 50
    e = 50
    e2 = 420
    # 上面一行标签对应 a, b, c
    x_offsets = [x, widths[0] + x - e2, widths[0] + widths[1] + x - e2]  # 上面三张图标签
    y_offsets = [y, y, y]  # 上面一行标签都放在y位置

    # 下面一行标签对应 d, e
    x_offsets += [(new_width - bai - widths[3]- widths[4])/2 + x, (new_width - bai - widths[3]- widths[4])/2 +widths[3]+x-e2]  # 下排两张图标签
    y_offsets += [heights[0] + y + h - e, heights[0] + y + h - e]  # 下面一行标签的y位置

    # 添加文本标签
    for i in range(5):
        ax.text(x_offsets[i], y_offsets[i], labels[i], color='black', fontsize=60, fontproperties=font, weight='bold')

        # 保存拼接后的图片
    output_dir = f'Z:/NW_global/NW_all_pic/{saveday}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/{NW}_{savename}.png'
    
    # 保存图像
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)

    # 关闭图形以释放内存
    plt.close(fig)

def six_bili(saveday, NW, pathname, xname, savename):#N比例拼图
    # 定义图像的文件名列表，新增 NUE
    image_files = ['Y', 'ET', 'WUE', 'NUE','N2O', 'CH4' ]
    
    # 加载图片
    images = []
    for image_file in image_files:
        image_path = f'Z:/NW_global/Newest/pic_data/{saveday}/{NW}/{pathname}/{image_file}_{xname}.png'
        image = Image.open(image_path)

        # 如果不是 N2O 或 Y，则裁剪左侧部分
        if image_file not in ['NUE', 'Y']:
            image = crop_image(image, 420)  # 修改此值来裁剪不同的区域
        
        images.append(image)

    # 获取每张图片的宽度和高度
    widths, heights = zip(*(i.size for i in images))
    h = 20  # 设置上下排间距
    bai = 40 
    # 计算拼接后的新图像大小
    new_width = sum(widths[:3]) + bai  # 上面四张图的总宽度
    new_height = max(heights[:3]) + max(heights[3:]) + h + 75  # 上下两行的高度之和

    # 创建一个空白图像
    new_image = Image.new('RGB', (new_width, new_height), color='white')

    # 将图片粘贴到新的空白图像上
    x_offset = 0
    for i in range(3):
        new_image.paste(images[i], (x_offset, 50))  # 上面一行
        x_offset += widths[i]

    x_offset = 0  # 第二排开始从左侧对齐
    for i in range(3, 6):
        new_image.paste(images[i], (x_offset, heights[0] + h))  # 下面一行
        x_offset += widths[i]

    # 使用 matplotlib 来添加文本
    fig, ax = plt.subplots(figsize=(new_width / 100, new_height / 100), dpi=100)
    ax.imshow(new_image)
    ax.axis('off')  # 关闭坐标轴

    # 设置字体（使用 matplotlib 自带字体）
    font = fm.FontProperties(family='DejaVu Sans', weight='bold')  # 使用系统的字体

    # 添加标签
    labels = ['a', 'b', 'c', 'd', 'e', 'f']  # 添加新的标签 h
    x = 1550
    y = 220 + 50
    e = 50
    e2 = 420
    # x_offsets的计算，保证每一行的标签位置正确
    x_offsets = [x, widths[0] + x-e2, widths[0] + widths[1] + x-e2, 
                 x, widths[0] + x-e2, widths[0] + widths[1] + x-e2]
    # y_offsets的计算，第二排的标签加上一个适当的间距
    y_offsets = [y, y, y, heights[0] + y + h - e, heights[0] + y + h - e, heights[0] + y + h - e]

    # 添加文本标签
    for i in range(6):
        ax.text(x_offsets[i], y_offsets[i], labels[i], color='black', fontsize=60, fontproperties=font, weight='bold')

    # 保存拼接后的图片
    output_dir = f'Z:/NW_global/NW_all_pic/{saveday}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/{NW}_{savename}.png'
    
    # 保存图像
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)

    # 关闭图形以释放内存
    plt.close(fig)

if __name__ == "__main__":
    #five_bili(saveday ='2025_02_28',NW = 'W',pathname= 'Wbili',xname = 'bili',savename = 'bili')
    #six_bili(saveday ='2025_02_28',NW = 'N',pathname= 'Nbili',xname = 'bili',savename = 'bili')
    #six_pic(saveday='2025_02_28',NW = 'W',pathname= 'Ypred',DW = 'W',xname = 'Ypred',savename = 'Ypred')
    #seven_pic(saveday='2025_02_28',NW = 'N',pathname= 'Ypred',DW = 'D',xname = 'Ypred',savename = 'Ypred')
    map(tif_file='ratio',
                        tif_name='N2O_max_column_ratio',
                        vmin=0.6,
                        vmax=1.3,
                        title='N2O',
                        colorbar_name='Ratio',
                        save_file=f'N2O_Y_ratio',
                        outday= '20251028',
                        saveday = '20251028',
                        NW = 'W',
                        road= 'W/ratio/adapt',
                        model = 2)


def comparison_map_3x1(data_optimal, data_ck, data_change, indicator, output_dir,
                       vmin=None, vmax=None, vmin_change=None, vmax_change=None):
    """
    生成3x1纵向布局的专业对比地图（使用与map函数相同的风格）

    参数:
        data_optimal: DataFrame, 包含x, y, value列（最优情景数据）
        data_ck: DataFrame, 包含x, y, value列（对照组数据）
        data_change: DataFrame, 包含x, y, value列（变化数据）
        indicator: 指标名称
        output_dir: 输出目录
        vmin, vmax: 数据范围（最优和对照组）
        vmin_change, vmax_change: 变化数据范围
    """
    # 设置全局字体
    plt.rcParams['font.family'] = 'Times New Roman'

    # 指标名称映射
    indicator_names = {
        'Y': ('Yield', 'kg/ha'),
        'ET': ('Evapotranspiration', 'mm'),
        'WUE': ('Water Use Efficiency', 'kg/m³'),
        'CH4': ('CH4 Emission', 'kg/ha'),
        'N2O': ('N2O Emission', 'kg/ha'),
        'GWP': ('Global Warming Potential', 'kg CO2-eq/ha')
    }
    title_name, colorbar_name = indicator_names.get(indicator, (indicator, indicator))

    # 读取底图
    shapefile = "E:/USERS/Global/country.shp"
    gdf = gpd.read_file(shapefile)

    # 检测是否跨越180度经线
    all_x = np.concatenate([data_optimal['x'].values, data_ck['x'].values, data_change['x'].values])
    all_y = np.concatenate([data_optimal['y'].values, data_ck['y'].values, data_change['y'].values])
    x_min, x_max = all_x.min(), all_x.max()
    crosses_dateline = x_min < 0 and x_max > 0 and (x_max - x_min) > 180

    print(f"  [调试] {indicator}: 经度范围 [{x_min:.2f}, {x_max:.2f}], 跨越180度经线: {crosses_dateline}")

    # 复制数据以避免修改原始数据
    data_optimal = data_optimal.copy()
    data_ck = data_ck.copy()
    data_change = data_change.copy()

    # 如果跨越180度经线，将负经度转换为正经度（0-360系统）
    if crosses_dateline:
        print(f"  [处理] 检测到跨越180度经线，将坐标转换为0-360系统")

        def convert_longitude(x):
            """将-180到180的经度转换为0到360"""
            return np.where(x < 0, x + 360, x)

        data_optimal['x'] = convert_longitude(data_optimal['x'].values)
        data_ck['x'] = convert_longitude(data_ck['x'].values)
        data_change['x'] = convert_longitude(data_change['x'].values)

        # 重新计算范围
        all_x = np.concatenate([data_optimal['x'].values, data_ck['x'].values, data_change['x'].values])
        x_min, x_max = all_x.min(), all_x.max()

        # 转换底图坐标（使用不同的投影）
        proj = ccrs.PlateCarree(central_longitude=180)
        transform = ccrs.PlateCarree()
    else:
        proj = ccrs.PlateCarree()
        transform = ccrs.PlateCarree()

    y_min, y_max = all_y.min(), all_y.max()

    # 添加缓冲区
    x_buffer = (x_max - x_min) * 0.02
    y_buffer = (y_max - y_min) * 0.02
    bounds = [x_min - x_buffer, x_max + x_buffer, y_min - y_buffer, y_max + y_buffer]

    # 设置数据范围
    if vmin is None:
        vmin = min(data_optimal['value'].min(), data_ck['value'].min())
    if vmax is None:
        vmax = max(data_optimal['value'].max(), data_ck['value'].max())
    if vmin_change is None:
        vmin_change = -max(abs(data_change['value'].min()), abs(data_change['value'].max()))
    if vmax_change is None:
        vmax_change = max(abs(data_change['value'].min()), abs(data_change['value'].max()))

    # 创建3x1布局
    fig, axes = plt.subplots(3, 1, figsize=(9, 21),
                             subplot_kw={'projection': proj})

    # 绘图数据配置
    plot_data = [
        (data_optimal, vmin, vmax, 'RdYlGn', f'{indicator} - Optimal'),
        (data_ck, vmin, vmax, 'RdYlGn', f'{indicator} - Control'),
        (data_change, vmin_change, vmax_change, 'RdBu_r', f'{indicator} - Change')
    ]

    # 使用scatter绘制点数据（适合离散点，避免插值问题）
    images = []
    for idx, (data, v_min, v_max, cmap, sub_title) in enumerate(plot_data):
        ax = axes[idx]
        ax.set_extent(bounds, crs=transform)

        # 使用scatter绘制离散点
        sc = ax.scatter(
            data['x'].values, data['y'].values,
            c=data['value'].values,
            cmap=cmap, s=1, alpha=0.8,
            vmin=v_min, vmax=v_max,
            transform=transform
        )
        images.append(sc)

        # 绘制底图
        gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.6, transform=transform)

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        ax.set_title(sub_title, fontsize=18, fontname='Times New Roman')

    # 调整布局
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.12)

    # 计算颜色条位置（基于子图实际位置）
    for idx, ax in enumerate(axes):
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0, pos.y0 - 0.035, pos.width, 0.012])

        cbar = plt.colorbar(images[idx], cax=cax, orientation='horizontal')
        cbar.locator = LinearLocator(numticks=5)
        cbar.formatter = FormatStrFormatter('%.2f')
        cbar.update_ticks()
        cbar.outline.set_linewidth(0.9)

        # 与 map 函数一致的颜色条设置
        cbar.ax.tick_params(
            direction='out', length=1.7, width=0.8, labelsize=12,
            labeltop=True, labelbottom=False, top=True, bottom=False
        )

        # 刻度标签对齐
        labels = cbar.ax.get_xticklabels()
        if len(labels) > 0:
            labels[0].set_ha('left')
            labels[-1].set_ha('right')
            for label in labels[1:-1]:
                label.set_ha('center')

        for label in cbar.ax.get_xticklabels():
            label.set_fontname('Times New Roman')

        # 颜色条标题
        cbar.ax.text(0, 2.5, colorbar_name, fontsize=15,
                     transform=cbar.ax.transAxes, va='bottom', ha='left',
                     fontname='Times New Roman')

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    map_file = f'{output_dir}/{indicator}_comparison_map.png'
    plt.savefig(map_file, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"  对比地图已保存: {map_file}")
    return map_file
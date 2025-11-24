import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER


def judge_lat_grid(lat, unit):
    return np.floor(lat / unit).astype(int)


def judge_lon_grid(lon, unit):
    return np.floor(lon / unit).astype(int)


def Generate_griddata_B(file_path):
    ws_data = pd.read_csv(file_path)
    # 限定经纬度范围
    ws_data = ws_data[
        (ws_data['lat'] >= 22.5) & (ws_data['lat'] < 29.625) &
        (ws_data['lon'] >= 114.75) & (ws_data['lon'] < 121.875)
        ].reset_index(drop=True)

    lat_num = 57
    lon_num = 57
    listall1 = np.zeros((lat_num, lon_num))
    for i in range(len(ws_data['lon'])):

        lat_temp = int(judge_lat_grid(ws_data['lat'].iloc[i] - 22.5, 0.125))
        lon_temp = int(judge_lon_grid(ws_data['lon'].iloc[i] - 114.75, 0.125))

        listall1[lat_temp, lon_temp] = ws_data['ws_mean'].iloc[i]
    lat2 = np.arange(22.5, 29.625, 0.125)
    lon2 = np.arange(114.75, 121.875, 0.125)
    lon2d, lat2d = np.meshgrid(lon2, lat2)
    lon2d = lon2d[:57, :57]
    lat2d = lat2d[:57, :57]
    listall2 = np.column_stack((lon2d.flatten(), lat2d.flatten(), listall1.flatten()))
    return listall2


def cov_to_csv(ws, csv_path):
    lat = np.arange(22.5, 29.625, 0.125)
    lon = np.arange(114.75, 121.875, 0.125)
    lon, lat = np.meshgrid(lon, lat)
    lon = lon[:57, :57]
    lat = lat[:57, :57]
    df = pd.DataFrame(np.column_stack((lon.flatten(), lat.flatten(), ws.flatten())), columns=['lon', 'lat', 'ws_mean'])  # 'ws_mean' 是风速的列名
    df.to_csv(csv_path, index=False)  # 保存到文件，不保存行索引


def draw(listall2, img_path):
    # 创建图像和坐标轴，使用Cartopy的PlateCarree投影
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # 设置坐标轴标签和刻度
    ax.set_xlabel('Longitude', fontweight='bold')
    ax.set_ylabel('Latitude', fontweight='bold')
    ax.tick_params(axis='both', direction='out', which='both')
    ax.xaxis.set_major_formatter('{:.0f}°'.format)
    ax.yaxis.set_major_formatter('{:.0f}°'.format)

    # 修改显示范围为新的坐标范围
    ax.set_xlim(114.75, 121.75)
    ax.set_ylim(22.5, 29.5)
    ax.set_title('Wind Speed with Map', fontweight='bold')

    # ------------------------- 绘制风速数据 -------------------------
    lon2d, lat2d = np.meshgrid(np.arange(114.75, 121.75, 0.125)[:57],
                               np.arange(22.5, 29.5, 0.125)[:57])
    lim = np.arange(0, 20, 1)
    cmap = plt.cm.get_cmap('jet', len(lim))
    drawlist = np.reshape(listall2[:, 2], (57, 57))

    masked_data = np.ma.masked_where(drawlist == 0, drawlist)
    contour = ax.contourf(lon2d, lat2d, masked_data, levels=lim, cmap=cmap)
    cbar = fig.colorbar(contour, ax=ax, ticks=lim, orientation='vertical')
    cbar.ax.set_yticklabels(lim)
    cbar.ax.set_ylabel('Wind Speed (m/s)', fontweight='bold')

    # ------------------------- 绘制地图 -------------------------
    ax.add_feature(cfeature.LAND.with_scale('50m'), zorder=10)  # 陆地图层位于最上面
    # ------------------------- 显示经纬度网格 -------------------------
    ax.set_xticks(np.arange(114.75, 121.75, 1.0), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(22.5, 29.5, 0.5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    ax.tick_params(axis='both', labelsize=10, direction='out', length=8)

    plt.draw()
    plt.savefig(img_path, dpi=300)
    plt.close()
    # plt.show()

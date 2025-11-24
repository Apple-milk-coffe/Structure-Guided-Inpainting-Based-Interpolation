import pandas as pd
import numpy as np
from scipy.spatial import KDTree

from config import Generate_griddata_B, draw

def prepare_buoy_data(buoy_csv_path,
                      grid_lat_start=22.5,
                      grid_lon_start=114.75,
                      delta=0.125,
                      grid_shape=(57, 57)):
    df = pd.read_csv(buoy_csv_path, encoding='GBK')
    lats = df['纬度'].values
    lons = df['经度'].values
    obs   = df['f10'].values

    buoy_coords = []
    buoy_obs    = []
    for lat, lon, v in zip(lats, lons, obs):
        i = int(round((lat - grid_lat_start) / delta))
        j = int(round((lon - grid_lon_start) / delta))
        if 0 <= i < grid_shape[0] and 0 <= j < grid_shape[1]:
            buoy_coords.append((i, j))
            buoy_obs.append(v)
    return buoy_coords, buoy_obs


def apply_local_buoy_correction(X_interp, buoy_coords, buoy_obs,
                                radius=2, gamma=0.8):
    X = X_interp.copy()
    nrows, ncols = X.shape
    for (ib, jb), val in zip(buoy_coords, buoy_obs):
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                i, j = ib + di, jb + dj
                if 0 <= i < nrows and 0 <= j < ncols:
                    d = np.hypot(di, dj)
                    if d <= radius:
                        w = (1 - d / radius)
                        X[i, j] = (1 - gamma * w) * X[i, j] + (gamma * w) * val
    return X


def AR_self_interpolate_with_buoy(B, buoy_csv_path, output_file,
                                  k_self=4, radius=2, gamma=0.8):

    lon, lat, ws = B[:,0], B[:,1], B[:,2].copy()
    coords = np.column_stack((lon, lat))
    missing = np.isnan(ws) | (ws == 0)
    filled_idx = np.where(~missing)[0]

    while np.any(missing):
        if filled_idx.size == 0:
            print("没有可用的已填补点，退出 AR 阶段。")
            break

        tree = KDTree(coords[filled_idx])
        new_filled = []

        for idx in np.where(missing)[0]:
            pt = coords[idx:idx + 1]
            d, i = tree.query(pt, k=min(k_self, filled_idx.size))

            # 安全归一化
            w = 1.0 / (d[0] + 1e-6)
            w_sum = w.sum()
            if w_sum <= 0:
                w = np.ones_like(w) / w.size
            else:
                w /= w_sum

            ws[idx] = np.dot(w, ws[filled_idx[i[0]]])
            new_filled.append(idx)

        if not new_filled:
            break
        filled_idx = np.where(~np.isnan(ws) & (ws != 0))[0]
        missing = np.isnan(ws) | (ws == 0)

    # 转为二维网格
    grid = ws.reshape(57, 57)

    # 读取浮标并修正
    buoy_coords, buoy_obs = prepare_buoy_data(buoy_csv_path,
                                              grid_lat_start=22.5,
                                              grid_lon_start=114.75,
                                              delta=0.125,
                                              grid_shape=(57,57))
    grid_corrected = apply_local_buoy_correction(grid, buoy_coords, buoy_obs,
                                                 radius=radius, gamma=gamma)

    # 保存结果
    lon2 = np.arange(114.75, 114.75+57*0.125, 0.125)
    lat2 = np.arange(22.5,   22.5+57*0.125,   0.125)
    lon2d, lat2d = np.meshgrid(lon2, lat2)
    out = np.column_stack((lon2d.flatten(), lat2d.flatten(), grid_corrected.flatten()))
    df = pd.DataFrame(out, columns=['lon','lat','ws_mean'])
    df.to_csv(output_file, index=False)
    print(f"AR+Buoy 插值完成，结果保存至 {output_file}")

    return out


if __name__ == '__main__':
    file_pathB = 'dataset/2021_02_01.csv'
    buoy_path = 'buoy/浮标平均_映射_2021_02_01.csv'
    output_csv = 'output/AR_IDW/2021_02_01.csv'
    output_png = 'output/AR_IDW/AR_IDW.png'

    B = Generate_griddata_B(file_pathB)
    Z = AR_self_interpolate_with_buoy(B, buoy_path, output_csv)
    draw(B, output_png)

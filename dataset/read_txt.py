import pandas as pd

file_path = 'Merge_day_2021_02_01_(N545E100135)_QS_A_B_C_CFO_3E_WindSpeed_data_(125)_new.txt'
df = pd.read_csv(file_path, sep='\s+', header=None, names=['lon', 'lat', 'ws_mean'])
lon_min, lon_max = 114.75, 121.75
lat_min, lat_max = 22.5, 29.5
filtered_df = df[(df['lon'] >= lon_min) & (df['lon'] <= lon_max) &
                 (df['lat'] >= lat_min) & (df['lat'] <= lat_max)]
output_path = '2021_02_01.csv'
filtered_df.to_csv(output_path, index=False)

print(f"文件已保存为 {output_path}")

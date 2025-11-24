import pandas as pd

def process_day(buoy_latlon, buoy_value, date_str):
    file_a = buoy_latlon
    file_b = buoy_value

    df_a = pd.read_csv(file_a, encoding='gbk')
    df_b = pd.read_csv(file_b, encoding='gbk')
    df_b.rename(columns={'III': '站号'}, inplace=True)
    df_b['dt'] = pd.to_datetime(df_b['dt'], format='%Y/%m/%d %H:%M:%S')

    # 筛选日期
    target_date = pd.to_datetime(date_str).date()
    df_b_filtered = df_b[df_b['dt'].dt.date == target_date]

    # 合并经纬度
    df_a_selected = df_a[['站号', '经度', '纬度']]
    df_merged = pd.merge(df_b_filtered, df_a_selected, on='站号', how='left')

    # 筛选、清洗
    df_merged_selected = df_merged[['f10', 'd10', '经度', '纬度']]
    df_valid = df_merged_selected[(df_merged_selected['f10'] != 9999) & (df_merged_selected['d10'] != 9999)]

    if df_valid.empty:
        print(f"{date_str} 无有效数据，跳过。")
        return

    # 计算平均值
    df_grouped = df_valid.groupby(['经度', '纬度'], as_index=False).agg({
        'f10': 'mean',
        'd10': 'mean'
    })
    df_grouped['f10'] = df_grouped['f10'] * 0.1
    out_path = f'浮标平均_映射_{date_str}.csv'
    df_grouped.to_csv(out_path, index=False, encoding='gbk')
    print(f'{date_str} 完成，保存为 {out_path}')


import sys
import os
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import datetime
import csv
import numpy as np
import pandas as pd
from modules import data as modules_data

import matplotlib.pyplot as plt

def customized_specific_period_col(df):
    customized_cols_infos = []
    
    # 紀錄原先的cols
    original_cols = set(df.columns)
    # 紀錄新的cols
    modified_cols = set(df.columns)
    # 篩選多出來的cols
    new_cols = list(modified_cols - original_cols)
    return df, customized_cols_infos, new_cols

def top_and_down(df, lookahead=8):
    tops = []
    bottoms = []

    is_current_max_high_appear = False
    max_high = df['High'].iloc[0]  # 初始设置为第一根K线的最高价
    max_datetime = df['datetime'].iloc[0]

    is_current_min_low_appear = False
    min_low = df['Low'].iloc[0]  # 初始设置为第一根K线的最低价
    min_datetime = df['datetime'].iloc[0]
    
    with open(f'check.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['datetime', 'Open', 'High', 'Low', 'Close', 'Min_Low', 'Min_Low_datetime', 'Max_High', 'Max_High_datetime'])

        for i in range(len(df) - lookahead):
            current_low = df['Low'].iloc[i]
            current_high = df['High'].iloc[i]
            current_datetime = df['datetime'].iloc[i]

            if current_low < min_low:
                future_lows = df['Low'].iloc[i+1:i+lookahead+1]
                if future_lows.min() >= current_low:
                    min_low = current_low
                    min_datetime = current_datetime
                    min_index = i
                    is_current_min_low_appear = True
                    # 没有被更新，则确认底峰
                    bottoms.append((min_index, min_low, min_datetime))
                    

            if current_high > max_high:
                # 检查未来 lookahead 根 K 线是否没有更新最高值
                future_highs = df['High'].iloc[i+1:i+lookahead+1]
                if future_highs.max() <= current_high:
                    max_high = current_high
                    max_datetime = current_datetime
                    max_index = i
                    is_current_max_high_appear = True
                    # 没有被更新，则确认顶峰
                    tops.append((max_index, max_high, max_datetime))
            
            if is_current_max_high_appear or is_current_min_low_appear:
                is_current_max_high_appear = False
                is_current_min_low_appear = False

                max_high = df['High'].iloc[i]
                max_datetime = df['datetime'].iloc[i]
                max_index = i
                min_low = df['Low'].iloc[i]
                min_datetime = df['datetime'].iloc[i]
                min_index = i

            

            writer.writerow([
                df['datetime'].iloc[i],
                df['Open'].iloc[i], 
                df['High'].iloc[i], 
                df['Low'].iloc[i], 
                df['Close'].iloc[i], 
                min_low, 
                min_datetime, 
                max_high, 
                max_datetime
            ])

            
            
    return tops, bottoms


symbol = "ETHUSDT"
interval = "15m"
look_back = 1 #使用回看n根數據
epochs = 150
batch_size = 128
total_klines = 1000
get_local_file_name = ''

input_model_infos = []

end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

end_time_seconds = end_time / 1000
end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# end_time_string = "2024-08-07 15:59:59"
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
df, customized_cols_infos, new_cols = customized_specific_period_col(df)

# 将 'datetime' 列转换为 datetime 类型
df['datetime'] = pd.to_datetime(df['datetime'])

# 将 'datetime' 列转换为 UTC+8
df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')

# 如果需要移除时区信息，可以使用 .dt.tz_localize(None)
df['datetime'] = df['datetime'].dt.tz_localize(None)

df.fillna(0, inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.interpolate(method='linear', inplace=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns

# 检查 NaN 值的数量
nan_counts = df[numeric_cols].isna().sum()
print("NaN Counts:\n", nan_counts)

inf_counts = np.isinf(df[numeric_cols]).sum()
print("Infinity Counts:\n", inf_counts)

total_nan_inf = nan_counts + inf_counts
print("Total NaN and Infinity Counts:\n", total_nan_inf)

# 检查所有列中 NaN 值的行
nan_volume_rows = df[df[numeric_cols].isna().any(axis=1)]
inf_volume_rows = df[np.isinf(df[numeric_cols]).any(axis=1)]
print("Rows with NaN column:\n", nan_volume_rows)
print("Rows with inf column:\n", inf_volume_rows)
df.reset_index(drop=True, inplace=True)

top, bottom = top_and_down(df, 4)

print(bottom)

# 繪製價格的折線圖（包含High和Low）
plt.figure(figsize=(14, 7))
plt.plot(df['datetime'], df['High'], label='High Price', color='blue', alpha=0.6)
plt.plot(df['datetime'], df['Low'], label='Low Price', color='orange', alpha=0.6)

# 標記頂峰
top_indices = [t[0] for t in top]
top_values = [t[1] for t in top]
plt.scatter(df['datetime'].iloc[top_indices], top_values, color='red', marker='^', label='Tops')

# 標記底峰
bottom_indices = [b[0] for b in bottom]
bottom_values = [b[1] for b in bottom]
plt.scatter(df['datetime'].iloc[bottom_indices], bottom_values, color='green', marker='v', label='Bottoms')

# 添加標籤和標題
plt.xlabel('DateTime')
plt.ylabel('Price')
plt.title('Price with Tops and Bottoms')
plt.legend()

# 顯示圖表
plt.show()
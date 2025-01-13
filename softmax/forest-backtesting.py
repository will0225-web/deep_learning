import sys
import os

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
import tensorflow as tf
import datetime
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import pytz

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import time_convert as time_convert
from joblib import load

def calculate_percentage_change(order_price, closing_price):
    return ((closing_price - order_price) / order_price) * 100

# 特徵欄位
# cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'MACD', 'Signal', 'Hist', 'MACD_Cross', 'Up Trend', 'Down Trend', 'Super Trend', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band']
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'RSI_6', 'RSI_12', 'RSI_24', 'MACD', 'Signal', 'Hist', 'MACD_Stronger', 'MACD_Cross', 'Up Trend', 'Down Trend', 'Super Trend', 'ATR', 'SMA_12', 'SMA_21', 'SMA_27', 'SMA_55', 'SMA_200', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band', 'Stochastic_Oscillator_9', 'Stochastic_Oscillator_14', 'Stochastic_Oscillator_21', 'EMA_12', 'EMA_26', 'EMA_50']
symbol = "ETHUSDT"
interval = "15m"
look_back = 12 #使用回看n根數據

# 當前
# end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# end_time_seconds = end_time / 1000
# end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
# end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# 特定
end_time_string = "2023-11-30 23:59:59"

# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 4000, cols, is_need_save_original_data=False, is_read_local=False, is_need_calculated=True)

# 再拿掉前後無參考性資料
df = df[1000:]
df.reset_index(drop=True, inplace=True)
df.to_csv('test.csv')

# 使用StandardScaler對數據進行縮放
# scaler = RobustScaler()
# scaled_data = scaler.fit_transform(df[cols])

scaled_data = df[cols]

macd_cross_indices = df.index[df['MACD_Cross'] != 0].tolist()


X, y = [], []
for index in macd_cross_indices:
    start_index = index - (look_back - 1)
    if start_index >= 0:  # 确保开始索引不为负数
        X.append(scaled_data[start_index:index + 1])
        y.append(df.iloc[index]['Target'])

y = [element.astype(int) for element in y]
X, y = np.array(X), np.array(y)


# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))

# 定義上層目錄的路徑
parent_path = os.path.join(current_path, '..')
time = int(datetime.datetime.timestamp(datetime.datetime.now()))
model_directory = f'{parent_path}/trained_models/grid_forest/1702021835_trainAcc-0.8496_accuracy-0.2526_f1:0.2272_n-300_maxd-20_minsl-10_minss-5'
model_filename = f'{model_directory}/softmax_trainAcc-0.8496_accuracy-0.2526_f1:0.2272_n-300_maxd-20_minsl-10_minss-5.joblib'
model = load(model_filename)

reshapre_X = X.reshape(X.shape[0], -1)
# 使用模型進行預測
predicted = model.predict(reshapre_X)

# backtesting
win_count = 0
loss_count = 0
threshold = 0.015  # 停損
at_least_profit = 0.005

macd_zero_line = 0
rsi_exit_delta = 12  # RSI退出阈值的偏移量

long_order_price = 0.0
short_order_price = 0.0

upper_bound = 0.0
lower_bound = 0.0

order_price = 0.0

percentage = 0.0

best_profit = 0

time = int(datetime.datetime.timestamp(datetime.datetime.now()))
output_module_name = f'{time}:softmax'

trend = 0 # 0沒有單, 1:long, 2:short

fee = 0.0004

new_df = df[look_back:]
new_df.reset_index(drop=True, inplace=True)
new_macd_cross_indices = new_df.index[new_df['MACD_Cross'] != 0].tolist()
scaled_data = scaled_data[look_back:]

current_order_predict = -1

hold_count = 0
hold_count_limit = 60


max_rsi = float('-inf')  # 初始化为负无穷大
min_rsi = float('inf')   # 初始化为正无穷大

# 创建一个与scaled_data同长度的结果数组，初始化为-1
predicted_full = np.full(len(scaled_data), -1)
real_y = np.full(len(scaled_data), -1)
# 遍历new_macd_cross_indices，并将预测结果映射到predicted_full
for idx, macd_idx in enumerate(new_macd_cross_indices):
    predicted_full[macd_idx] = predicted[idx]
    real_y[macd_idx] = y[idx]

for i in range(0, len(scaled_data) - 1):
    inverse_point = scaled_data.iloc[i]
    pre_inverse_point = scaled_data.iloc[i - 1]
    
    original_close_price = inverse_point[cols.index('Close')]
    original_low_price = inverse_point[cols.index('Low')]
    original_high_price = inverse_point[cols.index('High')]
    
    original_rsi = inverse_point[cols.index('RSI')]
    pre_rsi = pre_inverse_point[cols.index('RSI')]

    original_macd_hist = inverse_point[cols.index('Hist')]
    pre_macd_hist = pre_inverse_point[cols.index('Hist')]

    macd_cross = inverse_point[cols.index('MACD_Cross')]

    target = real_y[i]
    predicted_target = predicted_full[i]

    

    current_price = original_close_price
    

    if trend == 1 or trend == 2:
        current_profit = ((original_close_price - order_price) / order_price) - fee if trend == 1 else ((order_price - original_close_price)) / order_price - fee
        
        current_high_profit = ((original_high_price - order_price) / order_price) - fee if trend == 1 else ((order_price - original_high_price) / order_price) - fee
        current_low_profit = ((original_low_price - order_price) / order_price) - fee if trend == 1 else ((order_price - original_low_price) / order_price) - fee

        best_profit = max(best_profit, current_profit)

        # 更新最大或最小RSI
        if trend == 1:
            max_rsi = max(max_rsi, original_rsi)
        elif trend == 2:
            min_rsi = min(min_rsi, original_rsi)

        # 停損
        if (trend == 1 and original_low_price <= lower_bound) or (trend == 2 and original_high_price >= upper_bound):
            real_profit = current_low_profit if trend == 1 else current_high_profit
            if real_profit >= 0:
                win_count += 1
            else:
                loss_count += 1
            percentage += max(real_profit, -threshold) if real_profit < 0 else real_profit
            trend = 0
            current_order_predict = -1

        # 反向停損出場, 使用low或high出場
        if (trend == 1 and current_order_predict == 1 and original_high_price >= upper_bound) or (trend == 2 and current_order_predict == 0 and original_low_price <= lower_bound):
            real_profit = current_high_profit if trend == 1 else current_low_profit
            if real_profit >= 0:
                win_count += 1
            else:
                loss_count += 1
            percentage += min(real_profit, threshold) if real_profit >= 0 else real_profit
            trend = 0
            current_order_predict = -1
            
        # rsi 出場
        if (trend == 1 and original_rsi <= max_rsi - rsi_exit_delta) or (trend == 2 and original_rsi >= min_rsi + rsi_exit_delta) and current_profit >= at_least_profit:
            if current_profit >= 0:
                win_count += 1
            else:
                loss_count += 1
            percentage += current_profit
            trend = 0
            current_order_predict = -1

        # 到達最多hold的長度出場
        if hold_count == hold_count_limit and trend != 0:
            if current_profit >= 0:
                win_count += 1
            else:
                loss_count += 1
            percentage += current_profit
            trend = 0
            current_order_predict = -1
        else:
            hold_count += 1

        # macd狀態改變
        if macd_cross != 0:
            if trend == 1 and (predicted_target == 1 or predicted_target == 3 or predicted_target == 4 or predicted_target == 6 or predicted_target == 8 or predicted_target == 11):
                max_rsi = original_rsi
                current_order_predict = predicted_target
                upper_bound = current_price * (1 + threshold)
                lower_bound = current_price * (1 - threshold)
                hold_count = 0
            elif trend == 2 and (predicted_target == 0 or predicted_target == 2 or predicted_target == 5 or predicted_target == 7 or predicted_target == 9 or predicted_target == 10):
                min_rsi = original_rsi
                current_order_predict = predicted_target
                upper_bound = current_price * (1 + threshold)
                lower_bound = current_price * (1 - threshold)
                hold_count = 0
            elif trend != 0:
                if current_order_predict == 6 or current_order_predict == 7:
                    if current_profit >= 0:
                        win_count += 1
                    else:
                        loss_count += 1
                    percentage += current_profit
                    trend = 0
                    current_order_predict = -1
                else:
                    if predicted_target == 4 or predicted_target == 5:
                        if current_profit >= 0:
                            win_count += 1
                        else:
                            loss_count += 1
                        percentage += current_profit
                        trend = 0
                        current_order_predict = -1
                    elif (predicted_target == 0 or predicted_target == 1 or predicted_target == 2 or predicted_target == 3) and (current_order_predict != 4 or current_order_predict != 5):
                        if current_profit >= 0:
                            win_count += 1
                        else:
                            loss_count += 1
                        percentage += current_profit
                        trend = 0
                        current_order_predict = -1

    if trend == 0 and macd_cross != 0:
        best_profit = 0.0
        trend = 0
        max_rsi = float('-inf')  # 初始化为负无穷大
        min_rsi = float('inf')   # 初始化为正无穷大
        hold_count = 0
        current_order_predict = predicted_target 
        # -1就等於沒有交叉就不會進單

        if predicted_target == 1 or predicted_target == 3 or predicted_target == 4 or predicted_target == 6 or predicted_target == 8 or predicted_target == 11:
            trend = 1
            order_price = current_price
            upper_bound = current_price * (1 + threshold)
            lower_bound = current_price * (1 - threshold)
            max_rsi = original_rsi
        elif predicted_target == 0 or predicted_target == 2 or predicted_target == 5 or predicted_target == 7 or predicted_target == 9 or predicted_target == 10:
            trend = 2
            order_price = current_price
            upper_bound = current_price * (1 + threshold)
            lower_bound = current_price * (1 - threshold)
            min_rsi = original_rsi

    with open(f'{model_directory}/{output_module_name}_output.csv', 'a', newline='') as file:  # 'a'表示附加模式，這樣數據將會被添加到文件而不是覆蓋它
        writer = csv.writer(file)
        writer.writerow([original_close_price, original_high_price, original_low_price, original_rsi, min_rsi, max_rsi, win_count, loss_count, percentage, best_profit, trend, original_macd_hist, predicted_target, target])


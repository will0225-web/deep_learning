import sys
import os

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import datetime
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import custom_model_fit_indicators as custom_model_fit_indicators

def calculate_percentage_change(order_price, closing_price):
    return ((closing_price - order_price) / order_price) * 100

# 定义参数
# cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'MACD', 'Signal', 'Hist', 'Up Trend', 'Down Trend', 'Super Trend', 'ATR', 'SMA_27', 'SMA_55', 'SMA_200', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band']
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'MACD', 'Signal', 'Hist', 'Up Trend', 'Down Trend', 'Super Trend','TR' , 'ATR', 'SMA_27', 'SMA_55', 'SMA_200', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band']
symbol = "ETHUSDT"
interval = "5m"
look_back = 144 #使用回看n根數據

# 當前
# end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# end_time_seconds = end_time / 1000
# end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
# end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
# 特定
end_time_string = "2023-09-30 23:59:59"
# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000

# 获取 K 线数据
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 10000, cols, is_need_save_original_data=False, is_read_local=False, is_need_calculated=True)

# 再拿掉前後無參考性資料
df = df[1000:]
# first_na_index = df['Target'].isna().idxmax()
# # 如果存在非空值，刪除從該索引到 DataFrame 結尾的所有行
# if pd.isna(df['Target'][first_na_index]):
#     df = df.loc[:first_na_index - 1]

# 載入已經訓練好的模型
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.join(current_path, '..')
trained_model_name = f'{parent_path}/trained_models/1701206433_sigmoid_loss-1.0223_accuracy-0.5534_auc-0.5486_f1_score-0.4107'
model = tf.keras.models.load_model(trained_model_name, custom_objects={'f1_score': custom_model_fit_indicators.f1_score})

# 使用scaler對數據進行縮放
scaler = RobustScaler()
# scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[cols])

# 考慮到look_back，我們從第look_back根開始
X, y = [], []
for i in range(look_back, len(scaled_data) + 1):
    X.append(scaled_data[i - look_back:i])
    y.append(df.iloc[i - 1][['Target']].values)
    
# y = [element.astype(int) for element in y]
input_datas, y = np.array(X), np.array(y)

# 評估模型
# loss, accuracy, _, _, _ = model.evaluate(input_datas, y)
# loss, accuracy = model.evaluate(input_datas, y)
# print(f"Test Loss: {loss:.4f}")
# print(f"Test Accuracy: {accuracy:.4f}")

predict_datas = np.array(input_datas)
# 使用模型進行預測
predicted = model.predict(predict_datas)

# predictions = []
win_count = 0
loss_count = 0
threshold = 0.01  # 1%

is_long_order = False
is_short_order = False

upper_bound = 0.0
lower_bound = 0.0

order_price = 0.0

# total_profit = 0.0
total_percentage = 0.0

trend = -1

fee = 0.0004

time = int(datetime.datetime.timestamp(datetime.datetime.now()))
output_module_name = f'{time}:sigmoid'
for i in range(0, len(predicted)):
    input_data = input_datas[i]

    p = predicted[i]
    predicted_target = np.argmax(p)
    confidence = np.max(p)
    target = y[i]

    single_data_point = np.array([input_data[-1]])
    inverse_point = scaler.inverse_transform(single_data_point)
    original_close_price = inverse_point[0][cols.index('Close')]
    original_high_price = inverse_point[0][cols.index('High')]
    original_low_price = inverse_point[0][cols.index('Low')]

    profit = 0
    percentage = 0
    

    # 判断交易状态
    if is_long_order or is_short_order:
        
        # 检查是否达到止损点或止盈点
        if (is_long_order and (original_low_price <= lower_bound or original_close_price >= upper_bound or p < 0.5)) or \
           (is_short_order and (original_high_price >= upper_bound or original_close_price <= lower_bound or p >= 0.5)):
            if (is_long_order and (original_low_price <= lower_bound)) or (is_short_order and (original_high_price >= upper_bound)):
                loss_count += 1
                total_percentage += (-threshold - fee)
            else:
                if is_long_order:
                    profit = original_close_price - order_price 
                elif is_short_order:
                    profit = -(original_close_price - order_price)

                percentage = profit / order_price if is_long_order else profit / original_close_price


                if profit > 0:
                    win_count += 1
                else:
                    loss_count += 1

                if percentage > - threshold:
                    total_percentage += (percentage - fee)
                else:
                    total_percentage += (-threshold - fee)

            # 重置交易状态
            is_long_order = False
            is_short_order = False
            upper_bound = 0.0
            lower_bound = 0.0
    else:
        # 基于模型预测开启新的交易
        if p >= 0.5:
            is_long_order = True
            is_short_order = False

            trend = 1
        else:
            is_short_order = True
            is_long_order = False

            trend = 0

        order_price = original_close_price
        upper_bound = order_price * (1 + threshold)
        lower_bound = order_price * (1 - threshold)

    with open(f'{trained_model_name}/{output_module_name}_output.csv', 'a', newline='') as file:  # 'a'表示附加模式，這樣數據將會被添加到文件而不是覆蓋它
        writer = csv.writer(file)
        # dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)  # 從UNIX時間戳在UTC時區創建datetime對象

        # 轉換時區
        # tz = pytz.timezone('Asia/Taipei')  # 這是台灣的時區
        # localized_dt = dt.astimezone(tz)  # 使用astimezone方法轉換到台灣的時區

        # formatted_date = localized_dt.strftime('%Y-%m-%d %H:%M')
        writer.writerow([percentage, original_close_price, order_price, upper_bound, lower_bound, win_count, loss_count, total_percentage, trend, p, predicted_target, target, confidence])



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
from modules import custom_model_fit_indicators as custom_model_fit_indicators

def get_original_data(cols, symbol, interval):
    # 特定
    end_time_string = "2023-10-31 23:59:59"

    # 轉毫秒
    # end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
    # Step 1: 獲取數據
    df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 35040, cols, is_need_save_original_data=False, is_read_local=True, is_need_calculated=True)
    # 再拿掉前後無參考性資料
    df = df[1000:-300]
    df.reset_index(drop=True, inplace=True)

    return df


def calculate_percentage_change(order_price, closing_price):
    return ((closing_price - order_price) / order_price)

def multiclass_f1_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_f1_score(y_true, y_pred)

# 特徵欄位
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'ATR', 'SMA_55', 'SMA_200', 'Middle Band', 'Upper Band', 'Lower Band', 'Up Trend', 'Down Trend', 'Super Trend', 'EMA_50']
symbol = "ETHUSDT"
interval = "15m"
look_back = 288 #使用回看n根數據

original_df = get_original_data(cols, symbol, interval)


# 當前
end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

end_time_seconds = end_time / 1000
end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# 特定
# end_time_string = "2023-07-01 00:00:00"

# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 10000, cols, is_need_save_original_data=False, is_read_local=False, is_need_calculated=True)

# 再拿掉前後無參考性資料
df = df[1000:]
df.reset_index(drop=True, inplace=True)

# 載入已經訓練好的模型
# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.join(current_path, '..')
trained_model_name = f'{parent_path}/trained_models/1703354974_softmax_loss-1.5954_accuracy-0.4115_f1-0.2120_no_weight_best'
model = tf.keras.models.load_model(trained_model_name, custom_objects={'multiclass_f1_score': multiclass_f1_score})

# 使用StandardScaler對數據進行縮放
scaler = RobustScaler()
scaler.fit(original_df[cols])

scaled_data = scaler.fit_transform(df[cols])

# 創建X, y數據集
X, y = [], []
for i in range(look_back, len(scaled_data) + 1):
    X.append(scaled_data[i - look_back:i])
    y.append(df.iloc[i - 1][['Target_0', 'Target_1', 'Target_2']].values)

y = [element.astype(int) for element in y]
X, y = np.array(X), np.array(y)

loss, accuracy, f1 = model.evaluate(X, y)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1: {f1:.4f}")

# 使用模型進行預測
predict_datas = np.array(X)
predicted = model.predict(predict_datas)

# backtesting
win_count = 0
loss_count = 0
threshold = 0.01
win_threshold = 0.02
hold_count = 15


is_long_order = False
long_order_price = 0.0
is_short_order = False
short_order_price = 0.0

upper_bound = 0.0
lower_bound = 0.0

order_price = 0.0

percentage = 0.0

best_profit = 0

time = int(datetime.datetime.timestamp(datetime.datetime.now()))
output_module_name = f'{time}:softmax'

trend = 0

fee = 0.0004

current_hold_count = 0

for i in range(0, len(predicted)):
    input_data = X[i]

    p = predicted[i]
    predicted_target = np.argmax(p)
    confidence = np.max(predicted[i])
    target = df.iloc[i]['Target']

    t = np.array(input_data[-1])  # 取最後一筆資料
    t = scaler.inverse_transform([t])
    t = t[0][cols.index('Close')]

    single_data_point = np.array([input_data[-1]])
    # 使用scaler.inverse_transform()方法恢复该点到原始尺度
    inverse_point = scaler.inverse_transform(single_data_point)
    
    # 从结果中获取Close的原始价格
    original_close_price = inverse_point[0][cols.index('Close')]
    original_low_price = inverse_point[0][cols.index('Low')]
    original_high_price = inverse_point[0][cols.index('High')]
    
    if is_long_order or is_short_order:
        current_hold_count += 1
        if trend == 1:
            current_percentage = calculate_percentage_change(order_price, original_close_price) - fee
            if best_profit <= current_percentage:
                best_profit = current_percentage
            
            if original_high_price >= upper_bound:
                trend = 0
                is_long_order = False
                is_short_order = False

                percentage += calculate_percentage_change(order_price, upper_bound) - fee

                upper_bound = 0.0
                lower_bound = 0.0

                win_count +=1

                order_price = 0.0
            elif original_low_price <= lower_bound:
                trend = 0
                is_long_order = False
                is_short_order = False

                percentage += calculate_percentage_change(order_price, lower_bound) - fee

                upper_bound = 0.0
                lower_bound = 0.0

                loss_count +=1
                order_price = 0.0
            elif predicted_target == 1:
                current_hold_count = 0
                upper_bound = original_close_price * (1 + win_threshold)
                lower_bound = max(lower_bound, original_close_price * (1 - threshold))
            elif predicted_target == 2 or predicted_target == 3 or current_hold_count >= hold_count:
                trend = 0
                is_long_order = False
                is_short_order = False

                upper_bound = 0.0
                lower_bound = 0.0

                if current_percentage > 0:
                    win_count += 1
                else:
                    loss_count +=1

                percentage += current_percentage
                order_price = 0.0

        elif trend == 2: 
            current_percentage = (-calculate_percentage_change(order_price, original_close_price)) - fee
            if best_profit <= current_percentage:
                best_profit = current_percentage

            if original_high_price >= upper_bound:
                trend = 0
                is_long_order = False
                is_short_order = False

                percentage -= calculate_percentage_change(order_price, upper_bound) - fee

                upper_bound = 0.0
                lower_bound = 0.0

                loss_count +=1

                order_price = 0.0
            elif original_low_price <= lower_bound:
                trend = 0
                is_long_order = False
                is_short_order = False

                percentage -= calculate_percentage_change(order_price, lower_bound) - fee

                upper_bound = 0.0
                lower_bound = 0.0

                win_count +=1

                order_price = 0.0
            elif predicted_target == 2:
                current_hold_count = 0
                upper_bound = min(upper_bound, original_close_price * (1 + win_threshold))
                lower_bound = original_close_price * (1 - threshold)
            elif predicted_target == 1 or predicted_target == 4 or current_hold_count >= hold_count:
                trend = 0
                is_long_order = False
                is_short_order = False

                upper_bound = 0.0
                lower_bound = 0.0
                if current_percentage > 0:
                    win_count += 1
                else:
                    loss_count +=1

                percentage += current_percentage
                order_price = 0.0
    else:
        best_profit = 0.0
        trend = 0
        current_price = original_close_price
        current_hold_count = 0
        if predicted_target == 1:
            trend = 1
            is_long_order = True
            upper_bound = current_price * (1 + win_threshold)
            lower_bound = current_price * (1 - threshold)
            order_price = current_price
        elif predicted_target == 2:
            trend = 2
            is_short_order = True
            upper_bound = current_price * (1 + threshold)
            lower_bound = current_price * (1 - win_threshold)
            order_price = current_price
        elif predicted_target == 3:
            trend = 2
            is_short_order = True
            upper_bound = current_price * (1 + threshold)
            lower_bound = current_price * (1 - threshold)
            order_price = current_price
        elif predicted_target == 4:
            trend = 1
            is_long_order = True
            upper_bound = current_price * (1 + threshold)
            lower_bound = current_price * (1 - threshold)
            order_price = current_price

    with open(f'{trained_model_name}/{output_module_name}_output.csv', 'a', newline='') as file:  # 'a'表示附加模式，這樣數據將會被添加到文件而不是覆蓋它
        writer = csv.writer(file)

        writer.writerow([order_price, original_close_price, original_high_price, upper_bound, original_low_price, lower_bound, win_count, loss_count, percentage, best_profit, trend, predicted_target, target, confidence])


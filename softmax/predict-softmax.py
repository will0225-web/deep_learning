import sys
import os

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
import tensorflow as tf
import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import time_convert as time_convert
from modules import model as model_process

def load_model(dir_name):
    # 獲取當前文件的絕對路徑
    current_path = os.path.abspath(os.path.dirname(__file__))
    parent_path = os.path.join(current_path, '..')
    trained_model_name = f'{parent_path}/trained_models/test_predict/{dir_name}'
    model = tf.keras.models.load_model(trained_model_name)

    return model

def all_predict(df, cols, look_back, specific_datetime_str):
    # 載入已經訓練好的模型
    model = load_model("1699286719_softmax_loss-0.3679_accuracy-0.8357_all")

    # 使用MinMaxScaler對數據進行縮放
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[cols])
    
    specific_datetime_utc = pd.to_datetime(specific_datetime_str) - pd.Timedelta(hours=8)

    # 轉換特定的datetime到UNIX時間戳（毫秒）
    specific_datetime_utc_timestamp = int(specific_datetime_utc.timestamp() * 1000)

    # 根據特定的UTC時間戳過濾df
    df = df[df['timestamp'] <= specific_datetime_utc_timestamp]
    df.reset_index(drop=True, inplace=True)
    # 找到最後一個滿足macd_cross != 0的索引
    last_macd_cross_index = df[df['MACD_Stronger'] != 0].index[-1]
    # 刪除這個位置之後的所有資料
    df = df[:last_macd_cross_index + 1]

    only_backtesting_count = 1000
    # 只取最後5000個input_datas
    input_datas = [scaled_data[i-look_back:i] for i in range(look_back, len(scaled_data) + 1)]
    input_datas = input_datas[-only_backtesting_count:]

    X = np.array(input_datas)

    # 進行預測
    predicted = model.predict(X)

    # 根據預測結果判斷趨勢
    predicted_target = np.argmax(predicted[-1])

    if predicted_target == 0:
        trend = "無效"
    elif predicted_target == 1:
        trend = "下跌"
    elif predicted_target == 2:
        trend = "上涨"
    return trend

def bb_predict(df, cols, look_back, specific_datetime_str):
    # 載入已經訓練好的模型
    model = load_model("1699304468_softmax_loss-0.4010_accuracy-0.8359_bb")

    # 使用MinMaxScaler對數據進行縮放
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[cols])

    specific_datetime_utc = pd.to_datetime(specific_datetime_str) - pd.Timedelta(hours=8)

    # 轉換特定的datetime到UNIX時間戳（毫秒）
    specific_datetime_utc_timestamp = int(specific_datetime_utc.timestamp() * 1000)

    # 根據特定的UTC時間戳過濾df
    df = df[df['timestamp'] <= specific_datetime_utc_timestamp]
    df.reset_index(drop=True, inplace=True)
    # 找到最後一個滿足macd_cross != 0的索引
    last_macd_cross_index = df[df['MACD_Stronger'] != 0].index[-1]
    # 刪除這個位置之後的所有資料
    df = df[:last_macd_cross_index + 1]

    only_backtesting_count = 1000
    # 只取最後5000個input_datas
    input_datas = [scaled_data[i-look_back:i] for i in range(look_back, len(scaled_data) + 1)]
    input_datas = input_datas[-only_backtesting_count:]
    X = np.array(input_datas)

    # 進行預測
    predicted = model.predict(X)

    # 根據預測結果判斷趨勢
    predicted_target = np.argmax(predicted[-1])

    if predicted_target == 0:
        trend = "無效"
    elif predicted_target == 1:
        trend = "下跌"
    elif predicted_target == 2:
        trend = "上涨"

    return trend

def supertrend_predict(df, cols, look_back, specific_datetime_str):
    # 載入已經訓練好的模型
    model = load_model("1699297072_softmax_loss-0.4175_accuracy-0.9090_supertrend")

    # 使用MinMaxScaler對數據進行縮放
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[cols])

    specific_datetime_utc = pd.to_datetime(specific_datetime_str) - pd.Timedelta(hours=8)

    # 轉換特定的datetime到UNIX時間戳（毫秒）
    specific_datetime_utc_timestamp = int(specific_datetime_utc.timestamp() * 1000)

    # 根據特定的UTC時間戳過濾df
    df = df[df['timestamp'] <= specific_datetime_utc_timestamp]
    df.reset_index(drop=True, inplace=True)
    # 找到最後一個滿足macd_cross != 0的索引
    last_macd_cross_index = df[df['MACD_Stronger'] != 0].index[-1]
    # 刪除這個位置之後的所有資料
    df = df[:last_macd_cross_index + 1]

    only_backtesting_count = 1000
    # 只取最後5000個input_datas
    input_datas = [scaled_data[i-look_back:i] for i in range(look_back, len(scaled_data) + 1)]
    input_datas = input_datas[-only_backtesting_count:]
    X = np.array(input_datas)

    # 進行預測
    predicted = model.predict(X)

    # 根據預測結果判斷趨勢
    predicted_target = np.argmax(predicted[-1])

    if predicted_target == 0:
        trend = "無效"
    elif predicted_target == 1:
        trend = "下跌"
    elif predicted_target == 2:
        trend = "上涨"

    return trend

def macd_predict(df, cols, look_back, specific_datetime_str):
    # macd
    df = df[df['MACD_Stronger'] != 0]

    specific_datetime_utc = pd.to_datetime(specific_datetime_str) - pd.Timedelta(hours=8)
    # 轉換特定的datetime到UNIX時間戳（毫秒）
    specific_datetime_utc_timestamp = int(specific_datetime_utc.timestamp() * 1000)

    # 根據特定的UTC時間戳過濾df
    df = df[df['timestamp'] <= specific_datetime_utc_timestamp]
    df.reset_index(drop=True, inplace=True)
    
    # 載入已經訓練好的模型
    model = load_model("1699252021_softmax_loss-0.5352_accuracy-0.8004_macd")

    # 使用MinMaxScaler對數據進行縮放
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[cols])

    # 考慮到look_back，我們從第look_back根開始
    input_datas = [scaled_data[i-look_back:i] for i in range(look_back, len(scaled_data) + 1)]

    X = np.array(input_datas)

    # 進行預測
    predicted = model.predict(X)

    # 根據預測結果判斷趨勢
    predicted_target = np.argmax(predicted[-1])

    if predicted_target == 0:
        trend = "無效"
    elif predicted_target == 1:
        trend = "下跌"
    elif predicted_target == 2:
        trend = "上涨"

    return trend

# 特徵欄位
cols = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Signal', 'Hist', 'Up Trend', 'Down Trend', 'Super Trend', 'TR', 'ATR', 'SMA', 'MACD_Cross', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band']
symbol = "ETHUSDT"
interval = "15m"

# 當前
end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

end_time_seconds = end_time / 1000
end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# 特定
# end_time_string = "2023-08-31 23:59:59"

# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 15000, cols, is_need_save_original_data=False, is_read_local=False, is_need_calculated=True)

# 再拿掉前後無參考性資料
df = df[1000:]
df.reset_index(drop=True, inplace=True)

# 將 df 的 'datetime' 列轉換為 datetime 型別
df['datetime'] = pd.to_datetime(df['datetime'])

# 轉換datetime列為時間戳
df['timestamp'] = (df['datetime'].astype(np.int64) // 10**6)
predict_time = '2023-09-19 17:30:00'
# 總體
all_trend = all_predict(df, ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "Signal", "Hist", "Up Trend", "Down Trend", "Super Trend", "TR", "ATR", "SMA_27", "SMA_55", "SMA_200", "MACD_Cross", "MACD_Stronger", "kline_color", "Middle Band", "Upper Band", "Lower Band"], 288, predict_time)
# BB
bb_trend = bb_predict(df, ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "Signal", "Hist", "SMA_27", "SMA_55", "SMA_200", "kline_color", "Middle Band", "Upper Band", "Lower Band"], 288, predict_time)
# supertrend
supertrend_trend = supertrend_predict(df, ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "Signal", "Hist", "Up Trend", "Down Trend", "Super Trend", "TR", "ATR", "SMA_27", "SMA_55", "SMA_200", "MACD_Cross", "kline_color"], 288, predict_time)
# macd
macd_trend = macd_predict(df, ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "Signal", "Hist", "Up Trend", "Down Trend", "Super Trend", "TR", "ATR", "SMA_27", "SMA_55", "SMA_200", "MACD_Cross", "MACD_Stronger", "kline_color", "Middle Band", "Upper Band", "Lower Band"], 288, predict_time)

print(all_trend, bb_trend, supertrend_trend, macd_trend)
import sys
import os

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
import tensorflow as tf
import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import time_convert as time_convert
from modules import model as model_process
from keras import backend as K

def calculate_percentage_change(order_price, closing_price):
    return ((closing_price - order_price) / order_price) * 100


def f1_score(y_true, y_pred):
    """计算 F1 分数"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# 特徵欄位
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'MACD', 'Signal', 'Hist', 'Up Trend', 'Down Trend', 'Super Trend', 'TR', 'ATR', 'SMA_27', 'SMA_55', 'SMA_200', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band']
# cols = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "Signal", "Hist", "Up Trend", "Down Trend", "Super Trend", "TR", "ATR", "SMA_27", "SMA_55", "SMA_200", "MACD_Cross", "MACD_Stronger", "kline_color", "Middle Band", "Upper Band", "Lower Band"]
symbol = "ETHUSDT"
interval = "15m"
look_back = 144 #使用回看n根數據

# 當前
end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

end_time_seconds = end_time / 1000
end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# 特定
end_time_string = "2023-09-30 23:59:59"

# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 10000, cols, is_need_save_original_data=False, is_read_local=False, is_need_calculated=True)

# 再拿掉前後無參考性資料
df = df[1000:]
first_na_index = df['Target'].isna().idxmax()
# 如果存在非空值，刪除從該索引到 DataFrame 結尾的所有行
if pd.isna(df['Target'][first_na_index]):
    df = df.loc[:first_na_index - 1]

df.reset_index(drop=True, inplace=True)

# 載入已經訓練好的模型
# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.join(current_path, '..')
trained_model_name = f'{parent_path}/trained_models/1700119429_sigmoid_loss-0.7008_accuracy-0.6329'
model = tf.keras.models.load_model(trained_model_name, custom_objects={'f1_score': f1_score})

# only_backtesting_count = 5000

# # 你已經有 scaled_data，現在我們要從中提取最後 5000 + look_back - 1 條數據
# df = df[-(only_backtesting_count + look_back - 1):]
# df.reset_index(drop=True, inplace=True)

# 使用StandardScaler對數據進行縮放
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[cols])


# 考慮到look_back，我們從第look_back根開始
input_datas = [scaled_data[i-look_back:i] for i in range(look_back, len(scaled_data) + 1)]
y = [df.iloc[i - 1][['Target']].values for i in range(look_back, len(scaled_data) + 1)]


y = [element.astype(int) for element in y]
X, y = np.array(input_datas), np.array(y)

# 評估模型
loss, accuracy, _, _, _ = model.evaluate(X, y)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

model_process.record_valoss_valaccuracy(loss, accuracy, trained_model_name, len(X))
model_process.export_ROC(model, trained_model_name, X, y)
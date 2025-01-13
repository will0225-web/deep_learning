import sys
import os

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
import tensorflow as tf
import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import time_convert as time_convert
from modules import model as model_process
from modules import custom_model_fit_indicators as custom_model_fit_indicators

from tensorflow.keras.utils import to_categorical

def create_look_back_dataset(X, y, look_back=1):
    dataX, dataY = [], []
    for i in range(len(X) - look_back):
        dataX.append(X[i:(i + look_back)])
        dataY.append(y[i + look_back - 1])  # y的索引需要小心调整
    return np.array(dataX), np.array(dataY)

def multiclass_f1_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_f1_score(y_true, y_pred)

def add_custom_features(df, look_back=14):
    # 计算look_back期间的最高High和最低Low
    df['max_high_look_back'] = df['High'].rolling(window=look_back).max()
    df['min_low_look_back'] = df['Low'].rolling(window=look_back).min()
    
    # 计算斐波那契回撤水平
    df['fibonacci_0.382'] = df['min_low_look_back'] + 0.382 * (df['max_high_look_back'] - df['min_low_look_back'])
    df['fibonacci_0.5'] = df['min_low_look_back'] + 0.5 * (df['max_high_look_back'] - df['min_low_look_back'])
    df['fibonacci_0.618'] = df['min_low_look_back'] + 0.618 * (df['max_high_look_back'] - df['min_low_look_back'])
    
    return df

def add_three_klines(df, look_back=14):
    # 计算连续三个K线的颜色
    df['kline_color_shift_1'] = df['kline_color'].shift(1)
    df['kline_color_shift_2'] = df['kline_color'].shift(2)
    
    # 检查是否有连续三个绿色或红色的K线
    df['three_green_klines'] = ((df['kline_color'] == 1) & (df['kline_color_shift_1'] == 1) & (df['kline_color_shift_2'] == 1)).astype(int)
    df['three_red_klines'] = ((df['kline_color'] == 0) & (df['kline_color_shift_1'] == 0) & (df['kline_color_shift_2'] == 0)).astype(int)

    # 删除辅助列
    df.drop(['kline_color_shift_1', 'kline_color_shift_2'], axis=1, inplace=True)
    return df

def add_lower_low_higher_high(df, look_back=14):
    # 计算look_back期间的第一个和最后一个Close值
    df['first_close_look_back'] = df['Close'].shift(look_back - 1)
    df['last_close_look_back'] = df['Close'].shift(1)

    # 检查是否出现超过0.5%的变化
    df['lower_low'] = (((df['last_close_look_back'] - df['first_close_look_back']) / df['first_close_look_back']) < -0.005).astype(int)
    df['higher_high'] = (((df['last_close_look_back'] - df['first_close_look_back']) / df['first_close_look_back']) > 0.005).astype(int)

    # 计算look_back期间的lower_low和higher_high的数量
    df['lower_low_count'] = df['lower_low'].rolling(window=look_back).sum()
    df['higher_high_count'] = df['higher_high'].rolling(window=look_back).sum()

    # 删除辅助列
    df.drop(['first_close_look_back', 'last_close_look_back'], axis=1, inplace=True)
    return df

def combine_lower_low_higher_high_and_kline(df, look_back=14):
    # 檢查是否滿足指定條件
    df['green_and_higher_high'] = ((df['three_green_klines'] == 1) & (df['higher_high_count'] == look_back)).astype(int)
    df['red_and_lower_low'] = ((df['three_red_klines'] == 1) & (df['lower_low_count'] == look_back)).astype(int)
    return df

def add_rsi_overbought_oversold(df, overbought_threshold=70, oversold_threshold=30):
    # 檢查RSI是否超買
    df['rsi_overbought'] = (df['RSI'] > overbought_threshold).astype(int)

    # 檢查RSI是否超賣
    df['rsi_oversold'] = (df['RSI'] < oversold_threshold).astype(int)

    return df

def calculate_macd_hist_continuity(df):
    df['macd_hist_stronger_continuity'] = 0
    df['macd_hist_weaker_continuity'] = 0

    macd_stronger_status = df['MACD_Stronger']

    for i in range(len(df)):
        continuity_counter = 0

        if macd_stronger_status[i] == 1:  # 走强
            for j in range(i, -1, -1):
                if macd_stronger_status[j] == 1:
                    continuity_counter += 1
                else:
                    break
            df.at[i, 'macd_hist_stronger_continuity'] = continuity_counter

        elif macd_stronger_status[i] == 2:  # 走弱
            for j in range(i, -1, -1):
                if macd_stronger_status[j] == 2:
                    continuity_counter += 1
                else:
                    break
            df.at[i, 'macd_hist_weaker_continuity'] = continuity_counter

    return df

# 特徵欄位
# cols = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "Signal", "Hist", "SMA_27", "SMA_55", "SMA_200", "kline_color", "Middle Band", "Upper Band", "Lower Band"]
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'MACD', 'Signal', 'Hist', 'MACD_Long_Short', 'MACD_Stronger', 'MACD_Cross', 'Up Trend', 'Down Trend', 'Super Trend', 'ATR', 'TR', 'SMA_27', 'SMA_55', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_26', 'EMA_50']
symbol = "ETHUSDT"
interval = "15m"
look_back = 672 #使用回看n根數據


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

# 预处理数据
X = df[cols]
y = df['Target']

# 再拿掉前後無參考性資料
aaa = 960
X = add_three_klines(X)
X = calculate_macd_hist_continuity(X)
X = add_rsi_overbought_oversold(X, 67, 27)
X = add_custom_features(X, look_back=aaa)
X = add_lower_low_higher_high(X, look_back=aaa)
X = combine_lower_low_higher_high_and_kline(X, look_back=aaa)

X = X[1000:-300]
X.reset_index(drop=True, inplace=True)

# df = pd.concat([df, pd.get_dummies(df['Target'], prefix='Target')], axis=1)

# 載入已經訓練好的模型
# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.join(current_path, '..')
trained_model_name = f'{parent_path}/trained_models/1706163684_softmax_loss-1.0271_accuracy-0.6145_f1-0.1837'
model = tf.keras.models.load_model(trained_model_name, custom_objects={'multiclass_f1_score': multiclass_f1_score})

# 使用StandardScaler對數據進行縮放
# 选择要缩放的列
# robust_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'MACD', 'Signal', 'Hist', 'Up Trend', 'Down Trend', 'SMA_27', 'SMA_55', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_26', 'EMA_50']
robust_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'MACD', 'Signal', 'Hist', 'ATR', 'TR', 'Up Trend', 'Down Trend', 'SMA_27', 'SMA_55', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_26', 'EMA_50', 'max_high_look_back', 'min_low_look_back', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618']
standard_features = ['macd_hist_stronger_continuity', 'macd_hist_weaker_continuity', 'lower_low_count', 'higher_high_count']
minMax_features = ['RSI']

# 列出每个缩放器/转换器对应的特征
preprocessor = ColumnTransformer(
    transformers=[
        ('price', RobustScaler(), robust_features),
        ('percent', StandardScaler(), standard_features),
        ('bounded', MinMaxScaler(feature_range=(0, 1)), minMax_features),
    ],
    remainder='passthrough'  # 不需要缩放的特征保持原样
)


# 对特征进行缩放
X_scaled = preprocessor.fit_transform(X)

# 将目标变量转换为分类格式
y_categorical = to_categorical(y, num_classes=5)

# 使用look_back创建数据集
X, y = create_look_back_dataset(X_scaled, y_categorical, look_back)

# 評估模型
loss, accuracy, f1 = model.evaluate(X, y, batch_size=256)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1: {f1:.4f}")

model_process.record_valoss_valaccuracy(loss, accuracy, trained_model_name, len(X))
model_process.export_cm(trained_model_name, model, X, y, [0, 1, 2, 3, 4])
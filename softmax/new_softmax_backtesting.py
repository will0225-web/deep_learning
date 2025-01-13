import sys
import os

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

import csv

from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, PReLU, Conv1D, MaxPooling1D, Flatten, LeakyReLU, ReLU, Bidirectional, Attention, LayerNormalization, Input, Activation, RepeatVector, Permute, Multiply, Layer
from keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import model as model_process
from modules import custom_model_fit_indicators as custom_model_fit_indicators
from keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.utils import to_categorical

def get_original_data(symbol, interval, cols, indicator_look_back):
    original_df = modules_data.get_binance_klines_backward(symbol, interval, "2023-11-30 23:59:59", 52560, cols, is_need_save_original_data=False, is_read_local=True, is_need_calculated=True)

    # 预处理数据
    original_X = original_df[cols]
    original_y = original_df['Target']

    
    # X = indicators.add_lower_low_higher_high(original_X, look_back=indicator_look_back)
    # X = indicators.add_price_indicator(original_X, look_back=indicator_look_back)

    original_X = original_X[1600:-300]
    original_X.reset_index(drop=True, inplace=True)
    original_y = original_y[1600:-300]
    original_y.reset_index(drop=True, inplace=True)
    return original_X, original_y

def calculate_percentage_change(order_price, closing_price):
    return ((closing_price - order_price) / order_price)


def multiclass_f1_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_f1_score(y_true, y_pred)

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

def calculate_current_continued_target_count(predicted_target, current_index, all_predict_target):
    continued_target_count = 0
    for i in range(current_index, -1, -1):  # 从当前索引位置开始向前计数
        if np.argmax(all_predict_target[i]) == predicted_target:
            continued_target_count += 1
        else:
            break
    return continued_target_count

# 特徵欄位
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'MACD', 'Signal', 'Hist', 'MACD_Long_Short', 'MACD_Stronger', 'MACD_Cross', 'Up Trend', 'Down Trend', 'Super Trend', 'ATR', 'TR', 'SMA_55', 'SMA_200', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_26', 'EMA_50', 'green_three_soldiers', 'red_three_soldiers', 'rsi_overbought', 'rsi_oversold', 'macd_hist_stronger_continuity', 'macd_hist_weaker_continuity']
symbol = "ETHUSDT"
interval = "15m"
look_back = 144 #使用回看n根數據
epochs = 100
batch_size = 128
indicator_look_back = 1440

# 拿原始資料
original_X, original_y = get_original_data(symbol, interval, cols, indicator_look_back)
# 當前
end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

end_time_seconds = end_time / 1000
end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# 特定
# end_time_string = "2023-11-30 23:59:59"

# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 4000, cols, is_need_save_original_data=False, is_read_local=False, is_need_calculated=True)

# 预处理数据
X = df[cols]
y = df['Target']

selected_cols = ['Upper Band', 'MACD', 'Lower Band', 'Middle Band', 'SMA_200', 'RSI', 'MACD_Cross', 'MACD_Long_Short', 'SMA_55', 'kline_color']


# X = indicators.add_lower_low_higher_high(X, look_back=indicator_look_back)
# X = indicators.add_price_indicator(X, look_back=indicator_look_back)


# 再拿掉前後無參考性資料
X = X[1600:]
X.reset_index(drop=True, inplace=True)
y = y[1600:]
y.reset_index(drop=True, inplace=True)

X = X[selected_cols]

# 选择要缩放的列
# robust_features = ['Open', 'High', 'Low', 'Close', 'Up Trend', 'Down Trend', 'SMA_27', 'SMA_55', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_26', 'EMA_50', 'max_high_look_back', 'min_low_look_back', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'MACD', 'Signal', 'Hist']
# standard_features = ['Volume', 'ATR', 'TR', 'lower_low_count', 'higher_high_count']
# minMax_features = []

robust_features = ['Upper Band', 'Middle Band', 'Lower Band', 'SMA_200', 'SMA_55', 'MACD']
standard_features = []
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
original_X = original_X[selected_cols]
preprocessor.fit(original_X)  # original_X 是原始数据

# 对特征进行缩放
X_scaled = preprocessor.transform(X)

# 将目标变量转换为分类格式
y_categorical = to_categorical(y, num_classes=5)

# 使用look_back创建数据集
np_X, np_y = modules_data.create_look_back_dataset(X_scaled, y_categorical, look_back)

# 載入已經訓練好的模型
# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.join(current_path, '..')
trained_model_name = f'{parent_path}/trained_models/1707586561_softmax_loss-1.1621_accuracy-0.4951_f1-0.4421'
model = tf.keras.models.load_model(trained_model_name, custom_objects={'multiclass_f1_score': multiclass_f1_score})

df = df[1600:]
df.reset_index(drop=True, inplace=True)
X['High'] = df['High']
X['Low'] = df['Low']
X['Close'] = df['Close']

# 評估模型
loss, accuracy, f1 = model.evaluate(np_X, np_y, batch_size=batch_size)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1: {f1:.4f}")

# predicted = []
# # 遍历每个数据点
# for i in range(len(np_X)):
#     model.reset_states()
#     # 获取当前数据点，注意保持输入形状为 (1, look_back, 特征数)
#     current_point = np_X[i:i+1]
#     # 使用模型进行预测
#     predicted_value = model.predict(current_point)
    
#     # 将预测结果添加到列表中
#     predicted.append(predicted_value)
# predicted = np.array(predicted)

predicted = model.predict(np_X)
# predicted = np_y

time = int(datetime.datetime.timestamp(datetime.datetime.now()))
output_module_name = f'{time}:softmax'


total_profit = 0
position = 0  # 当前仓位状态，0表示无仓位，1表示多头仓位，-1表示空头仓位
entry_price = 0  # 开仓价格
entry_rsi = 0
exit_price = 0  # 平仓价格
entry_index = 0  # 开仓的索引
lose_count = 0
win_count = 0

rsi_exit_delta = 12
stop_loss = 0.005
horizontal_price_percentage = 0.006
max_profit = 0

entry_confidence = 0
min_continued_target_count = 1  # 举例，需要至少连续3次相同的预测, 包含自己, 所以最小值通常為1

fee = 0.0004

horizontal_price_hold_count_limit = 24  # 设定的时间段
hold_count_limit = 48
# max_profit_threshold = 0.004  # 最大利润阈值，这里是1%

for i in range(0, len(predicted)):
    # 获取原始数据集中的特定时间点的数据
    original_data = X.iloc[i + look_back - 1]  # 假设 original_X 是DataFrame

    # 获取原始的 Close 和 RSI 值
    original_close = original_data['Close']
    original_high = original_data['High']
    original_low = original_data['Low']
    original_rsi = original_data['RSI']
    
    p = predicted[i]
    predicted_target = np.argmax(p)
    confidence = np.max(predicted[i])

    # real target
    target = np.argmax(np_y[i])

    if position != 0:
        if position == 1:
            if (entry_price - original_low) / entry_price >= stop_loss:
                # 止損
                lose_count += 1
                total_profit -= (stop_loss + fee)

                position = 0
            else:
                current_profit = calculate_percentage_change(entry_price, original_close)
                entry_rsi = max(original_rsi, entry_rsi)
                max_profit = max(max_profit, current_profit)
                if (original_rsi < entry_rsi - rsi_exit_delta) or i - entry_index >= hold_count_limit or (i - entry_index >= horizontal_price_hold_count_limit and abs(max_profit) < horizontal_price_percentage):
                    # 結單
                    profit = current_profit - fee
                    total_profit += profit

                    if profit <= 0:
                        lose_count += 1
                    else:
                        win_count += 1
                    
                    position = 0

        elif position == -1:
            if (original_high - entry_price) / entry_price >= stop_loss:
                # 止損
                lose_count += 1
                total_profit -= (stop_loss + fee)

                position = 0
            else:
                current_profit = -calculate_percentage_change(entry_price, original_close)
                entry_rsi = min(original_rsi, entry_rsi)
                max_profit = max(max_profit, current_profit)
                if (original_rsi > entry_rsi + rsi_exit_delta) or i - entry_index >= hold_count_limit or (i - entry_index >= horizontal_price_hold_count_limit and abs(max_profit) < horizontal_price_percentage):
                    # 結單
                    profit = current_profit - fee
                    total_profit += profit

                    if profit <= 0:
                        lose_count += 1
                    else:
                        win_count += 1

                    position = 0
    else:
        max_profit = 0
        # 开仓逻辑
        # 计算当前位置的连续预测次数
        continued_count = calculate_current_continued_target_count(predicted_target, i, predicted)

        if predicted_target == 1 and confidence >= entry_confidence and continued_count >= min_continued_target_count:  # 预测为多头且当前无仓位
            position = 1
            entry_price = original_close
            entry_index = i
            entry_rsi = original_rsi
        elif predicted_target == 2 and confidence >= entry_confidence and continued_count >= min_continued_target_count:  # 预测为空头且当前无仓位
            position = -1
            entry_price = original_close
            entry_index = i
            entry_rsi = original_rsi


    with open(f'{trained_model_name}/{output_module_name}_output.csv', 'a', newline='') as file:  # 'a'表示附加模式，這樣數據將會被添加到文件而不是覆蓋它
        writer = csv.writer(file)
        # 定义标题行
        headers = ["Original Close", "Original High", "Original Low", "Original RSI", "Win Count", "Lose Count", "Total Profit", "Max Profit", "Position", "Predicted Target", "Target", "Confidence"]

        # 检查文件是否为空，如果是空的，写入标题行
        if file.tell() == 0:
            writer.writerow(headers)


        writer.writerow([original_close, original_high, original_low, original_rsi, win_count, lose_count, total_profit, max_profit, position, predicted_target, target, confidence])


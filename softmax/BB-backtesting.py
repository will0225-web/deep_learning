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

def get_original_data(symbol, interval, cols):
    original_df = modules_data.get_binance_klines_backward(symbol, interval, "2023-12-31 23:59:59", 70080, cols, is_need_save_original_data=False, is_read_local=True, is_need_calculated=True)

    # 预处理数据
    original_X = original_df[cols]
    original_y = original_df['Target']

    
    indicator_look_back = 192
    price_look_back = 480
    original_X = indicators.add_lower_low_higher_high(original_X, 0.06, look_back=indicator_look_back)
    original_X = indicators.add_price_indicator(original_X, look_back=price_look_back)
    original_X = indicators.add_24h_min_max_price(original_X, look_back=96)
    # 再拿掉前後無參考性資料
    original_X = original_X[(price_look_back + 1 + 1600):-300]
    original_X.reset_index(drop=True, inplace=True)
    original_y = original_y[(price_look_back + 1 + 1600):-300]
    original_y.reset_index(drop=True, inplace=True)
    return original_X, original_y


def multiclass_f1_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_f1_score(y_true, y_pred)

# 特徵欄位
cols = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'MACD', 'Signal', 'Hist', 'MACD_Long_Short', 'MACD_Stronger', 'MACD_Cross', 'Up Trend', 'Down Trend', 'Super Trend', 'ATR', 'TR', 'SMA_55', 'SMA_200', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_26', 'EMA_50', 'green_three_soldiers', 'red_three_soldiers', 'rsi_overbought', 'rsi_oversold']
symbol = "ETHUSDT"
interval = "15m"
look_back = 48 #使用回看n根數據
epochs = 100
batch_size = 128



# 拿原始資料
original_X, original_y = get_original_data(symbol, interval, cols)
# 當前
end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

end_time_seconds = (end_time / 1000) + 1
end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# 特定
# end_time_string = "2024-03-09 07:00:01"

# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 3000, cols, is_need_save_original_data=False, is_read_local=False, is_need_calculated=True)

# 预处理数据
X = df[cols]
y = df['Target']


indicator_look_back = 192
price_look_back = 480
X = indicators.add_lower_low_higher_high(X, 0.06, look_back=indicator_look_back)
X = indicators.add_price_indicator(X, look_back=price_look_back)
X = indicators.add_24h_min_max_price(X, look_back=96)


# 再拿掉前後無參考性資料
X = X[(price_look_back + 1 + 1600):]
X.reset_index(drop=True, inplace=True)
y = y[(price_look_back + 1 + 1600):]
y.reset_index(drop=True, inplace=True)

# 选择要缩放的列
ochl_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'TR', 'Upper Band', 'Lower Band', 'Middle Band']
ochl_data = X[ochl_cols]

robust_features = []
standard_features = ['Close', 'High', 'Low', 'Open', 'Volume', 'TR', 'Upper Band', 'Lower Band', 'Middle Band']
minMax_features = []

# 列出每个缩放器/转换器对应的特征
preprocessor = ColumnTransformer(
    transformers=[
        ('price', RobustScaler(), robust_features),
        ('percent', StandardScaler(), standard_features),
        ('bounded', MinMaxScaler(feature_range=(0, 1)), minMax_features),
    ],
    remainder='passthrough'  # 不需要缩放的特征保持原样
)
preprocessor.fit(original_X[ochl_cols])  # original_X 是原始数据
# 对特征进行缩放
X_ochl_scaled = preprocessor.transform(ochl_data)


current_cols = ['red_three_soldiers', 'green_three_soldiers', 'MACD_Stronger', 'rsi_oversold', 'rsi_overbought']
current_data = X[current_cols]

robust_features = []
standard_features = []
minMax_features = []

# 列出每个缩放器/转换器对应的特征
preprocessor = ColumnTransformer(
    transformers=[
        ('price', RobustScaler(), robust_features),
        ('percent', StandardScaler(), standard_features),
        ('bounded', MinMaxScaler(feature_range=(0, 1)), minMax_features),
    ],
    remainder='passthrough'  # 不需要缩放的特征保持原样
)
preprocessor.fit(original_X[current_cols])  # original_X 是原始数据
# 对特征进行缩放
X_current_scaled = preprocessor.transform(current_data)
X_current_scaled = X_current_scaled[look_back:]

# 将目标变量转换为分类格式
y_categorical = to_categorical(y, num_classes=5)

# 使用look_back创建数据集
np_X, np_y = modules_data.create_look_back_dataset(X_ochl_scaled, y_categorical, look_back)

# 載入已經訓練好的模型
# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.join(current_path, '..')
trained_model_name = f'{parent_path}/trained_models/1709987084_softmax_loss-0.7022_accuracy-0.7377_f1-0.3519'
model = tf.keras.models.load_model(trained_model_name, custom_objects={'multiclass_f1_score': multiclass_f1_score})

df = df[(price_look_back + 1 + 1600):]
df.reset_index(drop=True, inplace=True)
X['High'] = df['High']
X['Low'] = df['Low']
X['Close'] = df['Close']
X['datetime'] = df['datetime']

# 評估模型
loss, accuracy, f1 = model.evaluate([np_X, X_current_scaled], np_y, batch_size=batch_size)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1: {f1:.4f}")

predicted = model.predict([np_X, X_current_scaled])
# predicted = np_y

time = int(datetime.datetime.timestamp(datetime.datetime.now()))
output_module_name = f'{time}:softmax'


total_profit = 0
position = 0  # 当前仓位状态，0表示无仓位，1表示多头仓位，-1表示空头仓位
entry_price = 0  # 开仓价格
exit_price = 0  # 平仓价格
entry_index = 0  # 开仓的索引
entry_predicted_target = 0 #開倉predict_target
lose_count = 0
win_count = 0

stop_loss_3_4 = 0.012
take_profit = 0.012
stop_loss = 0.008
max_profit = 0

entry_confidence = 0.4
entry_confidence_3_4 = 0.7

fee = 0.0004
# fee = 0

hold_count_limit = 16

for i in range(0, len(predicted)):
    # 获取原始数据集中的特定时间点的数据
    original_data = X.iloc[i + look_back]  # 假设 original_X 是DataFrame


    # 获取原始的 Close 和 RSI 值
    original_close = original_data['Close']
    original_high = original_data['High']
    original_low = original_data['Low']
    original_datetime = original_data['datetime']
    
    p = predicted[i]
    predicted_target = np.argmax(p)
    confidence = np.max(predicted[i])

    # real target
    target = np.argmax(np_y[i])

    if position != 0:
        if position == 1 and entry_predicted_target == 1:
            if (entry_price - original_low) / entry_price >= stop_loss:
                real_stop_loss_profit = ((original_low - entry_price) / entry_price) - fee
                current_profit = max(real_stop_loss_profit, -stop_loss - fee)
                lose_count += 1

                profit = current_profit
                total_profit += profit
                
                position = 0
                max_profit = 0
                entry_price = 0 
            else:
                current_profit = indicators.calculate_percentage_change(original_high, entry_price)
                max_profit = max(max_profit, current_profit)
                
                if current_profit >= take_profit or i - entry_index >= hold_count_limit:
                    # 結單
                    if current_profit >= take_profit:
                        # 不超過take profit的percentage
                        profit = min(current_profit - fee, take_profit - fee)
                    else:
                        # 其餘正常算
                        profit = current_profit - fee

                    total_profit += profit
                    
                    if profit <= 0:
                        lose_count += 1
                    else:
                        win_count += 1
                    
                    position = 0
                    max_profit = 0
                    entry_price = 0

        elif position == -1 and entry_predicted_target == 2:
            if (original_high - entry_price) / entry_price >= stop_loss:
                real_stop_loss_profit = ((entry_price - original_high) / entry_price) - fee
                current_profit = max(real_stop_loss_profit, -stop_loss - fee)
                lose_count += 1
                
                profit = current_profit
                total_profit += profit

                position = 0
                max_profit = 0
                entry_price = 0
            else:
                current_profit = -indicators.calculate_percentage_change(original_low, entry_price)
                max_profit = max(max_profit, current_profit)

                if current_profit >= take_profit or i - entry_index >= hold_count_limit:
                    # 結單
                    if current_profit >= take_profit:
                        # 不超過take profit的percentage
                        profit = min(current_profit - fee, take_profit - fee)
                    else:
                        # 其餘正常算
                        profit = current_profit - fee

                    total_profit += profit

                    if profit <= 0:
                        lose_count += 1
                    else:
                        win_count += 1

                    position = 0
                    max_profit = 0
                    entry_price = 0
        # if position == -1 and entry_predicted_target == 3:
        #     if (entry_price - original_low) / entry_price >= stop_loss:
        #         real_stop_loss_profit = ((original_low - entry_price) / entry_price)
        #         current_profit = max(real_stop_loss_profit, -stop_loss)
        #         current_profit = -current_profit - fee
        #         win_count += 1

        #         profit = current_profit
        #         total_profit += profit
                
        #         position = 0
        #         max_profit = 0
        #         entry_price = 0 
        #     else:
        #         current_profit = indicators.calculate_percentage_change(original_high, entry_price)
        #         max_profit = max(max_profit, current_profit)
                
        #         if current_profit >= stop_loss_3_4 or i - entry_index >= hold_count_limit:
        #             # 結單
        #             if current_profit >= stop_loss_3_4:
        #                 # 不超過take profit的percentage
        #                 profit = min(current_profit, stop_loss_3_4)
        #             else:
        #                 # 其餘正常算
        #                 profit = current_profit
        #             # 轉反向
        #             profit = -profit - fee

        #             total_profit += profit
        #             if profit <= 0:
        #                 lose_count += 1
        #             else:
        #                 win_count += 1
                    
        #             position = 0
        #             max_profit = 0
        #             entry_price = 0
        # if position == 1 and entry_predicted_target == 4:
        #     if (original_high - entry_price) / entry_price >= stop_loss:
        #         real_stop_loss_profit = ((entry_price - original_high) / entry_price)
        #         current_profit = max(real_stop_loss_profit, -stop_loss)
        #         current_profit = -current_profit - fee
        #         win_count += 1
                
        #         profit = current_profit
        #         total_profit += profit

        #         position = 0
        #         max_profit = 0
        #         entry_price = 0
        #     else:
        #         current_profit = -indicators.calculate_percentage_change(original_low, entry_price)
        #         max_profit = max(max_profit, current_profit)

        #         if current_profit >= stop_loss_3_4 or i - entry_index >= hold_count_limit:
        #             # 結單
        #             if current_profit >= stop_loss_3_4:
        #                 # 不超過take profit的percentage
        #                 profit = min(current_profit, stop_loss_3_4)
        #             else:
        #                 # 其餘正常算
        #                 profit = current_profit

        #             # 轉反向
        #             profit = -profit - fee

        #             total_profit += profit

        #             if profit <= 0:
        #                 lose_count += 1
        #             else:
        #                 win_count += 1

        #             position = 0
        #             max_profit = 0
        #             entry_price = 0
    else:
        max_profit = 0
        # 开仓逻辑

        if (predicted_target == 1 and confidence >= entry_confidence) or (predicted_target == 3 and confidence >= entry_confidence_3_4) or (predicted_target == 3 and confidence < entry_confidence):
            position = 1

            entry_price = original_close
            entry_index = i

            entry_predicted_target = 1
        elif (predicted_target == 2 and confidence >= entry_confidence) or (predicted_target == 4 and confidence >= entry_confidence_3_4) or (predicted_target == 4 and confidence < entry_confidence):
            position = -1

            entry_price = original_close
            entry_index = i
            
            entry_predicted_target = 2
            

    with open(f'{trained_model_name}/{output_module_name}_output.csv', 'a', newline='') as file:  # 'a'表示附加模式，這樣數據將會被添加到文件而不是覆蓋它
        writer = csv.writer(file)
        # 定义标题行
        headers = ["Entry Price", "Original Close", "Original High", "Original Low", "Win Count", "Lose Count", "Total Profit", "Max Profit", "Position", "Original Datetime", "Predicted Target", "Target", "Confidence"]

        # 检查文件是否为空，如果是空的，写入标题行
        if file.tell() == 0:
            writer.writerow(headers)

        writer.writerow([entry_price, original_close, original_high, original_low, win_count, lose_count, total_profit, max_profit, position, original_datetime, predicted_target, target, confidence])


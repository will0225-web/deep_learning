import sys
import os

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, PReLU, GRU, Conv1D, MaxPooling1D, Flatten, ReLU
from keras.regularizers import l1_l2, l1, l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.utils import class_weight

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import model as model_process
from modules import custom_model_fit_indicators as custom_model_fit_indicators
from keras.callbacks import ReduceLROnPlateau


def multiclass_f1_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_f1_score(y_true, y_pred)


# 特徵欄位
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'RSI_6', 'RSI_12', 'RSI_24', 'MACD', 'Signal', 'Hist', 'MACD_Stronger', 'MACD_Cross', 'Up Trend', 'Down Trend', 'Super Trend', 'ATR', 'SMA_12', 'SMA_21', 'SMA_27', 'SMA_55', 'SMA_200', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_12', 'EMA_26', 'EMA_50']

symbol = "ETHUSDT"
interval = "15m"
epochs = 200
batch_size = 128
look_back = 220 #使用回看n根數據

# 當前
# end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# 特定
end_time_string = "2023-10-31 23:59:59"

# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 150000, cols, is_need_save_original_data=False, is_read_local=True, is_need_calculated=True)

# new_data_end_time_string = "2023-11-30 23:59:59"
# df_new = modules_data.get_binance_klines_backward(symbol, interval, new_data_end_time_string, 4000, cols, is_need_save=False, is_read_local=False, is_need_calculated=True) 

# 再拿掉前後無參考性資料
df = df[1000:-300]
# df_new = df_new[1000:-300]

df.reset_index(drop=True, inplace=True)
df.to_csv('test.csv')

# df_new.reset_index(drop=True, inplace=True)

macd_cross_indices = df.index[df['MACD_Cross'] != 0].tolist()
# df_new_macd_cross_indices = df_new.index[df_new['MACD_Cross'] != 0].tolist()

# 使用Scaler對數據進行縮放
scaler = RobustScaler()
scaled_data = scaler.fit_transform(df[cols])
# scaled_new_data = df_new[cols]
df = pd.concat([df, pd.get_dummies(df['Target'], prefix='Target')], axis=1)
X, y = [], []
for index in macd_cross_indices:
    start_index = index - (look_back - 1)
    if start_index >= 0:  # 确保开始索引不为负数
        X.append(scaled_data[start_index:index + 1])
        y.append(df.iloc[index][['Target_0', 'Target_1', 'Target_2', 'Target_3']].values)
y = [element.astype(int) for element in y]
X, y = np.array(X), np.array(y)

# 將數據分割為訓練集、驗證集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 建模與訓練
model = Sequential()

# model.add(Conv1D(
#     filters=50,
#     kernel_size=9,
#     activation='relu',
#     kernel_regularizer=l1_l2(l2=0.0005),
#     input_shape=(X_train.shape[1], X_train.shape[2])
# ))
# model.add(BatchNormalization())
# model.add(MaxPooling1D(pool_size=3))
# model.add(Dropout(0.3))

# model.add(Flatten())

model.add(Dense(300))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.3))

model.add(Dense(150, kernel_regularizer=l1_l2(l2=0.0001)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.4))

# 因為是多類別分類，所以最後一層有3個神經元，並使用softmax作為激活函數
model.add(Dense(units=4, activation='softmax'))

# 初始化ReduceLROnPlateau回調
reduce_lr = ReduceLROnPlateau(monitor='val_multiclass_f1_score',  # 監控驗證集的損失
                              factor=0.5,          # 學習率被減少的因子 (new_lr = lr * factor)
                              patience=10,         # 沒有進步的時期數，在這之後學習率會被減少
                              min_lr=0.00001,      # 學習率的下限
                              verbose=1,
                              mode='max')           # 信息展示模式
# 使用更小的初始學習率
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', multiclass_f1_score])

# 1. 从one-hot编码中还原y到单一标签格式
y_single_label = np.argmax(y, axis=1)
# 2. 使用compute_class_weight计算权重
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_single_label), y=y_single_label)
# 3. 创建一个与类标签匹配的权重字典
class_weights_dict = {i: weights[i] for i in range(len(weights))}

callback = tf.keras.callbacks.EarlyStopping(monitor='val_multiclass_f1_score', patience=50, restore_best_weights=True, mode='max')
history = model.fit(X_train, y_train, class_weight=class_weights_dict, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[callback, reduce_lr])

# 評估模型
loss, accuracy, f1 = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test f1: {f1:.4f}")

# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))

# 定義上層目錄的路徑
parent_path = os.path.join(current_path, '..')
time = int(datetime.datetime.timestamp(datetime.datetime.now()))
trained_model_name = f'{parent_path}/trained_models/{time}_softmax_loss-{loss:.4f}_accuracy-{accuracy:.4f}_f1-{f1:.4f}'
model_process.save_model(trained_model_name, model, loss, accuracy)
model_process.export_epoch_info(history, trained_model_name)
model_process.export_cm(trained_model_name, model, X_test, y_test, [0, 1, 2, 3])
model_process.export_layer_parameters(trained_model_name, model, cols, symbol, interval, look_back, df, batch_size, epochs, end_time_string)

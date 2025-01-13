import sys
import os

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Conv1D, MaxPooling1D, PReLU, ReLU, Layer, Reshape, LeakyReLU, Flatten
from keras.regularizers import l1_l2, l1, l2
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import os
import tensorflow as tf

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import model as model_process
from modules import custom_model_fit_indicators as custom_model_fit_indicators
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall, AUC

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 创建用于注意力计算的权重
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # 计算注意力分数
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        # 输出加权平均的序列
        return tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        return super(AttentionLayer, self).get_config()
    
def multiclass_f1_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_f1_score(y_true, y_pred)

# 特徵欄位
# cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'MACD', 'Signal', 'Hist', 'Up Trend', 'Down Trend', 'Super Trend', 'TR', 'ATR', 'SMA_27', 'SMA_55', 'SMA_200', 'MACD_Cross', 'MACD_Stronger', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band']
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'RSI_24', 'MACD', 'Signal', 'Hist', 'Up Trend', 'Down Trend', 'Super Trend', 'ATR', 'SMA_55', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band']
symbol = "ETHUSDT"
interval = "5m"
look_back = 12 #使用回看n根數據
epochs = 100
batch_size = 256

# 當前
# end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# 特定
end_time_string = "2023-10-31 23:59:59"

# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 100000, cols, is_need_save_original_data=False, is_read_local=True, is_need_calculated=True)
# 再拿掉前後無參考性資料
df = df[1000:-1000]
df.reset_index(drop=True, inplace=True)
df.to_csv('test.csv')

new_data_end_time_string = "2023-11-30 23:59:59"
df_new = modules_data.get_binance_klines_backward(symbol, interval, new_data_end_time_string, 9000, cols, is_need_save_original_data=False, is_read_local=False, is_need_calculated=True) 
df_new = df_new[1000:-1000]
df_new.reset_index(drop=True, inplace=True)

# 使用MinMaxScaler對數據進行縮放
scaler = RobustScaler()
# 保持一致性的縮放
scaler.fit(df[cols])

scaled_data = scaler.fit_transform(df[cols])
scaled_new_data= scaler.fit_transform(df_new[cols])

# df = pd.concat([df, pd.get_dummies(df['Target'], prefix='Target')], axis=1)
# df_new = pd.concat([df_new, pd.get_dummies(df_new['Target'], prefix='Target')], axis=1)

# 創建X, y數據集
X, y = [], []
for i in range(look_back, len(scaled_data) + 1):
    X.append(scaled_data[i - look_back:i])
    y.append(df.iloc[i - 1][['Target']].values)

y = [element.astype(int) for element in y]
X, y = np.array(X), np.array(y)

X_new, y_new = [], []
for new_index in range(look_back, len(scaled_new_data) + 1):
    X_new.append(scaled_new_data[new_index - look_back:new_index])
    y_new.append(df_new.iloc[new_index - 1][['Target']].values)
y_new = [element.astype(int) for element in y_new]
X_new, y_new = np.array(X_new), np.array(y_new)

# 2. 分割訓練和測試數據
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 將數據分割為訓練集、驗證集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

num_class_1 = np.sum(y_train == 1)
num_class_0 = len(y_train) - num_class_1

print(f"0 的数量: {num_class_0}")
print(f"1 的数量: {num_class_1}")

# 3. 建模與訓練
model = Sequential()

model.add(Conv1D(
    filters=600,
    kernel_size=5,
    input_shape=(X_train.shape[1], X_train.shape[2]),
    kernel_regularizer=l1_l2(l1=0.00001, l2=0.0001),
))
model.add(BatchNormalization())
model.add(PReLU())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))

# model.add(Conv1D(
#     filters=400,
#     kernel_size=9,
#     input_shape=(X_train.shape[1], X_train.shape[2]),
#     kernel_regularizer=l1_l2(l1=0.00001, l2=0.0001),
# ))
# model.add(BatchNormalization())
# model.add(ReLU())
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.4))

model.add(LSTM(units=512,
                return_sequences=True,
                kernel_regularizer=l1_l2(l1=0.00001, l2=0.00005),
                # input_shape=(X_train.shape[1], X_train.shape[2]),
                # recurrent_regularizer=l1_l2(l1=0.00001, l2=0.00001)
                ))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# model.add(LSTM(units=100,
#                 return_sequences=False,
#                 kernel_regularizer=l1_l2(l1=0.00001, l2=0.00005),
#                 # input_shape=(X_train.shape[1], X_train.shape[2]),
#                 # recurrent_regularizer=l1_l2(l1=0.00001, l2=0.00001)
#                 ))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))

# model.add(LSTM(units=50,
#                return_sequences=False, 
#                kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)
#             #    recurrent_regularizer=l1_l2(l1=0.0002, l2=0.002)
#                ))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))

# model.add(AttentionLayer())

model.add(Flatten())

model.add(Dense(500,
                kernel_regularizer=l1_l2(l1=0.00005, l2=0.0005)
                ))
model.add(BatchNormalization())
# model.add(ReLU())
model.add(LeakyReLU())
model.add(Dropout(0.4))

# model.add(Dense(150,
#                 kernel_regularizer=l1_l2(l1=0.00005, l2=0.0005)
#                 ))
# model.add(BatchNormalization())
# model.add(ReLU())
# # model.add(LeakyReLU())
# model.add(Dropout(0.4))

model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.001)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)  # 使用RMSprop优化器

# 初始化ReduceLROnPlateau回調
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # 監控驗證集的損失
                              factor=0.5,          # 學習率被減少的因子 (new_lr = lr * factor)
                              patience=5,         # 沒有進步的時期數，在這之後學習率會被減少
                              min_lr=0.00001,      # 學習率的下限
                              verbose=1)           # 信息展示模式

# sigmoid
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', multiclass_f1_score])

# 計算類別權重, 資料不平衡時, 讓漲或跌的輸出情況平衡
# weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train.reshape(-1))
# class_weights = {0: weights[0], 1: weights[1]}

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True)
# history = model.fit(X_train, y_train, class_weight=class_weights, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[callback, reduce_lr, checkpoint])
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[callback, reduce_lr, checkpoint])

# 評估模型
loss, accuracy, f1 = model.evaluate(X_new, y_new)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test f1: {f1:.4f}")


# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))

# 定義上層目錄的路徑
parent_path = os.path.join(current_path, '..')
time = int(datetime.datetime.timestamp(datetime.datetime.now()))
trained_model_name = f'{parent_path}/trained_models/{time}_sigmoid_loss-{loss:.4f}_accuracy-{accuracy:.4f}_f1-{f1:.4f}'
model_process.save_model(trained_model_name, model, loss, accuracy)
model_process.export_epoch_info(history, trained_model_name)
model_process.export_ROC(model, trained_model_name, X_new, y_new)
model_process.export_cm(trained_model_name, model, X_new, y_new, [0, 1])
model_process.export_layer_parameters(trained_model_name, model, cols, symbol, interval, look_back, df, batch_size, epochs, end_time_string)

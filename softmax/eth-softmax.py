import sys
import os

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import datetime
import pandas as pd
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, PReLU, Conv1D, MaxPooling1D, Flatten, LeakyReLU, ReLU, Bidirectional, Attention, LayerNormalization, Input, Activation, RepeatVector, Permute, Multiply, GlobalAveragePooling1D
from keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
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
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'MACD', 'Signal', 'Hist', 'Up Trend', 'Down Trend', 'Super Trend', 'ATR', 'TR', 'SMA_21', 'SMA_55', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_26', 'EMA_50']
symbol = "ETHUSDT"
interval = "15m"
look_back = 192 #使用回看n根數據
epochs = 100
batch_size = 256

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
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 35040, cols, is_need_save_original_data=False, is_read_local=True, is_need_calculated=True)
# 再拿掉前後無參考性資料
df = df[1000:-300]
df.reset_index(drop=True, inplace=True)
df.to_csv('test.csv')

new_data_end_time_string = "2023-12-31 23:59:59"
df_new = modules_data.get_binance_klines_backward(symbol, interval, new_data_end_time_string, 4000, cols, is_need_save_original_data=False, is_read_local=False, is_need_calculated=True) 
df_new = df_new[1000:-300]
df_new.reset_index(drop=True, inplace=True)

# 使用MinMaxScaler對數據進行縮放
scaler = RobustScaler()
# 保持一致性的縮放
scaler.fit(df[cols])

scaled_data = scaler.fit_transform(df[cols])
scaled_new_data= scaler.fit_transform(df_new[cols])

df = pd.concat([df, pd.get_dummies(df['Target'], prefix='Target')], axis=1)
# df_new = pd.concat([df_new, pd.get_dummies(df_new['Target'], prefix='Target')], axis=1)

# 創建X, y數據集
X, y = [], []
for i in range(look_back, len(scaled_data) + 1):
    X.append(scaled_data[i - look_back:i])
    y.append(df.iloc[i - 1][['Target_0', 'Target_1', 'Target_2', 'Target_3', 'Target_4']].values)

y = [element.astype(int) for element in y]
X, y = np.array(X), np.array(y)

X_new, y_new = [], []
for new_index in range(look_back, len(scaled_new_data) + 1):
    X_new.append(scaled_new_data[new_index - look_back:new_index])
    y_new.append(df_new.iloc[new_index - 1][['Target_0', 'Target_1', 'Target_2', 'Target_3', 'Target_4']].values)
y_new = [element.astype(int) for element in y_new]
X_new, y_new = np.array(X_new), np.array(y_new)


# 2. 分割訓練和測試數據
# X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
# X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42, stratify=y_temp)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# X_val = X_test
# y_val = y_test
# 將數據分割為訓練集、驗證集和測試集
# X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2222, shuffle=False)

X_train = X
y_train = y
X_test = X_new
y_test = y_new
X_val = X_new
y_val = y_new

# 3. 建模與訓練
# 定义输入层
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
# , kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)
# 卷积层
x = Conv1D(128, kernel_size=3)(inputs)
x = BatchNormalization()(x)
x = ReLU()(x)
# x = MaxPooling1D(pool_size=2)(x)
# x = Dropout(0.2)(x)

# x = Conv1D(128, kernel_size=3, strides=5)(x)
# x = BatchNormalization()(x)
# x = ReLU()(x)
# x = MaxPooling1D(pool_size=2)(x)
# x = Dropout(0.3)(x)

# x = Conv1D(256, kernel_size=7, strides=7)(x)
# x = BatchNormalization()(x)
# x = ReLU()(x)
# x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)

# 双向LSTM层
bi_lstm = Bidirectional(LSTM(200, return_sequences=True))(x)
x = LayerNormalization()(bi_lstm)
x = Dropout(0.3)(x)

bi_lstm_2 = Bidirectional(LSTM(150, return_sequences=False))(x)
x = LayerNormalization()(bi_lstm_2)
x = Dropout(0.3)(x)

# x = Flatten()(x)

# 密集层
x = Dense(256)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Dropout(0.3)(x)

outputs = Dense(units=5, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

optimizer = Adam(learning_rate=0.0001)
# optimizer = RMSprop(learning_rate=0.00005)
# optimizer = Nadam(learning_rate=0.00005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', multiclass_f1_score])

# 初始化ReduceLROnPlateau回調
reduce_lr = ReduceLROnPlateau(monitor='val_multiclass_f1_score',  # 監控驗證集的損失
                              factor=0.2,          # 學習率被減少的因子 (new_lr = lr * factor)
                              patience=5,         # 沒有進步的時期數，在這之後學習率會被減少
                              min_lr=0.00001,      # 學習率的下限
                              verbose=1,
                              mode='max')           # 信息展示模式


# 1. 从one-hot编码中还原y到单一标签格式
y_single_label = np.argmax(y_train, axis=1)
# 2. 使用compute_class_weight计算权重
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_single_label), y=y_single_label)
# 3. 创建一个与类标签匹配的权重字典
class_weights_dict = {i: weights[i] for i in range(len(weights))}
# class_weights_dict = {i: 1 for i in range(len(weights))}

callback = tf.keras.callbacks.EarlyStopping(monitor='val_multiclass_f1_score', patience=15, restore_best_weights=True, mode='max')
history = model.fit(X_train, y_train, class_weight=class_weights_dict, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[callback, reduce_lr])
# history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[callback, reduce_lr])

# 評估模型
loss, accuracy, f1 = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1: {f1:.4f}")

# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))

# 定義上層目錄的路徑
parent_path = os.path.join(current_path, '..')
time = int(datetime.datetime.timestamp(datetime.datetime.now()))
trained_model_name = f'{parent_path}/trained_models/{time}_softmax_loss-{loss:.4f}_accuracy-{accuracy:.4f}_f1-{f1:.4f}'
model_process.save_model(trained_model_name, model, loss, accuracy)
model_process.export_epoch_info(history, trained_model_name)
model_process.export_cm(trained_model_name, model, X_test, y_test, [0, 1, 2, 3, 4])
model_process.export_layer_parameters(trained_model_name, model, cols, symbol, interval, look_back, df, batch_size, epochs, end_time_string)
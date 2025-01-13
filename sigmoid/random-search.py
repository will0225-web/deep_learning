import sys
import os

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import tensorflow as tf
import csv

from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, PReLU, Conv1D, MaxPooling1D, Flatten, ReLU, LSTM, LeakyReLU
from keras.regularizers import l1_l2, l1, l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from keras_tuner import HyperModel, Objective

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import model as model_process
from modules import custom_callback as custom_callback
from modules import custom_model_fit_indicators as custom_model_fit_indicators
from keras.callbacks import ReduceLROnPlateau, Callback

from keras_tuner.tuners import RandomSearch, BayesianOptimization
from tensorflow.keras.metrics import Precision, Recall, AUC


class CNNHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        # Conv1D layers
        num_conv_layers = hp.Int('num_conv_layers', 1, 3)
        filters = hp.Int('initial_conv_filters', 8, 128, step=16)
        for i in range(num_conv_layers):
            use_regularization = hp.Choice(f'use_regularization_conv_{i+1}', values=[False, True])
            regularization = None
            if use_regularization:
                reg_type = hp.Choice(f'reg_type_conv_{i+1}', values=['l1', 'l2', 'l1_l2'])
                if reg_type == 'l1':
                    regularization = l1(hp.Float(f'l1_value_conv_{i+1}', 0.00001, 0.0005, step=0.00005))
                elif reg_type == 'l2':
                    regularization = l2(hp.Float(f'l2_value_conv_{i+1}', 0.0001, 0.003, step=0.0005))
                else:
                    regularization = l1_l2(l1=hp.Float(f'l1_l2_l1_value_conv_{i+1}', 0.00001, 0.0005, step=0.00005),
                                           l2=hp.Float(f'l1_l2_l2_value_conv_{i+1}', 0.0001, 0.003, step=0.0005))
            
            current_filters = filters // (2**i)
            if i == 0:
                model.add(Conv1D(
                    filters=current_filters,
                    kernel_size=hp.Int(f'conv_{i+1}_kernel', 3, 21, step=5),
                    input_shape=self.input_shape,
                    kernel_regularizer=regularization
                ))
            else:
                model.add(Conv1D(
                    filters=current_filters,
                    kernel_size=hp.Int(f'conv_{i+1}_kernel', 3, 21, step=5),
                    kernel_regularizer=regularization
                ))
            model.add(BatchNormalization())
            model.add(ReLU())
            model.add(MaxPooling1D(pool_size=3))
            model.add(Dropout(rate=hp.Float(f'dropout_conv_{i+1}', 0.4, 0.6, step=0.1)))

        # LSTM layers
        lstm_layer_count = hp.Int('num_lstm_layers', 1, 4)
        units = hp.Int('initial_lstm_units', 16, 256, step=32)
        for i in range(lstm_layer_count):
            use_regularization = hp.Choice(f'use_regularization_lstm_{i+1}', values=[False, True])
            regularization = None
            if use_regularization:
                reg_type = hp.Choice(f'reg_type_lstm_{i+1}', values=['l1', 'l2', 'l1_l2'])
                if reg_type == 'l1':
                    regularization = l1(hp.Float(f'l1_value_lstm_{i+1}', 0.00001, 0.0005, step=0.00005))
                elif reg_type == 'l2':
                    regularization = l2(hp.Float(f'l2_value_lstm_{i+1}', 0.0001, 0.003, step=0.0005))
                else:
                    regularization = l1_l2(l1=hp.Float(f'l1_l2_l1_value_lstm_{i+1}', 0.00001, 0.0005, step=0.00005),
                                           l2=hp.Float(f'l1_l2_l2_value_lstm_{i+1}', 0.0001, 0.003, step=0.0005))

            current_units = units // (2**i)
            model.add(LSTM(
                units=current_units,
                return_sequences=True if i < lstm_layer_count - 1 else False,
                kernel_regularizer=regularization
            ))
            model.add(Dropout(rate=hp.Float(f'dropout_lstm_{i+1}', 0.4, 0.6, step=0.1)))
        

        # Dense layers
        num_dense_layers = hp.Int('num_dense_layers', 1, 3)
        dense_units = hp.Int('initial_dense_units', 8, 128, step=32)
        for i in range(num_dense_layers):
            use_regularization = hp.Choice(f'use_regularization_dense_{i+1}', values=[False, True])
            regularization = None
            if use_regularization:
                reg_type = hp.Choice(f'reg_type_dense_{i+1}', values=['l1', 'l2', 'l1_l2'])
                if reg_type == 'l1':
                    regularization = l1(hp.Float(f'l1_value_dense_{i+1}', 0.00001, 0.0005, step=0.00005))
                elif reg_type == 'l2':
                    regularization = l2(hp.Float(f'l2_value_dense_{i+1}', 0.0001, 0.003, step=0.0005))
                else:
                    regularization = l1_l2(l1=hp.Float(f'l1_l2_l1_value_dense_{i+1}', 0.00001, 0.0005, step=0.00005),
                                           l2=hp.Float(f'l1_l2_l2_value_dense_{i+1}', 0.0001, 0.003, step=0.0005))
            
            current_units = dense_units // (2**i)
            model.add(Dense(
                units=current_units,
                kernel_regularizer=regularization
            ))
            model.add(BatchNormalization())
            
            # Choose between ReLU and PReLU activation functions
            activation = hp.Choice(f'dense_{i+1}_activation', ['relu', 'prelu', 'leakyrelu'])
            if activation == 'relu':
                model.add(ReLU())
            elif activation == 'prelu':
                model.add(PReLU())
            else:
                model.add(LeakyReLU())

            
            model.add(Dropout(rate=hp.Float(f'dropout_dense_{i+1}', 0.4, 0.6, step=0.1)))
        
        
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(learning_rate=hp.Float('learning_rate', 1e-5, 1e-1, sampling='log'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', AUC(), custom_model_fit_indicators.f1_score])

        return model


# 特徵欄位
# cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'MACD', 'Signal', 'Hist', 'Up Trend', 'Down Trend', 'Super Trend', 'TR', 'ATR', 'SMA_27', 'SMA_55', 'SMA_200', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band']
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'MACD', 'Signal', 'Hist', 'Up Trend', 'Down Trend', 'Super Trend']
symbol = "ETHUSDT"
interval = "5m"
look_back = 144 #使用回看n根數據
epochs = 100
batch_size = 64

# 特定
end_time_string = "2023-10-31 23:59:59"

# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 100000, cols, is_need_save_original_data=False, is_read_local=True, is_need_calculated=True)

# 再拿掉前後無參考性資料
df = df[1000:-300]

# 留下macd交叉點位的資料
# df = df[df['MACD_Stronger'] != 0]

df.reset_index(drop=True, inplace=True)


# 使用Scaler對數據進行縮放
scaler = RobustScaler()
scaled_data = scaler.fit_transform(df[cols])

# 創建X, y數據集
X, y = [], []
for i in range(look_back, len(scaled_data) + 1):
    X.append(scaled_data[i - look_back:i])
    y.append(df.iloc[i - 1][['Target']].values)
    
y = [element.astype(int) for element in y]
X, y = np.array(X), np.array(y)

# 2. 分割訓練和測試數據
# 將數據分割為訓練集、驗證集和測試集
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.07, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.07777, shuffle=False)

num_class_1 = np.sum(y_train == 1)
num_class_0 = len(y_train) - num_class_1

print(f"0 的数量: {num_class_0}")
print(f"1 的数量: {num_class_1}")

# 使用compute_class_weight计算权重
weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train.reshape(-1))
class_weights = {0: weights[0], 1: weights[1]}

# 初始化ReduceLROnPlateau回調
reduce_lr = ReduceLROnPlateau(monitor='threshold_metric',  # 監控驗證集的損失
                              factor=0.5,          # 學習率被減少的因子 (new_lr = lr * factor)
                              patience=3,         # 沒有進步的時期數，在這之後學習率會被減少
                              min_lr=0.00001,      # 學習率的下限
                              verbose=1,
                              mode='max')           # 信息展示模式

# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
custom_early_stopping = custom_callback.CustomEarlyStopping(patience=6, log_metric='threshold_metric')
input_shape = (X_train.shape[1], X_train.shape[2])
hypermodel = CNNHyperModel(input_shape)
threshold_metric = custom_callback.ThresholdMetric(0.7)
tuner = BayesianOptimization(
    hypermodel,
    objective=Objective('threshold_metric', direction='max'),
    max_trials=30,
    directory='bayesian',
    project_name='cnn'
)

# Start the search for the best hyperparameters
tuner.search(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[threshold_metric, reduce_lr, custom_early_stopping])

# 获取最佳模型
model = tuner.get_best_models(num_models=1)[0]
# model = tuner.get_best_models()[0]
history = model.fit(X_train, y_train, class_weight=class_weights, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[threshold_metric, reduce_lr, custom_early_stopping])

# 評估模型
loss, accuracy, AUC, f1_score = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test AUC: {AUC:.4f}")
print(f"Test f1_score: {f1_score:.4f}")

# 保存整個模型到目錄中
# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))

# 定義上層目錄的路徑
parent_path = os.path.join(current_path, '..')
time = int(datetime.datetime.timestamp(datetime.datetime.now()))
trained_model_name = f'{parent_path}/trained_models/{time}_sigmoid_loss-{loss:.4f}_accuracy-{accuracy:.4f}_auc-{AUC:.4}_f1_score-{f1_score:.4}'
model_process.save_model(trained_model_name, model, loss, accuracy)
model_process.export_epoch_info(history, trained_model_name)
model_process.export_ROC(model, trained_model_name, X_test, y_test)
model_process.export_layer_parameters(trained_model_name, model, cols, symbol, interval, look_back, df, batch_size, epochs, end_time_string)

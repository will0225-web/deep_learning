import sys
import os

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Conv1D, MaxPooling1D, PReLU, ReLU, Layer, Reshape, LeakyReLU, Flatten, LayerNormalization, Bidirectional
from keras.regularizers import l1_l2, l1, l2
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import os
import tensorflow as tf
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from imblearn.over_sampling import SMOTE

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import model as model_process
from modules import custom_model_fit_indicators as custom_model_fit_indicators
from keras.callbacks import ReduceLROnPlateau
from keras_tuner import HyperModel, Objective
from keras_tuner.tuners import BayesianOptimization

def multiclass_f1_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_f1_score(y_true, y_pred)

class CNNHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        # Conv1D layers
        num_conv_layers = hp.Int('num_conv_layers', 0, 5)
        filters = hp.Int('initial_conv_filters', 16, 128, step=16)
        for i in range(num_conv_layers):
            use_regularization = hp.Choice(f'use_regularization_conv_{i+1}', values=[False, True])
            regularization = None
            if use_regularization:
                reg_type = hp.Choice(f'reg_type_conv_{i+1}', values=['l1', 'l2', 'l1_l2'])
                if reg_type == 'l1':
                    regularization = l1(hp.Float(f'l1_value_conv_{i+1}', 0.00001, 0.0002, step=0.00001))
                elif reg_type == 'l2':
                    regularization = l2(hp.Float(f'l2_value_conv_{i+1}', 0.0001, 0.002, step=0.0001))
                else:
                    regularization = l1_l2(l1=hp.Float(f'l1_l2_l1_value_conv_{i+1}', 0.00001, 0.0002, step=0.00001),
                                           l2=hp.Float(f'l1_l2_l2_value_conv_{i+1}', 0.0001, 0.002, step=0.0001))
            
            current_filters = filters * (2**i)
            if i == 0:
                model.add(Conv1D(
                    filters=current_filters,
                    kernel_size=hp.Int(f'conv_{i+1}_kernel', 3, 9, step=2),
                    input_shape=self.input_shape,
                    kernel_regularizer=regularization, 
                    strides=3
                ))
            else:
                model.add(Conv1D(
                    filters=current_filters,
                    kernel_size=hp.Int(f'conv_{i+1}_kernel', 3, 9, step=2),
                    kernel_regularizer=regularization, 
                    strides=3
                ))
            model.add(BatchNormalization())
            model.add(ReLU())
            if i == num_conv_layers - 1:
                # model.add(MaxPooling1D(pool_size=2))
                model.add(Dropout(rate=hp.Float(f'dropout_conv_{i+1}', 0.2, 0.6, step=0.1)))
            
        # LSTM layers
        lstm_layer_count = hp.Int('num_lstm_layers', 1, 4)
        units = hp.Int('initial_lstm_units', 16, 512, step=32)
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
            if num_conv_layers == 0:
                model.add(Bidirectional(LSTM(
                    input_shape=self.input_shape,
                    units=current_units,
                    return_sequences=True if i < lstm_layer_count - 1 else False,
                    kernel_regularizer=regularization
                )))
            else:
                model.add(Bidirectional(LSTM(
                    units=current_units,
                    return_sequences=True if i < lstm_layer_count - 1 else False,
                    kernel_regularizer=regularization
                )))
            model.add(LayerNormalization())
            model.add(Dropout(rate=hp.Float(f'dropout_lstm_{i+1}', 0.3, 0.6, step=0.1)))

        # model.add(Flatten())

        # Dense layers
        num_dense_layers = hp.Int('num_dense_layers', 1, 5)
        dense_units = hp.Int('initial_dense_units', 32, 1024, step=32)
        for i in range(num_dense_layers):
            use_regularization = hp.Choice(f'use_regularization_dense_{i+1}', values=[False, True])
            regularization = None
            if use_regularization:
                reg_type = hp.Choice(f'reg_type_dense_{i+1}', values=['l1', 'l2', 'l1_l2'])
                if reg_type == 'l1':
                    regularization = l1(hp.Float(f'l1_value_dense_{i+1}', 0.00001, 0.0002, step=0.00001))
                elif reg_type == 'l2':
                    regularization = l2(hp.Float(f'l2_value_dense_{i+1}', 0.0001, 0.002, step=0.0001))
                else:
                    regularization = l1_l2(l1=hp.Float(f'l1_l2_l1_value_dense_{i+1}', 0.00001, 0.0002, step=0.00001),
                                           l2=hp.Float(f'l1_l2_l2_value_dense_{i+1}', 0.0001, 0.002, step=0.0001))
            
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

            model.add(Dropout(rate=hp.Float(f'dropout_dense_{i+1}', 0.3, 0.6, step=0.1)))
        
        model.add(Dense(5, activation='softmax'))

        optimizer = Adam(learning_rate=hp.Float('learning_rate', 0.0001, 0.0005, step=0.0001))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                       metrics=[
                        multiclass_f1_score,
                        'accuracy'
        ])

        return model


# 特徵欄位
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'RSI_12', 'RSI_24', 'MACD', 'Signal', 'Hist', 'MACD_Stronger', 'Up Trend', 'Down Trend', 'Super Trend', 'ATR', 'SMA_21', 'SMA_55', 'SMA_200', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band', 'Stochastic_Oscillator_14', 'Stochastic_Oscillator_21', 'EMA_26', 'EMA_50']
symbol = "ETHUSDT"
interval = "15m"
epochs = 100
batch_size = 128
look_back = 384 #使用回看n根數據

# 當前
# end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# 特定
end_time_string = "2023-10-31 23:59:59"

# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 35040, cols, is_need_save_original_data=False, is_read_local=True, is_need_calculated=True)
# 再拿掉前後無參考性資料
df = df[1000:-300]
df.reset_index(drop=True, inplace=True)
df.to_csv('test.csv')

new_data_end_time_string = "2023-11-30 23:59:59"
df_new = modules_data.get_binance_klines_backward(symbol, interval, new_data_end_time_string, 6000, cols, is_need_save_original_data=False, is_read_local=False, is_need_calculated=True) 
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

# 將數據分割為訓練集、驗證集和測試集
# X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
# X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, shuffle=False)
X_train = X
y_train = y
X_val = X_new
y_val = y_new

# 应用 SMOTE
reduce_lr = ReduceLROnPlateau(monitor='val_multiclass_f1_score',  # 監控驗證集的損失
                              factor=0.2,          # 學習率被減少的因子 (new_lr = lr * factor)
                              patience=3,         # 沒有進步的時期數，在這之後學習率會被減少
                              min_lr=0.00001,      # 學習率的下限
                              verbose=1,
                              mode='max')           # 信息展示模式

callback = tf.keras.callbacks.EarlyStopping(monitor='val_multiclass_f1_score', patience=15, restore_best_weights=True, mode='max')

input_shape = (X_train.shape[1], X_train.shape[2])
hypermodel = CNNHyperModel(input_shape)
tuner = BayesianOptimization(
    hypermodel,
    objective=Objective("val_multiclass_f1_score", direction="max"),
    # objective='val_loss',
    max_trials=30,
    directory='bayesian_LSTM',
    project_name='LSTM'
)

# Start the search for the best hyperparameters
tuner.search(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[reduce_lr, callback])

# 获取最佳模型
model = tuner.get_best_models(num_models=1)[0]
# 模型训练
# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train.ravel())
# class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=class_weights_dict, validation_data=(X_test, y_test))

# 評估模型
loss, accuracy, f1 = model.evaluate(X_new, y_new)
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
# model_process.export_epoch_info(history, trained_model_name)
model_process.export_macd_cm(trained_model_name, model, X_new, y_new, [0, 1, 2, 3, 4])
model_process.export_layer_parameters(trained_model_name, model, cols, symbol, interval, look_back, df, batch_size, epochs, end_time_string)
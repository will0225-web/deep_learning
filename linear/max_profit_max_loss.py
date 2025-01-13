import sys
import os
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam

from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, PReLU, Conv1D, MaxPooling1D, Flatten, LeakyReLU, ReLU, Bidirectional, Attention, LayerNormalization, Input, Activation, RepeatVector, Permute, Multiply, Layer, concatenate
from tensorflow.keras.initializers import GlorotUniform

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
from keras.callbacks import ReduceLROnPlateau, TensorBoard

from tensorflow.keras.utils import to_categorical

def customized_specific_period_col(df):
    customized_cols_infos = []
    
    # 紀錄原先的cols
    original_cols = set(df.columns)
    # indicator_look_back = 288
    price_look_back = 480
    # df, lower_low_higher_high_info = indicators.add_lower_low_higher_high(df, 0.04, look_back=indicator_look_back, is_need_return_function_info=True)
    # customized_cols_infos.append(lower_low_higher_high_info)
    # df, supertrend_delta_info = indicators.super_trend_delta_and_risk(df, is_need_return_function_info=True)
    # customized_cols_infos.append(supertrend_delta_info)
    # df, supertrend_delta_info = indicators.super_trend_delta_and_riskv1(df, is_need_return_function_info=True)
    # customized_cols_infos.append(supertrend_delta_info)

    df, price_indicator_info = indicators.add_price_indicator(df, look_back=price_look_back, is_need_return_function_info=True)
    customized_cols_infos.append(price_indicator_info)

    # 紀錄新的cols
    modified_cols = set(df.columns)
    # 篩選多出來的cols
    new_cols = list(modified_cols - original_cols)
    return df, customized_cols_infos, new_cols

def transform_and_data_info(df, y, cols, robust_features, standard_features, minMax_features, front_drop_count, back_drop_count, look_back, original_X=[]):
    data = df[cols]

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
    if len(original_X) == 0:
        scaled = preprocessor.fit_transform(data)
    else:
        preprocessor.fit(original_X[cols])
        # 对特征进行缩放
        scaled = preprocessor.transform(data)

    if look_back != 0:
        scaled, y = modules_data.create_look_back_dataset(scaled, y, look_back)
    
    if front_drop_count == 0 and back_drop_count == 0:
        scaled = scaled
        y = y
    elif front_drop_count != 0:
        scaled = scaled[front_drop_count:]
        y = y[front_drop_count:]
    elif back_drop_count == 0:
        scaled = scaled[:back_drop_count]
        y = y[:back_drop_count]

    # 加上customized cols資訊
    transform_data_info = {
        'cols': cols,
        'robust_features': robust_features,
        'standard_features': standard_features,
        'minMax_features': minMax_features,
        'front_drop_count': front_drop_count,
        'back_drop_count': back_drop_count,
        'look_back': look_back
    }
    return scaled, y, transform_data_info

symbol = "ETHUSDT"
interval = "5m"
look_back = 144 #使用回看n根數據
epochs = 100
batch_size = 64
total_klines = 200000
get_local_file_name = 'ETHUSDT_5m_2023-12-31_23-59-59_450000_calculated.csv'
input_model_infos = []

# 特定
end_time_string = "2023-12-31 23:59:59"


# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
df, customized_cols_infos, new_cols = customized_specific_period_col(df)
df, target_function_info = modules_data.set_target(df)

# 拿掉前後無參考性資料
drop_front_data_count = 1000
drop_back_data_count = 300
df = df[drop_front_data_count:-drop_back_data_count]
df.reset_index(drop=True, inplace=True)


X = df
max_profit_y = df['Max_High']
max_loss_y = df['Max_Low']

X.drop(['Max_High', 'Max_Low'], axis=1, inplace=True)


# 选择要缩放的列
ochl_cols = ["Close", "High", "Low", "Open", "MACD", "Signal", "Hist", "Middle Band", "Lower Band", "Upper Band"]

ochl_robust_features = ["MACD", "Signal", "Hist"]
ochl_standard_features = ["Close", "High", "Low", "Open", "Middle Band", "Lower Band", "Upper Band"]
ochl_minMax_features = []
X_ochl_scaled, X_look_back_max_profit, ochl_data_info = transform_and_data_info(X, max_profit_y, ochl_cols, ochl_robust_features, ochl_standard_features, ochl_minMax_features, 0, 0, look_back)
_, X_look_back_max_loss, _ = transform_and_data_info(X, max_loss_y, ochl_cols, ochl_robust_features, ochl_standard_features, ochl_minMax_features, 0, 0, look_back)
input_model_infos.append(ochl_data_info)

current_cols = ['Super Trend', 'Volume', 'ATR', 'RSI', "red_three_soldiers", "green_three_soldiers", 'MACD_Stronger', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618']
current_robust_features = []
current_standard_features = ['Volume', 'ATR', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618']
current_minMax_features = []
# 這個要把look_back取走才會跟接下來X_ochl搭到, 因為X_ochl會從look_back的index開始抓資料
X_current_scaled, X_current_max_profit, current_data_info = transform_and_data_info(X, max_profit_y, current_cols, current_robust_features, current_standard_features, current_minMax_features, look_back, 0, 0)
input_model_infos.append(current_data_info)

# 假设X_scaled是您的特征数据，y_categorical是您的目标数据
test_and_validation_size = 0.2
validation_ratio_of_test = 0.5  # 在测试和验证数据集中，验证集占的比例

# 首先分割出训练集和剩余集（测试集+验证集）
X_train_max_profit, X_temp_max_profit, y_train_max_profit, y_temp_max_profit = train_test_split(X_ochl_scaled, X_look_back_max_profit, test_size=test_and_validation_size, shuffle=False)
X_train_max_loss, X_temp_max_loss, y_train_max_loss, y_temp_max_loss = train_test_split(X_ochl_scaled, X_look_back_max_loss, test_size=test_and_validation_size, shuffle=False)
X_current_train_max_profit, X_current_temp_max_profit, y_current_train_max_profit, y_current_temp_max_profit = train_test_split(X_current_scaled, X_current_max_profit, test_size=test_and_validation_size, shuffle=False)



# 接着将剩余集分割为测试集和验证集
X_val_max_profit, X_test_max_profit, y_val_max_profit, y_test_max_profit = train_test_split(X_temp_max_profit, y_temp_max_profit, test_size=validation_ratio_of_test, shuffle=False)
X_val_max_loss, X_test_max_loss, y_val_max_loss, y_test_max_loss = train_test_split(X_temp_max_loss, y_temp_max_loss, test_size=validation_ratio_of_test, shuffle=False)
X_current_val_max_profit, X_current_test_max_profit, y_current_val_max_profit, y_current_test_max_profit = train_test_split(X_current_temp_max_profit, y_current_temp_max_profit, test_size=validation_ratio_of_test, shuffle=False)

# 共享層
# shared_conv = Conv1D(filters=64, kernel_size=3, activation='relu')

# 輸入層
input_look_back_5m = Input(shape=(look_back, X_train_max_profit.shape[2]), name='input_look_back_5m')  # OCHL数据
input_current_5m = Input(shape=(X_current_train_max_profit.shape[1],), name='input_current_5m')  # 当前周期指标数据

# CNN层，用于提取时间序列中的局部特征
conv1 = Conv1D(filters=128, kernel_size=3, activation='relu')(input_look_back_5m)
pool1 = MaxPooling1D(pool_size=2)(conv1)
conv2 = Conv1D(filters=64, kernel_size=3, activation='relu')(pool1)
pool2 = MaxPooling1D(pool_size=2)(conv2)
# 通过Flatten层将卷积层的输出扁平化，以便与LSTM层连接
cnn_out = Flatten()(pool2)

# OCHL数据的LSTM层
lstm_layer = LSTM(units=150, return_sequences=True)(input_look_back_5m)
lstm_dropout = Dropout(0.3)(lstm_layer)
lstm_layer = LSTM(units=150, return_sequences=False)(lstm_dropout)
lstm_dropout = Dropout(0.3)(lstm_layer)
lstm_layer = Dense(100, activation='relu')(lstm_dropout)
lstm_out  = Dropout(0.3)(lstm_layer)

# 先合并处理过的look_back数据（CNN和LSTM的输出）
merged_look_back = concatenate([cnn_out, lstm_out])
# 对合并后的look_back数据进行进一步处理（可选）
# 例如，可以加入一个或多个Dense层来进一步提取特征
merged_look_back_dense = Dense(100, activation='relu')(merged_look_back)
merged_look_back_dropout = Dropout(0.3)(merged_look_back_dense)

# 当前周期指标数据的Dense层
current_dense = Dense(150, activation='relu')(input_current_5m)
current_droupout = Dropout(0.3)(current_dense)
current_dense = Dense(100, activation='relu')(current_droupout)
current_out = Dropout(0.3)(current_dense)


# 合并两个输入处理路径的输出
# merged = concatenate([cnn_out, lstm_out, current_out])
# 现在将处理过的look_back数据和静态数据的输出合并
merged_final = concatenate([merged_look_back_dropout, current_out])

# 预测层
x = Dense(64, activation='relu')(merged_final)
x = Dense(32, activation='relu')(x)
x = Dropout(0.3)(x)
max_profit_output = Dense(1, activation='linear', name='max_profit')(x)
max_loss_output = Dense(1, activation='linear', name='max_loss')(x)

# 构建模型
model = Model(inputs=[input_look_back_5m, input_current_5m], outputs=[max_profit_output, max_loss_output])

optimizer = Adam(learning_rate=0.0001)
# 编译模型
model.compile(optimizer=optimizer, loss='mean_squared_error')

# 學習率
# 初始化ReduceLROnPlateau回調
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # 監控驗證集的損失
                            factor=0.3,          # 學習率被減少的因子 (new_lr = lr * factor)
                            patience=5,         # 沒有進步的時期數，在這之後學習率會被減少
                            min_lr=0.000001,      # 學習率的下限
                            verbose=1,       
                            mode='min')           # 信息展示模式
# 训练模型
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, mode='min')

# 设置 TensorBoard 日志目录
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [reduce_lr, callback]

history = model.fit([X_train_max_profit, X_current_train_max_profit], [y_train_max_profit, y_train_max_loss], epochs=epochs, batch_size=batch_size, validation_data=([X_val_max_profit, X_current_val_max_profit], [y_val_max_profit, y_val_max_loss]), callbacks=callbacks)


# 评估模型
evaluation_results = model.evaluate([X_test_max_profit, X_current_test_max_profit], [y_test_max_profit, y_test_max_loss], batch_size=batch_size, return_dict=True)
loss = evaluation_results['loss']
for key, value in evaluation_results.items():
    print(f"{key}, {value}")


# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))

target_types = []
# 定義上層目錄的路徑
parent_path = os.path.join(current_path, '..')
time = int(datetime.datetime.timestamp(datetime.datetime.now()))
trained_model_name = f'{parent_path}/trained_models/{time}_softmax_loss-{loss:.4f}'
model_process.save_model(trained_model_name, model, loss, 0)
# model_process.export_epoch_info(history, trained_model_name)
# model_process.export_cm(trained_model_name, model, [X_test_max_profit, X_current_test_max_profit], [y_test_max_profit, y_test_max_loss], target_types, batch_size=batch_size)
model_process.export_layer_parameters(trained_model_name, model, symbol, interval, look_back, df, batch_size, epochs, end_time_string, evaluation_results, test_and_validation_size, validation_ratio_of_test, [], get_local_file_name)
model_process.export_connection_info(trained_model_name, batch_size, symbol, interval, look_back, end_time_string, total_klines, target_function_info, input_model_infos, target_types, customized_cols_infos, drop_front_data_count, drop_back_data_count, get_local_file_name, new_cols)
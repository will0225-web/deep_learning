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


def multiclass_f1_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_f1_score(y_true, y_pred)

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
    df, supertrend_risk_score_info = indicators.calculate_supertrend_risk(df, is_need_return_function_info=True)
    customized_cols_infos.append(supertrend_risk_score_info)

    # df = indicators.add_24h_min_max_price(df, look_back=96)
    ### export customized col
    # df['EMA_9'], ema_9_function_info = indicators.calculate_ema(df['Close'], 9, is_need_return_function_info=True)
    # customized_cols_infos.append(ema_9_function_info)
    # df['EMA_21'], ema_21_function_info = indicators.calculate_ema(df['Close'], 21, is_need_return_function_info=True)
    # customized_cols_infos.append(ema_21_function_info)
    # df['ATR_35'], atr_35_function_info  = indicators.calculate_atr(df['TR'], 35, is_need_return_function_info=True)
    # customized_cols_infos.append(atr_35_function_info)

    # df['EMA_14'], ema_14_function_info  = indicators.calculate_ema(df['Close'], 14, is_need_return_function_info=True)
    # customized_cols_infos.append(ema_14_function_info)

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
interval = "15m"
look_back = 96 #使用回看n根數據
epochs = 100
batch_size = 128
total_klines = 100000
get_local_file_name = 'ETHUSDT_15m_2023-12-31_23-59-59_150000_calculated.csv'
input_model_infos = []

# 當前
# end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# end_time_seconds = end_time / 1000
# end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
# end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# 特定
end_time_string = "2023-12-31 23:59:59"


# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df, target_function_info = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
df, customized_cols_infos, new_cols = customized_specific_period_col(df)

# 拿掉前後無參考性資料
drop_front_data_count = 1000
drop_back_data_count = 300
df = df[drop_front_data_count:-drop_back_data_count]
df.reset_index(drop=True, inplace=True)

hold_count_limit = 12
df['Supertrend Target'] = signals.supertrend_simple_target(df, 12, 0.005, 0.002, hold_count_limit)
df['Stop Loss Target'] = signals.stop_loss_target(df, 0.005, 0.005, hold_count_limit)

# 'Target'列不重複的數值當作有的target
supertrend_target_types = sorted(df['Supertrend Target'].unique().tolist())
stop_loss_target_types = sorted(df['Stop Loss Target'].unique().tolist())

X = df
supertrend_target = df['Supertrend Target'].values
stop_loss_target = df['Stop Loss Target'].values
# 将目标变量转换为分类格式

y_categorical = to_categorical(supertrend_target, num_classes=len(supertrend_target_types))
y_stop_loss_binary = to_categorical(stop_loss_target, num_classes=len(stop_loss_target_types))


# 选择要缩放的列
ochl_cols = ["Close", "High", "Low", "Open", "Volume", "ATR", "MACD", "Signal", "Hist", 'Middle Band', 'Lower Band', 'Upper Band', 'RSI', 'Super Trend']

ochl_robust_features = ["MACD", "Signal", "Hist"]
ochl_standard_features = ["Close", "High", "Low", "Open", "Volume", "ATR", 'Middle Band', 'Lower Band', 'Upper Band']
ochl_minMax_features = []
X_ochl_scaled, X_ochl_y, ochl_data_info = transform_and_data_info(X, y_categorical, ochl_cols, ochl_robust_features, ochl_standard_features, ochl_minMax_features, 0, 0, look_back)
input_model_infos.append(ochl_data_info)

stop_cols = ["Close", "High", "Low", "Open", "Volume", "ATR", "MACD", "Signal", "Hist", 'Middle Band', 'Lower Band', 'Upper Band', 'RSI', 'Super Trend']

stop_robust_features = ["MACD", "Signal", "Hist"]
stop_standard_features = ["Close", "High", "Low", "Open", "Volume", "ATR", 'Middle Band', 'Lower Band', 'Upper Band']
stop_minMax_features = []
X_stop_scaled, X_stop_y, stop_data_info = transform_and_data_info(X, y_stop_loss_binary, stop_cols, stop_robust_features, stop_standard_features, stop_minMax_features, 0, 0, look_back)
input_model_infos.append(stop_data_info)


# current_cols = ['Super Trend', 'Volume', 'ATR', 'RSI', 'red_three_soldiers', 'green_three_soldiers', "MACD", "Signal", "Hist", 'MACD_Stronger', 'MACD_Cross']
# current_robust_features = ["MACD", "Signal", "Hist"]
# current_standard_features = ['Volume', 'ATR']
# current_minMax_features = []
# # 這個要把look_back取走才會跟接下來X_ochl搭到, 因為X_ochl會從look_back的index開始抓資料
# X_current_scaled, X_current_y, current_data_info = transform_and_data_info(X, y_categorical, current_cols, current_robust_features, current_standard_features, current_minMax_features, look_back, 0, 0)
# input_model_infos.append(current_data_info)



# 假设X_scaled是您的特征数据，y_categorical是您的目标数据
test_and_validation_size = 0.2
validation_ratio_of_test = 0.5  # 在测试和验证数据集中，验证集占的比例

# 首先分割出训练集和剩余集（测试集+验证集）
X_train, X_temp, y_train_supertrend, y_temp_supertrend = train_test_split(X_ochl_scaled, X_ochl_y, test_size=test_and_validation_size, shuffle=False)
_, _, y_train_stop_loss, y_temp_stop_loss = train_test_split(X_stop_scaled, X_stop_y, test_size=test_and_validation_size, shuffle=False)
# X_current_train, X_current_temp, y_current_train, y_current_temp = train_test_split(X_current_scaled, X_current_y, test_size=test_and_validation_size, shuffle=False)


# 接着将剩余集分割为测试集和验证集
X_val, X_test, y_val_supertrend, y_test_supertrend = train_test_split(X_temp, y_temp_supertrend, test_size=validation_ratio_of_test, shuffle=False)
_, _, y_val_stop_loss, y_test_stop_loss = train_test_split(X_temp, y_temp_stop_loss, test_size=validation_ratio_of_test, shuffle=False)
# X_current_val, X_current_test, y_current_val, y_current_test = train_test_split(X_current_temp, y_current_temp, test_size=validation_ratio_of_test, shuffle=False)

input_layer = Input(shape=(look_back, X_train.shape[2]), name='ochl_input')  # OCHL数据

# 共享层
# 这里假设是一个简单的LSTM模型
lstm_layer = LSTM(128, return_sequences=True)(input_layer)
lstm_layer = LSTM(128, return_sequences=False)(lstm_layer)
shared_dense = Dense(64, activation='relu')(lstm_layer)
shared_dropout = Dropout(0.3)(shared_dense)

# 主任务输出层 - 由于现在是三分类问题，所以使用softmax
main_dense = Dense(128, activation='relu')(shared_dropout)
main_dense = Dense(128, activation='relu')(main_dense)
main_output = Dense(y_train_supertrend.shape[1], activation='softmax', name='main_output')(main_dense)

# 辅助任务输出层 - 仍然是二分类问题
secondary_dense = Dense(128, activation='relu')(shared_dropout)
secondary_dense = Dense(128, activation='relu')(secondary_dense)
auxiliary_output = Dense(y_train_stop_loss.shape[1], activation='softmax', name='auxiliary_output')(secondary_dense)  # 二分类任务：是否触及止损

# 构建模型
model = Model(inputs=input_layer, outputs=[main_output, auxiliary_output])

optimizer = Adam(learning_rate=0.001)

# 编译模型
model.compile(optimizer=optimizer,
              loss={'main_output': 'categorical_crossentropy', 'auxiliary_output': 'categorical_crossentropy'},
              metrics={'main_output': ['accuracy', multiclass_f1_score], 'auxiliary_output': ['accuracy', multiclass_f1_score]},
              loss_weights={'main_output': 1, 'auxiliary_output': 0.5}) # 调整loss权重以平衡两个任务的重要性

# 學習率
# 初始化ReduceLROnPlateau回調
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # 監控驗證集的損失
                              factor=0.3,          # 學習率被減少的因子 (new_lr = lr * factor)
                              patience=5,         # 沒有進步的時期數，在這之後學習率會被減少
                              min_lr=0.00001,      # 學習率的下限
                              verbose=1,
                              mode='min')           # 信息展示模式
# 训练模型
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min')

# 设置 TensorBoard 日志目录
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [reduce_lr, callback]

y_train = [y_train_supertrend, y_train_stop_loss] 
y_val = [y_val_supertrend, y_val_stop_loss]
y_test = [y_test_supertrend, y_test_stop_loss]

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks)


# 评估模型
evaluation_results = model.evaluate(X_test, y_test, batch_size=batch_size, return_dict=True)
loss = evaluation_results['main_output_loss']
accuracy = evaluation_results['main_output_accuracy']
f1_score = evaluation_results['main_output_multiclass_f1_score']
for key, value in evaluation_results.items():
    print(f"{key}, {value}")

import seaborn as sns
import matplotlib.pyplot as plt
# 假设 X_test 是你的测试数据
predictions = model.predict(X_test)

# 对于主任务的预测处理
main_predictions = np.argmax(predictions[0], axis=1)
main_true = np.argmax(y_test[0], axis=1)  # 假设 y_test[0] 是主任务的真实标签

# 对于辅助任务的预测处理
auxiliary_predictions = np.argmax(predictions[1], axis=1)  # 假设辅助任务是二分类
auxiliary_true = np.argmax(y_test[1], axis=1)  # 假设 y_test[1] 是辅助任务的真实标签

# 计算混淆矩阵
main_cm = confusion_matrix(main_true, main_predictions)
auxiliary_cm = confusion_matrix(auxiliary_true, auxiliary_predictions)

# 绘制混淆矩阵
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(main_cm, annot=True, fmt="d", ax=axes[0])
axes[0].set_title('Main Task Confusion Matrix')

sns.heatmap(auxiliary_cm, annot=True, fmt="d", ax=axes[1])
axes[1].set_title('Auxiliary Task Confusion Matrix')

plt.show()

# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))

# 定義上層目錄的路徑
parent_path = os.path.join(current_path, '..')
time = int(datetime.datetime.timestamp(datetime.datetime.now()))
trained_model_name = f'{parent_path}/trained_models/{time}_softmax_loss-{loss:.4f}_accuracy-{accuracy:.4f}_f1-{f1_score:.4f}'
# model_process.save_model(trained_model_name, model, loss, accuracy)
# model_process.export_epoch_info(history, trained_model_name)
# model_process.export_cm(trained_model_name, model, X_test, y_test, supertrend_target_types, batch_size=batch_size)
# model_process.export_layer_parameters(trained_model_name, model, symbol, interval, look_back, df, batch_size, epochs, end_time_string, evaluation_results, test_and_validation_size, validation_ratio_of_test, [], get_local_file_name)
# model_process.export_connection_info(trained_model_name, batch_size, symbol, interval, look_back, end_time_string, total_klines, target_function_info, input_model_infos, supertrend_target_types, customized_cols_infos, drop_front_data_count, drop_back_data_count, get_local_file_name, new_cols)
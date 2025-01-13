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

from scipy import stats

from tensorflow.keras.utils import to_categorical

import shap

def multiclass_f1_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_f1_score(y_true, y_pred)


def calculate_targets(df, period, multiplier=1.5):
    # 计算未来period的收盘价变化率
    df['SL_Touch'] = 0  # 初始化趋势标签为0
    df['ATR_Multiplier'] = df['ATR'] * multiplier
    # 重置索引以确保索引是连续的整数
    df = df.reset_index(drop=True)
    # # 根据价格变化率和阈值判断趋势
    # df['Trend_Target'] = np.where(df['Price_Change'] > threshold, 1,
    #                               np.where(df['Price_Change'] < -threshold, 2, 0))

    for i in range(len(df) - period):
        close_price = df.at[i, 'Close']
        future_atr_multiplier = df.at[i, 'ATR_Multiplier']

        # 计算未来period内每一天的价格变化率
        for j in range(1, period + 1):
            future_high = df.at[i + j, 'High']
            future_low = df.at[i + j, 'Low']

            if future_low <= close_price - future_atr_multiplier:
                df.at[i, 'SL_Touch'] = 1
                break
            elif future_high >= close_price + future_atr_multiplier:
                df.at[i, 'SL_Touch'] = 2
                break
    return df

def calculate_trend(df, period=144, threshold=0.05):
    # 计算周期内的最高点和最低点
    df['Highest_High'] = df['High'].rolling(window=period).max()
    df['Lowest_Low'] = df['Low'].rolling(window=period).min()
    
    # 找到最高点和最低点的位置
    df['High_Pos'] = df['High'].rolling(window=period).apply(lambda x: np.argmax(x), raw=True)
    df['Low_Pos'] = df['Low'].rolling(window=period).apply(lambda x: np.argmin(x), raw=True)
    
    # 计算最高点和最低点的差异
    df['High_Low_Diff'] = (df['Highest_High'] - df['Lowest_Low']) / df['Lowest_Low']
    
    # 判断趋势：最高点在最低点左边且差异大于7% -> 向下（2），最高点在最低点右边且差异大于7% -> 向上（1），否则 -> 无趋势（0）
    df['Trend'] = np.where((df['High_Low_Diff'] > threshold) & (df['High_Pos'] < df['Low_Pos']), 2,
                           np.where((df['High_Low_Diff'] > threshold) & (df['High_Pos'] > df['Low_Pos']), 1, 0))
    
    # 删除rolling没有结果的行
    df = df.dropna(subset=['Highest_High', 'Lowest_Low', 'High_Pos', 'Low_Pos'])
    
    # 删除不必要的列
    df = df.drop(columns=['Highest_High', 'Lowest_Low', 'High_Pos', 'Low_Pos', 'High_Low_Diff'])
    
    return df

def customized_specific_period_col(df):
    customized_cols_infos = []
    
    # 紀錄原先的cols
    original_cols = set(df.columns)
    price_look_back = 672
    
    # indicator_look_back = 288
    
    # df, lower_low_higher_high_info = indicators.add_lower_low_higher_high(df, 0.04, look_back=indicator_look_back, is_need_return_function_info=True)
    # customized_cols_infos.append(lower_low_higher_high_info)
    
    df, price_indicator_info = indicators.add_price_indicator(df, look_back=price_look_back, is_need_return_function_info=True)
    customized_cols_infos.append(price_indicator_info)

    # df, supertrend_delta_info = indicators.super_trend_delta_and_risk(df, is_need_return_function_info=True)
    # customized_cols_infos.append(supertrend_delta_info)
    df, supertrend_delta_info = indicators.super_trend_delta_and_risk_v1(df, is_need_return_function_info=True)
    customized_cols_infos.append(supertrend_delta_info)
    # df, supertrend_risk_score_info = indicators.calculate_supertrend_risk(df, is_need_return_function_info=True)
    # customized_cols_infos.append(supertrend_risk_score_info)

    df, look_back_24h_min_max_price_info = indicators.add_24h_min_max_price(df, look_back=96, is_need_return_function_info=True)
    customized_cols_infos.append(look_back_24h_min_max_price_info)

    df = calculate_volatility(df, 8)
    di_len = 14
    adx_len = 14
    df = calculate_adx(df, di_len, adx_len)
    df = calculate_moving_averages(df)
    df = calculate_bollinger_bands(df)
    df = calculate_momentum(df)
    df = calculate_ppo(df)
    df = calculate_trend(df, 288, 0.04)

    # 计算价格变化和对数收益率
    df['Open_Change_Rate'] = df['Open'].pct_change().fillna(0)
    df['High_Change_Rate'] = df['High'].pct_change().fillna(0)
    df['Low_Change_Rate'] = df['Low'].pct_change().fillna(0)
    df['Close_Change_Rate'] = df['Close'].pct_change().fillna(0)
    # 计算成交量变化率
    df['Volume_Change'] = df['Volume'].pct_change().fillna(0)

    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_High_Ratio'] = df['Close'] / df['High']
    df['Close_Low_Ratio'] = df['Close'] / df['Low']

    # 计算布林带宽度
    df['BB_Width'] = df['Upper Band'] - df['Lower Band']
    # 计算布林带宽度变化率
    df['BB_Width_Ratio'] = df['BB_Width'].pct_change().fillna(0)
    
    # 對數收益率
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Log_Returns'].rolling(window=20).std()
    
    df['SMA_5'] = indicators.calculate_sma(df['Close'], 5)
    df['SMA_10'] = indicators.calculate_sma(df['Close'], 10)

    df = calculate_VWAP(df)

    # 紀錄新的cols
    modified_cols = set(df.columns)
    # 篩選多出來的cols
    new_cols = list(modified_cols - original_cols)
    return df, customized_cols_infos, new_cols

def calculate_VWAP(df):
    # data['Typical_Price'] = (data['Close'] + data['High'] + data['Low']) / 3
    df['Typical_Price'] = df['Close']
    df['VP'] = df['Typical_Price'] * df['Volume']

    df['Cumulative_VP'] = df['VP'].cumsum()
    df['Cumulative_Volume'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cumulative_VP'] / df['Cumulative_Volume']

    return df


def calculate_dm(df):
    df['up'] = df['High'] - df['High'].shift(1)
    df['down'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0)
    df['-DM'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0)
    return df

def calculate_rma(series, period):
    rma = series.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return rma

def calculate_di(df, period):
    df['TR_sum'] = calculate_rma(df['TR'], period)
    df['+DM_sum'] = calculate_rma(df['+DM'], period)
    df['-DM_sum'] = calculate_rma(df['-DM'], period)
    df['+DI'] = 100 * (df['+DM_sum'] / df['TR_sum'])
    df['-DI'] = 100 * (df['-DM_sum'] / df['TR_sum'])
    return df

def calculate_dx(df):
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    return df

def calculate_adx(df, di_len, adx_len):
    df = calculate_dm(df)
    df = calculate_di(df, di_len)
    df = calculate_dx(df)
    df['ADX'] = calculate_rma(df['DX'], adx_len)
    return df

def calculate_volatility(df, look_back):
    df['Close_Volatility'] = df['Close'].rolling(window=look_back).std()
    df['High_Volatility'] = df['High'].rolling(window=look_back).std()
    df['Low_Volatility'] = df['Low'].rolling(window=look_back).std()
    df['Open_Volatility'] = df['Open'].rolling(window=look_back).std()
    return df

def calculate_moving_averages(df, short_window=50, long_window=200):
    df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_window).mean()
    return df

def calculate_bollinger_bands(df):
    df['BB_Width'] = df['Upper Band'] - df['Lower Band']
    df['BB_Pos'] = (df['Close'] - df['Lower Band']) / (df['Upper Band'] - df['Lower Band'])
    return df

def calculate_momentum(df, window=10):
    df['Momentum'] = df['Close'].diff(window)
    return df

def calculate_ppo(df, short_window=12, long_window=26):
    short_ema = df['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['PPO'] = (short_ema - long_ema) / long_ema * 100
    return df

# 假设df是你的数据框，包含所有需要的列
def calculate_rolling_stats(df, window=20):
    rolling_mean = df.rolling(window=window).mean()
    rolling_std = df.rolling(window=window).std()
    return rolling_mean, rolling_std

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

def print_outliers_and_return(data):
    # 计算z-score
    z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
    print("Z-scores:\n", z_scores)

    # 找到异常值
    threshold = 3  # 通常使用3作为z-score的阈值
    outliers = (z_scores > threshold)
    print("Outliers:\n", outliers)

    # 计算每列的异常值数量
    outliers_counts = outliers.sum(axis=0)
    print("Outliers Counts per column:\n", outliers_counts)

    # 计算总的异常值数量
    total_outliers = outliers_counts.sum()
    print("Total Outliers:", total_outliers)


    total_rows = data.shape[0]
    # 计算异常值所占的比例
    outliers_ratio = total_outliers / (total_rows * data.shape[1])
    print("Outliers Ratio: {:.2%}".format(outliers_ratio))

    return outliers

symbol = "ETHUSDT"
interval = "15m"
look_back = 144 #使用回看n根數據
epochs = 150
batch_size = 128
total_klines = 150000
get_local_file_name = 'train_data_with_hidden_states.csv'
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
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
df, customized_cols_infos, new_cols = customized_specific_period_col(df)
df = calculate_targets(df, 48, 1.5)

# 拿掉前後無參考性資料
drop_front_data_count = 1000
drop_back_data_count = 300
df = df[drop_front_data_count:-drop_back_data_count]
df.reset_index(drop=True, inplace=True)

df.fillna(0, inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.interpolate(method='linear', inplace=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns

# 检查 NaN 值的数量
nan_counts = df[numeric_cols].isna().sum()
print("NaN Counts:\n", nan_counts)

inf_counts = np.isinf(df[numeric_cols]).sum()
print("Infinity Counts:\n", inf_counts)

total_nan_inf = nan_counts + inf_counts
print("Total NaN and Infinity Counts:\n", total_nan_inf)

# 检查所有列中 NaN 值的行
nan_volume_rows = df[df[numeric_cols].isna().any(axis=1)]
inf_volume_rows = df[np.isinf(df[numeric_cols]).any(axis=1)]
print("Rows with NaN column:\n", nan_volume_rows)
print("Rows with inf column:\n", inf_volume_rows)

df.reset_index(drop=True, inplace=True)

X = df
# y_highest = df['Highest_Price_Target']
# y_lowest = df['Lowest_Price_Target']

# # 缩放目标变量
# scaler_highest = StandardScaler()
# y_highest_scaled = scaler_highest.fit_transform(y_highest.values.reshape(-1, 1))

# y = y_highest_scaled

y = df['SL_Touch']
# y = df['Trend']
target_counts = df['SL_Touch'].value_counts()
print(target_counts)
target_types = sorted(y.unique().tolist())
y_categorical = to_categorical(y, num_classes=len(target_types))

test_and_validation_size = 0.3
validation_ratio_of_test = 0.5  # 在测试和验证数据集中，验证集占的比例

# 生成三种不同的 look_back 数据
look_back_values = [144]
X_look_backs = []
y_look_backs = []

ochl_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'ATR', 'fibonacci_0.5', 'fibonacci_0.382', 'fibonacci_0.618', 'RSI', 'VWAP', 'Middle Band', 'Upper Band', 'Lower Band', 'Super Trend', 'Up Trend', 'Down Trend']
outliers = print_outliers_and_return(df[ochl_cols])

ochl_robust_features = ['ATR', 'Volume']
ochl_standard_features = ['Close', 'High', 'Low', 'Open', 'fibonacci_0.5', 'fibonacci_0.382', 'fibonacci_0.618', 'VWAP', 'Middle Band', 'Upper Band', 'Lower Band', 'Up Trend', 'Down Trend']
ochl_minMax_features = []
X_ochl_scaled, X_ochl_y, ochl_data_info = transform_and_data_info(X, y_categorical, ochl_cols, ochl_robust_features, ochl_standard_features, ochl_minMax_features, 0, 0, look_back)

# for look_back in look_back_values:
#     X_ochl_scaled, X_ochl_y, ochl_data_info = transform_and_data_info(X, y_categorical, ochl_cols, ochl_robust_features, ochl_standard_features, ochl_minMax_features, 0, 0, look_back)
#     X_look_backs.append(X_ochl_scaled)
#     y_look_backs.append(X_ochl_y)

# # 对齐所有 look_back 数据
# min_length = min(len(y) for y in y_look_backs)
# X_look_backs_aligned = [X[-min_length:] for X in X_look_backs]
# y_look_backs_aligned = y_look_backs[0][-min_length:]  # 目标变量相同，选一个对齐即可

# # 拆分数据集
# X_train_splits = []
# X_val_splits = []
# X_test_splits = []
# y_train_splits = []
# y_val_splits = []
# y_test_splits = []

# for X_look_back_aligned in X_look_backs_aligned:
X_train, X_temp, y_train, y_temp = train_test_split(X_ochl_scaled, X_ochl_y, test_size=test_and_validation_size, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_ratio_of_test, shuffle=False)
#     X_train_splits.append(X_train)
#     X_val_splits.append(X_val)
#     X_test_splits.append(X_test)
#     y_train_splits.append(y_train)
#     y_val_splits.append(y_val)
#     y_test_splits.append(y_test)

# 目標價
# target_price_cols = ["Close", 'ATR', 'ATR_Price_Percentage', 'Volume']

# target_price_robust_features = ['ATR', 'ATR_Price_Percentage', 'Volume']
# target_price_standard_features = ["Close"]
# target_price_minMax_features = []

# X_target_price_scaled, X_target_price_y, target_price_data_info = transform_and_data_info(X, y_categorical, target_price_cols, target_price_robust_features, target_price_standard_features, target_price_minMax_features, max(look_back_values), 0, 0)
# input_model_infos.append(target_price_data_info)
# X_target_price_train, X_target_price_temp, y_target_price_train, y_target_price_temp = train_test_split(X_target_price_scaled, X_target_price_y, test_size=test_and_validation_size, shuffle=False)
# X_target_price_val, X_target_price_test, y_target_price_val, y_target_price_test = train_test_split(X_target_price_temp, y_target_price_temp, test_size=validation_ratio_of_test, shuffle=False)


########## 模型建置 ##########

# 构建多输入模型
# inputs = []
# lstm_layers = []

# for i, look_back in enumerate(look_back_values):
input_layer = Input(shape=(look_back, X_ochl_scaled.shape[2]), name=f'input_{look_back}')
lstm_layer = LSTM(32, return_sequences=True)(input_layer)
lstm_layer = Dropout(0.3)(lstm_layer)
lstm_layer = LSTM(16, return_sequences=True)(lstm_layer)
lstm_layer = Dropout(0.3)(lstm_layer)
lstm_layer = LSTM(16, return_sequences=False)(lstm_layer)
lstm_layer = Dropout(0.3)(lstm_layer)
    # inputs.append(input_layer)
    # lstm_layers.append(lstm_layer)

# 合并 LSTM 层的输出
# trend_lstm_merged = concatenate(lstm_layers)

# target_price_input = Input(shape=(X_target_price_train.shape[1],), name='target_price_input')  # 当前周期指标数据
# # price_input = Input(shape=(look_back, X_ochl_train.shape[2]), name='price_input')  # OCHL LSTM数据
# # target_calculation_input = Input(shape=(look_back, X_target_calculation_train.shape[2]), name='target_calculation_input')  # 目标计算 LSTM数据
# # strength_input = Input(shape=(look_back, X_strength_train.shape[2]), name='strength_input')

# target_price_dense = Dense(256)(target_price_input)
# # target_price_dense = BatchNormalization()(target_price_dense)
# target_price_dense = Activation('relu')(target_price_dense)
# target_price_dense = Dropout(0.4)(target_price_dense)
# target_price_dense = Dense(128)(target_price_dense)
# # target_price_dense = BatchNormalization()(target_price_dense)
# target_price_dense = Activation('relu')(target_price_dense)
# target_price_dense = Dropout(0.4)(target_price_dense)

# 合并注意力层的输出
# merged = concatenate([price_lstm, target_calculation_lstm, strength_lstm, attention_price_target, attention_price_strength, attention_target_strength])
# merged = concatenate([trend_lstm_merged, target_price_dense])

dense_1 = Dense(128)(lstm_layer)
dense_1 = ReLU()(dense_1)
dense_1 = Dropout(0.3)(dense_1)
dense_2 = Dense(64)(dense_1)
dense_2 = ReLU()(dense_2)
dense_2 = Dropout(0.3)(dense_2)
dense_2 = Dense(32)(dense_2)
dense_2 = ReLU()(dense_2)
dense_2 = Dropout(0.3)(dense_2)
# 输出层
output = Dense(3, activation='softmax')(dense_2)

model = Model(inputs=input_layer, outputs=output)

optimizer = Adam(learning_rate=0.002)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', multiclass_f1_score])


reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # 監控驗證集的損失
                                factor=0.5,          # 學習率被減少的因子 (new_lr = lr * factor)
                                patience=5,         # 沒有進步的時期數，在這之後學習率會被減少
                                min_lr=0.000001,      # 學習率的下限
                                verbose=1,
                                mode='min')           # 信息展示模式
# 训练模型
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, mode='min')

# 设置 TensorBoard 日志目录
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [reduce_lr, callback]

# 權重
y_train_labels = np.argmax(y_categorical, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weight_dict = dict(enumerate(class_weights))

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), class_weight=class_weight_dict, callbacks=callbacks)

model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

y_pred_class = np.argmax(y_pred, axis=1)
y_pred_confidence = np.max(y_pred, axis=1)

# y_pred_high = scaler_highest.inverse_transform(y_pred_high_scaled)
# print(y_pred_high)
# y_ochl_test_inverse = scaler_highest.inverse_transform(y_ochl_test)

# from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error

# # 计算均方误差 (MSE)，使用缩放后的数据
# mse_high = mean_squared_error(y_ochl_test_inverse, y_pred_high)
# # 计算平均绝对误差 (MAE)，使用原始数据
# mae_high = mean_absolute_error(y_ochl_test_inverse, y_pred_high)
# # 计算最大误差 (Max Error)，使用原始数据
# max_err_high = max_error(y_ochl_test_inverse, y_pred_high)

# print(f'Mean Squared Error for highest price: {mse_high}')
# print(f'Mean Absolute Error for highest price: {mae_high}')
# print(f'Max Error for highest price: {max_err_high}')

# y_pred_high = scaler_highest.inverse_transform(y_pred_high_scaled)

# 将预测结果放回原始数据帧
# test_start_index = len(X_train_splits[0]) + len(X_val_splits[0]) + look_back_values[0] - 1
# df.loc[test_start_index:test_start_index+len(y_pred)-1, 'Predicted_Target'] = y_pred_class
# df.loc[test_start_index:test_start_index+len(y_pred)-1, 'Confidence'] = y_pred_confidence

# # 导出为 CSV 文件
# df.to_csv('df_with_predictions.csv', index=False)
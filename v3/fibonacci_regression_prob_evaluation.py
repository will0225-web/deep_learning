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

from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, PReLU, Conv1D, MaxPooling1D, Flatten, LeakyReLU, ReLU, Bidirectional, Attention, LayerNormalization, Input, Activation, RepeatVector, Permute, Multiply, Layer, concatenate, Concatenate
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
from sklearn.linear_model import LogisticRegression

import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import csv


def tolerance_mse(y_true, y_pred, scaler, tolerance=5):
    # 获取缩放参数
    scale = tf.cast(scaler.scale_[0], tf.float32)
    mean = tf.cast(scaler.mean_[0], tf.float32)
    
    # 将 y_true 和 y_pred 还原到原始尺度
    y_true_orig = y_true * scale + mean
    y_pred_orig = y_pred * scale + mean
    
    # 计算容忍范围
    tolerance_scaled = tf.cast(tolerance, tf.float32)
    
    # 计算预测值与真实值之间的差异
    diff = tf.abs(y_true_orig - y_pred_orig)
    
    # 如果差异在容忍范围内，损失为零，否则计算MSE
    loss = tf.where(diff <= tolerance_scaled, tf.zeros_like(diff), tf.square(diff))
    
    # 将损失重新缩放回原始尺度
    loss_scaled = loss / tf.square(scale)
    
    return tf.reduce_mean(loss_scaled)

def tolerance_mae(y_true, y_pred, scaler, tolerance=5):
    # 获取缩放参数
    scale = tf.cast(scaler.scale_[0], tf.float32)
    mean = tf.cast(scaler.mean_[0], tf.float32)
    
    # 将 y_true 和 y_pred 还原到原始尺度
    y_true_orig = y_true * scale + mean
    y_pred_orig = y_pred * scale + mean
    
    # 计算容忍范围
    tolerance_scaled = tf.cast(tolerance, tf.float32)
    
    # 计算预测值与真实值之间的差异
    diff = tf.abs(y_true_orig - y_pred_orig)
    
    # 如果差异在容忍范围内，损失为零，否则计算MAE
    loss = tf.where(diff <= tolerance_scaled, tf.zeros_like(diff), diff)
    
    # 将损失重新缩放回原始尺度
    loss_scaled = loss / scale
    
    return tf.reduce_mean(loss_scaled)

def calculate_targets(df, period):
    df['Future_Low'] = df['Low'].rolling(window=period, min_periods=1).min().shift(-period)
    df['Future_Low_Pos'] = df['Low'].rolling(window=period, min_periods=1).apply(lambda x: np.argmin(x) + 1).shift(-period)
    return df

# def calculate_targets(df, period, multiplier=1.5):
#     # 计算未来period的收盘价变化率
#     df['SL_Touch'] = 0  # 初始化趋势标签为0
#     df['ATR_Multiplier'] = df['ATR'] * multiplier
#     # 重置索引以确保索引是连续的整数
#     df = df.reset_index(drop=True)
#     # # 根据价格变化率和阈值判断趋势
#     # df['Trend_Target'] = np.where(df['Price_Change'] > threshold, 1,
#     #                               np.where(df['Price_Change'] < -threshold, 2, 0))

#     for i in range(len(df) - period):
#         close_price = df.at[i, 'Close']
#         future_atr_multiplier = df.at[i, 'ATR_Multiplier']

#         # 计算未来period内每一天的价格变化率
#         for j in range(1, period + 1):
#             future_high = df.at[i + j, 'High']
#             future_low = df.at[i + j, 'Low']

#             if future_low <= close_price - future_atr_multiplier:
#                 df.at[i, 'SL_Touch'] = 1
#                 break
#             elif future_high >= close_price + future_atr_multiplier:
#                 df.at[i, 'SL_Touch'] = 2
#                 break
#     return df

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
    
    # df, price_indicator_info = indicators.add_price_indicator(df, look_back=price_look_back, is_need_return_function_info=True)
    # customized_cols_infos.append(price_indicator_info)

    # df, supertrend_delta_info = indicators.super_trend_delta_and_risk(df, is_need_return_function_info=True)
    # customized_cols_infos.append(supertrend_delta_info)
    # df, supertrend_delta_info = indicators.super_trend_delta_and_risk_v1(df, is_need_return_function_info=True)
    # customized_cols_infos.append(supertrend_delta_info)
    # df, supertrend_risk_score_info = indicators.calculate_supertrend_risk(df, is_need_return_function_info=True)
    # customized_cols_infos.append(supertrend_risk_score_info)

    # df, look_back_24h_min_max_price_info = indicators.add_24h_min_max_price(df, look_back=96, is_need_return_function_info=True)
    # customized_cols_infos.append(look_back_24h_min_max_price_info)

    # df = calculate_volatility(df, 8)
    # di_len = 14
    # adx_len = 14
    # df = calculate_adx(df, di_len, adx_len)
    # df = calculate_moving_averages(df)
    # df = calculate_bollinger_bands(df)
    # df = calculate_momentum(df)
    # df = calculate_ppo(df)
    # df = calculate_trend(df, 288, 0.04)
    df = indicators.trend_min_max_price_indicator(df, 192)
    df = indicators.calculate_candelstick_patterns(df)

    df = calculate_volatility(df, 20)
    di_len = 14
    adx_len = 14
    df = calculate_adx(df, di_len, adx_len)
    df = calculate_moving_averages(df)
    df = calculate_bollinger_bands(df)
    df = calculate_momentum(df)
    df = calculate_ppo(df)
    # df = calculate_trend(df, 288, 0.04)
    df = indicators.trend_min_max_price_indicator(df, 192)
    df = indicators.calculate_candelstick_patterns(df)
    df = calculate_VWAP(df)

    # 计算价格变化和对数收益率
    # df['Open_Change_Rate'] = df['Open'].pct_change().fillna(0)
    # df['High_Change_Rate'] = df['High'].pct_change().fillna(0)
    # df['Low_Change_Rate'] = df['Low'].pct_change().fillna(0)
    # df['Close_Change_Rate'] = df['Close'].pct_change().fillna(0)
    # # 计算成交量变化率
    # df['Volume_Change'] = df['Volume'].pct_change().fillna(0)

    # df['Close_Open_Ratio'] = df['Close'] / df['Open']
    # df['High_Low_Ratio'] = df['High'] / df['Low']
    # df['Close_High_Ratio'] = df['Close'] / df['High']
    # df['Close_Low_Ratio'] = df['Close'] / df['Low']

    # # 计算布林带宽度
    # df['BB_Width'] = df['Upper Band'] - df['Lower Band']
    # # 计算布林带宽度变化率
    # df['BB_Width_Ratio'] = df['BB_Width'].pct_change().fillna(0)
    
    # # 對數收益率
    # df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    # df['Volatility'] = df['Log_Returns'].rolling(window=20).std()
    
    # df['SMA_5'] = indicators.calculate_sma(df['Close'], 5)
    # df['SMA_10'] = indicators.calculate_sma(df['Close'], 10)

    df['EMA_7'] = indicators.calculate_ema(df['Close'], 7)
    df['EMA_25'] = indicators.calculate_ema(df['Close'], 25)
    df['EMA_99'] = indicators.calculate_ema(df['Close'], 99)

    # df = calculate_VWAP(df)

    # 紀錄新的cols
    modified_cols = set(df.columns)
    # 篩選多出來的cols
    new_cols = list(modified_cols - original_cols)
    return df, customized_cols_infos, new_cols

def calculate_VWAP(df):
    df['Typical_Price'] = (df['Close'] + df['High'] + df['Low']) / 3
    # df['Typical_Price'] = df['Close']
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

def calculate_ahead_high_low(df, percentage=0.001):
    df['Future_High'] = df['Close'] * (1 + percentage)
    df['Future_Low'] = df['Close'] * (1 - percentage)
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


# def evaluate_tolerance(model, X_test, y_pos_test, scaler_pos, tolerance=2):
#     # 预测
#     preds = model.predict(X_test)
    
#     # 逆缩放预测值
#     preds_low = preds[0]
#     preds_pos = preds[1]
#     preds_pos_inverse = scaler_pos.inverse_transform(preds_pos)
    
#     # 将 y_pos_test 逆缩放回原始范围
#     y_pos_test_inverse = scaler_pos.inverse_transform(y_pos_test)

#     # 计算容忍范围内的准确性
#     correct = 0
#     total_loss = 0
#     for pred, true in zip(preds_pos_inverse, y_pos_test_inverse):
#         if abs(pred - true) <= tolerance:
#             correct += 1
#         # 计算损失
#         total_loss += (0 if abs(pred - true) <= tolerance else (pred - true) ** 2)

#     accuracy = correct / len(y_pos_test)
#     avg_loss = total_loss / len(y_pos_test)

#     print(f"Accuracy within tolerance range: {accuracy:.2%}")
#     print(f"Average Loss with tolerance: {avg_loss}")

#     return accuracy, avg_loss

# 准备特征和目标变量
# def prepare_features_and_targets(df, max_lookahead):
#     features = []
#     targets_0_382 = []
#     targets_0_5 = []
#     targets_0_618 = []

#     for i in range(len(df) - max_lookahead):
#         for lookahead in range(1, max_lookahead + 1):
#             high_window = df['High'].iloc[i:i+lookahead].max()
#             features.append([df['High'].iloc[i], lookahead])
#             targets_0_382.append(int(high_window > df['fibonacci_0.382'].iloc[i]))
#             targets_0_5.append(int(high_window > df['fibonacci_0.5'].iloc[i]))
#             targets_0_618.append(int(high_window > df['fibonacci_0.618'].iloc[i]))

#     return np.array(features), np.array(targets_0_382), np.array(targets_0_5), np.array(targets_0_618)


symbol = "ETHUSDT"
interval = "15m"
look_back = 1 #使用回看n根數據
epochs = 150
batch_size = 128
total_klines = 150000
get_local_file_name = 'ETHUSDT_15m_2023-12-31_23-59-59_150000_calculated.csv'

input_model_infos = []

# 當前
# end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# end_time_seconds = end_time / 1000
# end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
# end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# 特定
# end_time_string = "2023-12-31 23:59:59"


# # 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# # Step 1: 獲取數據
# df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)

# # 将 'datetime' 列转换为 datetime 类型
# df['datetime'] = pd.to_datetime(df['datetime'])

# # 将 'datetime' 列转换为 UTC+8
# df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')

# # 如果需要移除时区信息，可以使用 .dt.tz_localize(None)
# df['datetime'] = df['datetime'].dt.tz_localize(None)
# df, customized_cols_infos, new_cols = customized_specific_period_col(df)

# # 添加新的列
# # df['High_Condition'] = np.where(df['High'] > df['Current_Max_High'], 0, 1)
# # df['Low_Condition'] = np.where(df['Low'] < df['Current_Min_Low'], 0, 1)

# drop_front_data_count = 1000
# df = df[drop_front_data_count:]
# df.reset_index(drop=True, inplace=True)

# df.fillna(0, inplace=True)
# df.replace([np.inf, -np.inf], np.nan, inplace=True)
# df.interpolate(method='linear', inplace=True)

# numeric_cols = df.select_dtypes(include=[np.number]).columns

# # 检查 NaN 值的数量
# nan_counts = df[numeric_cols].isna().sum()
# print("NaN Counts:\n", nan_counts)

# inf_counts = np.isinf(df[numeric_cols]).sum()
# print("Infinity Counts:\n", inf_counts)

# total_nan_inf = nan_counts + inf_counts
# print("Total NaN and Infinity Counts:\n", total_nan_inf)

# # 检查所有列中 NaN 值的行
# nan_volume_rows = df[df[numeric_cols].isna().any(axis=1)]
# inf_volume_rows = df[np.isinf(df[numeric_cols]).any(axis=1)]
# print("Rows with NaN column:\n", nan_volume_rows)
# print("Rows with inf column:\n", inf_volume_rows)
# df.reset_index(drop=True, inplace=True)

# cols = [
#         'Open', 'High', 'Low', 'Close', 'Future_High', 'Volume', 'fibonacci_0.382', 
#         'fibonacci_0.5', 'fibonacci_0.618', 'Open_Close_pct', 'High_Low_pct', 
#         'Up_Shadow_pct', 'Down_Shadow_pct', 'EMA_7', 'EMA_25', 'EMA_99', 'RSI', 'Middle Band', 'Upper Band', 'Lower Band', 'ATR', 'lookahead', 'increase'
#     ]

# # 最大前瞻窗口
max_lookahead = 24

# # 生成特征和目标
# X, _, _ = signals.prepare_features_and_targets(df, max_lookahead, cols, min_increase=0.003, max_increase=0.02, step=0.001)

X = pd.read_csv('output.csv')

cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'fibonacci_0.382', 
        'fibonacci_0.5', 'fibonacci_0.618', 'Open_Close_pct', 'High_Low_pct', 
        'Up_Shadow_pct', 'Down_Shadow_pct', 'EMA_7', 'EMA_25', 'EMA_99', 'BB_Width', 'Close_Volatility', 'RSI', 'ATR', 'VWAP', 'ADX', 'Lookahead', 'Increase'
    ]
X = X[cols]

robust_features = ['Volume', 'Open_Close_pct', 'High_Low_pct', 'Up_Shadow_pct', 'Down_Shadow_pct', 'ATR', 'BB_Width', 'Close_Volatility', 'VWAP', 'ADX']
standard_features = ['Low', 'Close', 'High', 'Open', 'fibonacci_0.618', 'fibonacci_0.5', 'fibonacci_0.382', 'EMA_7', 'EMA_25', 'EMA_99']
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
preprocessor.fit(X)

model_up = xgb.XGBClassifier()  # or xgb.XGBRegressor() depending on your model type
model_up.load_model('xgboost_up.model')

model_down = xgb.XGBClassifier()  # or xgb.XGBRegressor() depending on your model type
model_down.load_model('xgboost_down.model')

get_local_file_name = ''
# predict目前最新的資料
end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

end_time_seconds = end_time / 1000
end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 20000, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)

# 将 'datetime' 列转换为 datetime 类型
df['datetime'] = pd.to_datetime(df['datetime'])

# 将 'datetime' 列转换为 UTC+8
df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')

# 如果需要移除时区信息，可以使用 .dt.tz_localize(None)
df['datetime'] = df['datetime'].dt.tz_localize(None)
df, customized_cols_infos, new_cols = customized_specific_period_col(df)

drop_front_data_count = 1000
df = df[drop_front_data_count:]
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


# 预测并存储结果
# predict_lookahead = 24
# predict_increase = 0.012
min_increase = 0.004
max_increase = 0.025
step = 0.001
model_up.set_params(device='cuda')
# 初始化存储结果的列表
results = []
# 打开CSV文件，并写入表头
with open('results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['lookahead', 'increase', 'threshold', 'win_count', 'loss_count', 'win_rate'])
    for lookahead in range(8, max_lookahead + 1):
        for increase in np.arange(min_increase, max_increase + step, step):
            # 计算所有的数据
            
            
            # 创建输入数据
            input_data = df.iloc[:len(df) - lookahead].copy()
            input_data['Lookahead'] = lookahead
            input_data['Increase'] = increase
            
            # 选择相关的列
            # input_data = input_data[[
            #     'Open', 'High', 'Low', 'Close', 'Future_High', 'Future_Low', 'Volume',
            #     'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618',
            #     'Open_Close_pct', 'High_Low_pct', 'Up_Shadow_pct', 'Down_Shadow_pct',
            #     'EMA_7', 'EMA_25', 'EMA_99', 'RSI', 'Middle Band', 'Upper Band', 'Lower Band',
            #     'ATR', 'lookahead', 'increase'
            # ]]

            input_data = input_data[cols]
            input_data = calculate_ahead_high_low(input_data, increase)

            # 预处理数据
            preprocessed_data = preprocessor.transform(input_data)
            # 批量预测
            prob_up = model_up.predict_proba(preprocessed_data)[:, 1]
            prob_down = model_down.predict_proba(preprocessed_data)[:, 1]
            # 将预测结果更新回input_data
            input_data['prob_up'] = prob_up
            input_data['prob_down'] = prob_down
            # 使用input_data中的数据来计算high_windows
            high_windows_tolerance = lookahead
            input_data['high_windows'] = input_data['High'].shift(-1).rolling(window=high_windows_tolerance).max().shift(-(high_windows_tolerance - 1))
            input_data['low_windows'] = input_data['Low'].shift(-1).rolling(window=high_windows_tolerance).min().shift(-(high_windows_tolerance - 1))
            # 计算 y 的值
            # input_data['y'] = (input_data['high_windows'] >= input_data['Future_High']).astype(int)
            input_data['y'] = (input_data['low_windows'] <= input_data['Future_Low']).astype(int)

            for threshold in np.arange(0.5, 0.91, 0.1):
                # 根据threshold设置y_pred，prob_up大于threshold则为1，否则为0
                # over_threshold = input_data[input_data['prob_up'] >= threshold]
                over_threshold = input_data[input_data['prob_down'] >= threshold]

                # 统计y = 1的数量
                win_count = np.sum(over_threshold['y'] == 1)
                # 统计y = 0的数量
                loss_count = np.sum(over_threshold['y'] == 0)
                win_rate = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else 0
                # 将结果直接写入CSV文件
                writer.writerow([lookahead, increase, threshold, win_count, loss_count, win_rate])  
                # 将结果存储在列表中
                # results.append([lookahead, increase, threshold, win_count, loss_count, win_rate])
                print(f"Lookahead: {lookahead}, Increase: {increase}, Threshold: {threshold}, Win Count: {win_count}, Loss Count: {loss_count}, Win Rate: {win_rate:.2%}")

input_data.to_csv('input_data.csv', index=False)
print("Results saved to results.csv")

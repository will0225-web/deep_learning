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

# 假设df是你的数据框，包含所有需要的列
def calculate_rolling_stats(df, window=20):
    rolling_mean = df.rolling(window=window).mean()
    rolling_std = df.rolling(window=window).std()
    return rolling_mean, rolling_std

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

    # df, look_back_24h_min_max_price_info = indicators.add_24h_min_max_price(df, look_back=96, is_need_return_function_info=True)
    # customized_cols_infos.append(look_back_24h_min_max_price_info)

    df = calculate_volatility(df, 20)
    di_len = 14
    adx_len = 14
    df = calculate_adx(df, di_len, adx_len)
    # df = calculate_moving_averages(df)
    df = calculate_bollinger_bands(df)
    df = calculate_momentum(df)
    df = calculate_ppo(df)
    # df = calculate_trend(df, 288, 0.04)
    df = indicators.trend_min_max_price_indicator(df, 192)
    df = indicators.calculate_candelstick_patterns(df)


    # # 對數收益率
    # df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    # df['Volatility'] = df['Log_Returns'].rolling(window=20).std()
    
    # df['SMA_5'] = indicators.calculate_sma(df['Close'], 5)
    # df['SMA_10'] = indicators.calculate_sma(df['Close'], 10)

    df['EMA_7'] = indicators.calculate_ema(df['Close'], 7)
    df['EMA_25'] = indicators.calculate_ema(df['Close'], 25)
    df['EMA_99'] = indicators.calculate_ema(df['Close'], 99)

    df = calculate_VWAP(df)

    # 紀錄新的cols
    modified_cols = set(df.columns)
    # 篩選多出來的cols
    new_cols = list(modified_cols - original_cols)
    return df, customized_cols_infos, new_cols

def set_y_label(df, lookahead=8, percentage=0.05):
    # 初始化y列
    df['y'] = 0
    
    for i in range(len(df) - lookahead):
        current_close = df['Close'].iloc[i]
        future_window = df['Close'].iloc[i+1:i+lookahead+1]
        
        max_future_price = future_window.max()
        min_future_price = future_window.min()
        
        if max_future_price >= current_close * (1 + percentage) and min_future_price <= current_close * (1 - percentage):
            df.at[i, 'y'] = 3  # 未来同时出现上涨和下跌5%的可能性
        elif max_future_price >= current_close * (1 + percentage):
            df.at[i, 'y'] = 1  # 未来有上涨5%的可能性
        elif min_future_price <= current_close * (1 - percentage):
            df.at[i, 'y'] = 2  # 未来有下跌5%的可能性
        else:
            df.at[i, 'y'] = 0  # 未来没有明显的上涨或下跌
        
    return df


# symbol = "ETHUSDT"
# interval = "15m"
# look_back = 1 #使用回看n根數據
# epochs = 150
# batch_size = 128
# total_klines = 150000
# get_local_file_name = 'ETHUSDT_15m_2023-12-31_23-59-59_150000_calculated.csv'

# input_model_infos = []

# # end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# # end_time_seconds = end_time / 1000
# # end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
# # end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# end_time_string = "2023-12-31 23:59:59"
# df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
# df, customized_cols_infos, new_cols = customized_specific_period_col(df)

# # 将 'datetime' 列转换为 datetime 类型
# df['datetime'] = pd.to_datetime(df['datetime'])

# # 将 'datetime' 列转换为 UTC+8
# df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')

# # 如果需要移除时区信息，可以使用 .dt.tz_localize(None)
# df['datetime'] = df['datetime'].dt.tz_localize(None)

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

# df.drop(['datetime', 'TR_sum', '+DM_sum', '-DM_sum', 'year', 'month', 'weekday', 'day', '+DM', '-DM', '+DI', '-DI', 'VP', 'EMA_26'], axis=1, inplace=True)

# lookahead = 480
# percnetage = 0.05
# # cols = [
# #         'Open', 'High', 'Low', 'Close', 'Volume', 'fibonacci_0.382', 
# #         'fibonacci_0.5', 'fibonacci_0.618', 'Open_Close_pct', 'High_Low_pct', 
# #         'Up_Shadow_pct', 'Down_Shadow_pct', 'EMA_7', 'EMA_25', 'EMA_99', 'BB_Width', 'Close_Volatility', 'RSI', 'ATR', 'VWAP', 'ADX', 'Lookahead'
# #     ]
# # X = df[cols] + ['Lookahead', 'Increase', 'y_up', 'y_down']
# df = set_y_label(df, lookahead, percnetage)
# print(df['y'].value_counts())
# df["lookahead"] = lookahead
# df["percentage"] = percnetage

X = pd.read_csv('X.csv')

X = X.select_dtypes(include=[np.number])

# 計算相關係數矩陣
corr_matrix = X.corr()

# 對 y_up 相關係數進行排序，並排除 y_down 和 y_up 自身
top_50_corr_up = corr_matrix['y'].drop(['y']).abs().sort_values(ascending=False).head(50)


import matplotlib.pyplot as plt
import seaborn as sns

# 視覺化 y_up 的相關係數前 50 名
plt.figure(figsize=(10, 6))
sns.barplot(x=top_50_corr_up.values, y=top_50_corr_up.index)
plt.title('Top 50 Correlations with y_up')
plt.show()
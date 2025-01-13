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
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

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
    df = calculate_moving_averages(df)
    df = calculate_bollinger_bands(df)
    df = calculate_momentum(df)
    df = calculate_ppo(df)
    # df = calculate_trend(df, 288, 0.04)
    df = indicators.trend_min_max_price_indicator(df, 192)
    df = indicators.calculate_candelstick_patterns(df)
    df = calculate_VWAP(df, window=20)

    # # 對數收益率
    # df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    # df['Volatility'] = df['Log_Returns'].rolling(window=20).std()
    
    # df['SMA_5'] = indicators.calculate_sma(df['Close'], 5)
    # df['SMA_10'] = indicators.calculate_sma(df['Close'], 10)

    df['EMA_7'] = indicators.calculate_ema(df['Close'], 7)
    df['EMA_25'] = indicators.calculate_ema(df['Close'], 25)
    df['EMA_99'] = indicators.calculate_ema(df['Close'], 99)

    
    # 紀錄新的cols
    modified_cols = set(df.columns)
    # 篩選多出來的cols
    new_cols = list(modified_cols - original_cols)
    return df, customized_cols_infos, new_cols

def calculate_VWAP(df, window=20):
    df['Typical_Price'] = (df['Close'] + df['High'] + df['Low']) / 3
    # df['Typical_Price'] = df['Close']
    df['VP'] = df['Typical_Price'] * df['Volume']

    # df['Cumulative_VP'] = df['VP'].cumsum()
    # df['Cumulative_Volume'] = df['Volume'].cumsum()
    # 使用滚动窗口计算 VP 和 Volume 的累积和
    df['Cumulative_VP'] = df['VP'].rolling(window=window).sum()
    df['Cumulative_Volume'] = df['Volume'].rolling(window=window).sum()

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

def set_y_label(df, lookahead=8, percentage=0.05):
    # set_y_label
    
    for i in range(len(df) - lookahead):
        current_close = df['Close'].iloc[i]
        future_window = df['Close'].iloc[i+1:i+lookahead+1]

        future_high_window = df['High'].iloc[i+1:i+lookahead+1]
        future_low_window = df['Low'].iloc[i+1:i+lookahead+1]
        
        # max_future_price = future_window.max()
        # min_future_price = future_window.min()

        max_future_price = future_high_window.max()
        min_future_price = future_low_window.min()

        up_price = current_close * (1 + percentage)
        down_price = current_close * (1 - percentage)
        
        if max_future_price >= up_price and min_future_price <= down_price:
            df.at[i, 'y'] = 3  # 未来同时出现上涨和下跌5%的可能性
        elif max_future_price >= up_price:
            df.at[i, 'y'] = 1  # 未来有上涨5%的可能性
        elif min_future_price <= down_price:
            df.at[i, 'y'] = 2  # 未来有下跌5%的可能性
        else:
            df.at[i, 'y'] = 0  # 未来没有明显的上涨或下跌

        df.at[i, 'up_percentage'] = up_price
        df.at[i, 'down_percentage'] = down_price
    df['lookahead'] = lookahead
    # df['up_percentage'] = percentage
    # df['down_percentage'] = -percentage
    return df

def set_y_label_sequence(df, lookahead=8, percentage=0.05):
    for i in range(len(df) - lookahead):
        current_close = df['Close'].iloc[i]
        up_price = current_close * (1 + percentage)
        down_price = current_close * (1 - percentage)

        y_label = 0  # 默认为没有明显的上涨或下跌

        for j in range(1, lookahead + 1):
            future_high = df['High'].iloc[i + j]
            future_low = df['Low'].iloc[i + j]

            if future_high >= up_price and future_low <= down_price:
                # 如果同时达到up_price和down_price，判断谁先到达
                y_label = 3
                break
            elif future_high >= up_price:
                y_label = 1  # 先上涨
                break
            elif future_low <= down_price:
                y_label = 2  # 先下跌
                break

        df.at[i, 'y'] = y_label
        df.at[i, 'up_percentage'] = up_price
        df.at[i, 'down_percentage'] = down_price

    df['lookahead'] = lookahead
    return df

def set_y_win_lose_label_sequence(df, lookahead=8, tp_percentage=0.008, sl_percentage=0.004):
    for i in range(len(df) - lookahead):
        current_close = df['Close'].iloc[i]
        
        # 计算多头的止盈和止损价格
        long_tp_price = current_close * (1 + tp_percentage)
        long_sl_price = current_close * (1 - sl_percentage)
        

        y_label = -1  # 默认为没有达到止盈或止损的情况
        long_hit = False  # 标记多头是否触发止盈或止损

        for j in range(1, lookahead + 1):
            future_high = df['High'].iloc[i + j]
            future_low = df['Low'].iloc[i + j]

            # 检查多头是否触发止盈或止损
            if not long_hit:
                if future_low <= long_sl_price:
                    y_label = 0  # 多头止损（long lose）
                    long_hit = True  # 多头已经处理
                    break
                elif future_high >= long_tp_price:
                    y_label = 1  # 多头获胜（long win）
                    long_hit = True  # 多头已经处理
                    break

        # 如果在lookahead期间没有触发止盈或止损，使用未来的Close价格与当前Close价格进行比较
        if not long_hit:
            future_close = df['Close'].iloc[i + lookahead]
            required_profit = current_close * (1 + sl_percentage)
            if future_close >= required_profit:
                y_label = 1  # Future close higher than current close (long win)
            else:
                y_label = 0  # Future close lower or equal to current close (long lose)

        # 标记止盈和止损价格
        df.at[i, 'y'] = y_label
        df.at[i, 'long_tp_price'] = long_tp_price
        df.at[i, 'long_sl_price'] = long_sl_price

    df['lookahead'] = lookahead
    return df

def backtesting(input_data, lookahead, percentage, prob, fee=0.0012):
    total_profit_percentage = 0  # 初始化总利润
    trades = []  # 记录每笔交易的利润
    profit_over_time = []  # 记录每根 K 线的利润百分比变化
    # 遍历每一行数据，进行回测
    for i in range(len(input_data) - lookahead):
        entry_price = input_data['Close'].iloc[i]  # 进场价格
        predicted_class = input_data['predicted_class'].iloc[i]  # 预测类别
        # prob_class_1 = input_data['prob_class_1'].iloc[i]  # target 1（多头）的预测概率
        # prob_class_2 = input_data['prob_class_2'].iloc[i]  # target 2（空头）的预测概率
        final_close_price = input_data['Close'].iloc[i + lookahead]  # 使用lookahead之后的下一个收盘价

        # 进场多头交易
        if predicted_class == 1:
            future_high = input_data['High'].iloc[i+1:i+lookahead+1].max()  # 未来的最高价
            future_low = input_data['Low'].iloc[i+1:i+lookahead+1].min()  # 未来的最低价
            tp_price = input_data['long_tp_price'].iloc[i]  # 多头止盈目标
            sl_price = input_data['long_sl_price'].iloc[i]  # 多头止损目标
            # tp_price = entry_price * (1 + percentage)  # 多头止盈目标
            # sl_price = entry_price * (1 - percentage)  # 多头止损目标

            if future_low <= sl_price:
                # 计算止损时的亏损百分比
                profit_percentage = (sl_price - entry_price) / entry_price - fee
            elif future_high >= tp_price:
                # 计算止盈时的收益百分比
                profit_percentage = (tp_price - entry_price) / entry_price - fee
            else:
                # 如果没有达到止盈或止损，则使用 entry 与最后一个 close 的差额来计算利润
                profit_percentage = (final_close_price - entry_price) / entry_price - fee


            total_profit_percentage += profit_percentage
            trades.append(profit_percentage)
        

        # 进场空头交易
        # elif predicted_class == 2 and prob_class_2 >= prob:
        #     future_high = input_data['High'].iloc[i+1:i+lookahead+1].max()  # 未来的最高价
        #     future_low = input_data['Low'].iloc[i+1:i+lookahead+1].min()  # 未来的最低价
        #     tp_price = input_data['short_tp_price'].iloc[i]  # 多头止盈目标
        #     sl_price = input_data['short_sl_price'].iloc[i]  # 多头止损目标
        #     # tp_price = entry_price * (1 - percentage)  # 空头止盈目标
        #     # sl_price = entry_price * (1 + percentage)  # 空头止损目标

        #     if future_high >= sl_price:
        #         # 计算空头止损时的亏损百分比
        #         profit_percentage = (entry_price - sl_price) / entry_price - fee
        #     elif future_low <= tp_price:
        #         # 计算空头止盈时的收益百分比
        #         profit_percentage = (entry_price - tp_price) / entry_price - fee
        #     else:
        #         # 如果没有达到止盈或止损，则使用 entry 与最后一个 close 的差额来计算利润
        #         profit_percentage = (entry_price - final_close_price) / entry_price - fee

        #     total_profit_percentage += profit_percentage
        #     trades.append(profit_percentage)
        profit_over_time.append(total_profit_percentage)  # 记录累计利润变化
    # 返回总利润和每笔交易的记录
    return total_profit_percentage, trades, profit_over_time

# 绘制累计利润变化的折线图
def plot_total_profit_change(profit_over_time):
    plt.figure(figsize=(10, 6))
    plt.plot(profit_over_time, label="Cumulative Profit Percentage Over Time", color='blue')
    plt.title("Cumulative Profit Percentage Over K-Line")
    plt.xlabel("K-Line Index")
    plt.ylabel("Cumulative Profit Percentage")
    plt.legend()
    plt.grid(True)
    plt.show()

def validation(test_data, model, preprocessor, close_scaler, cols, lookahead, tp, sl, price_related_features, robust_features):
    # 预测并存储结果
    model.set_params(device='cuda')

    input_data = test_data.iloc[:len(test_data) - lookahead].copy()
    input_data = input_data[cols]
    # input_data = set_y_label(input_data, lookahead, percentage)
    # input_data = set_y_label_sequence(input_data, lookahead, percentage)
    input_data = set_y_win_lose_label_sequence(input_data, lookahead, tp, sl)
    y = input_data['y']
    input_data.drop('y', axis=1, inplace=True)
    print(y.value_counts())

    # 先筛选出 Volume 大于 10,000 的数据
    # input_data = input_data[input_data['Volume'] >= 15000]
    
    # preprocessed_data, _, _ = scaler(input_data, price_related_features, robust_features, close_scaler=close_scaler, preprocessor=preprocessor)
    # preprocessed_data = input_data
    preprocessed_data = preprocessor.transform(input_data)

    # probs = model.predict_proba(preprocessed_data)
    probs = model.predict_proba(preprocessed_data)[:, 0]  # 只取属于正类的概率

    prob_threshold = 0.9
    predicted_classes = (probs >= prob_threshold).astype(int)

    # 将预测的类别和标签加入数据中
    input_data['predicted_class'] = predicted_classes
    input_data['y'] = y

    class_accuracies = []
    class_denominators = []
    class_numerators = []
    # 计算每个类别的准确率
    for class_label in [0, 1]:
        # 获取当前类别的所有样本
        class_data = input_data[input_data['predicted_class'] == class_label]

        # 分母: prob >= prob_threshold 的样本数量
        denominator = len(class_data)

        # 分子: predicted_class == y 且 prob >= prob_threshold 的样本数量
        numerator = (class_data['predicted_class'] == class_data['y']).sum()

        # 计算准确率，考虑分母为0的情况
        if denominator > 0:
            class_accuracy = numerator / denominator
        else:
            class_accuracy = 0

        class_accuracies.append(class_accuracy)
        class_denominators.append(denominator)
        class_numerators.append(numerator)

    total_accuracy = np.mean(class_accuracies)
    total_profit_percentage, trades, profit_over_time = backtesting(input_data, lookahead, tp, prob_threshold)
    # 绘制总利润变化的折线图
    plot_total_profit_change(profit_over_time)
    return total_accuracy, class_accuracies, class_denominators, class_numerators, total_profit_percentage
    

def get_test_data():
    # predict目前最新的資料
    end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

    end_time_seconds = end_time / 1000
    end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
    end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
    df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 20000, '', is_need_save_original_data=False, is_need_calculated=True)

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

    return df

def scaler(df, price_related_features, robust_features, close_scaler=None, preprocessor=None):
    robust_features = robust_features
    standard_features = []
    minMax_features = []

    # 假设 X 是你的数据集
    # 定义需要以 Close 尺度缩放的特征
    close_feature = ['Close']
    price_related_features = price_related_features

    if close_scaler is None:
        # 初始化 StandardScaler 并仅在 Close 上进行拟合
        close_scaler = StandardScaler()
        X_close_scaled = close_scaler.fit_transform(df[close_feature])
    else:
        X_close_scaled = close_scaler.transform(df[close_feature])

    X_scaled_without_price = df.drop(price_related_features + close_feature, axis=1)
    if preprocessor is None:
        # 列出每个缩放器/转换器对应的特征
        preprocessor = ColumnTransformer(
            transformers=[
                ('robust', RobustScaler(), robust_features),
                ('standard', StandardScaler(), standard_features),
                ('minMax', MinMaxScaler(feature_range=(0, 1)), minMax_features),
            ],
            remainder='passthrough'  # 不需要缩放的特征保持原样
        )
        X_scaled = preprocessor.fit_transform(X_scaled_without_price)
    else:
        X_scaled = preprocessor.transform(X_scaled_without_price)

    # 使用 Close 的均值和标准差手动缩放其他相关特征
    price_related_scaled = (df[price_related_features] - close_scaler.mean_[0]) / close_scaler.scale_[0]

    # 将缩放后的 price_related_features 替换回 X_scaled
    X_scaled_df = pd.DataFrame(X_scaled, columns=preprocessor.get_feature_names_out())

    # 将 price_related_scaled 替换回 X_scaled_df 中
    for i, feature in enumerate(price_related_features):
        X_scaled_df[feature] = price_related_scaled.iloc[:, i]

    X_scaled_df['Close'] = X_close_scaled
    # 导出特定列到 CSV 文件
    X_scaled = X_scaled_df.to_numpy()

    return X_scaled, close_scaler, preprocessor


symbol = "ETHUSDT"
interval = "15m"
look_back = 1 #使用回看n根數據
epochs = 150
batch_size = 128
total_klines = 150000
get_local_file_name = 'ETHUSDT_15m_2023-12-31_23-59-59_150000_calculated.csv'

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

# df.to_csv('original_df_15m.csv', index=False)

df = pd.read_csv('original_df_15m.csv')

lookahead = 96
tp = 0.02
sl = 0.02
cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'fibonacci_1', 'fibonacci_0',
        'Open_Close_pct', 'High_Low_pct', 'Up_Shadow_pct', 'Down_Shadow_pct',
        'EMA_7', 'EMA_25', 'EMA_99', 
        'BB_Width', 'Upper Band', 'Lower Band', 'Middle Band',
        'Close_Volatility', 'High_Volatility', 'Low_Volatility', 'Open_Volatility', 'ATR', 'VWAP', 'Cumulative_VP', 'Cumulative_Volume', 
    ]
X = df[cols]

# X = set_y_label(X, lookahead, percentage)
# X = set_y_label_sequence(X, lookahead, percentage)
X = set_y_win_lose_label_sequence(X, lookahead, tp, sl)
drop_front_data_count = 1000
X = X[drop_front_data_count:-300]
X.reset_index(drop=True, inplace=True)

# X.to_csv('X.csv', index=False)

y = X['y']
X.drop('y', axis=1, inplace=True)

print(y.value_counts())

robust_features = ['Volume', 'Open_Close_pct', 'High_Low_pct', 'Up_Shadow_pct', 'Down_Shadow_pct', 'ATR', 'BB_Width', 'Close_Volatility', 'High_Volatility', 'Low_Volatility', 'Open_Volatility', 'Cumulative_VP', 'Cumulative_Volume']
standard_features = ['Close', 'High', 'Low', 'Open', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'fibonacci_1', 'fibonacci_0', 'EMA_7', 'EMA_25', 'EMA_99', 'Upper Band', 'Lower Band', 'Middle Band', 'VWAP', 'long_tp_price', 'long_sl_price']
minMax_features = []

# 假设 X 是你的数据集
# 定义需要以 Close 尺度缩放的特征
### 這部分得到的成果沒有比較好
# close_feature = ['Close']
# price_related_features = ['High', 'Low', 'Open', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'fibonacci_1', 'fibonacci_0', 'EMA_7', 'EMA_25', 'EMA_99', 'Upper Band', 'Lower Band', 'Middle Band', 'VWAP', 'up_percentage', 'down_percentage']
# X_scaled, close_scaler, preprocessor = scaler(X, price_related_features, robust_features)

preprocessor = ColumnTransformer(
    transformers=[
        ('robust', RobustScaler(), robust_features),
        ('standard', StandardScaler(), standard_features),
        ('minMax', MinMaxScaler(feature_range=(0, 1)), minMax_features),
    ],
    remainder='passthrough'  # 不需要缩放的特征保持原样
)
X_scaled = preprocessor.fit_transform(X)

test_data = get_test_data()
n = 100
d = 7
lr = 0.25

model_big_trend = xgb.XGBClassifier(n_estimators=n, max_depth=d, learning_rate=lr, objective='binary:logistic', random_state=42, device='cuda', verbosity=1)

model_big_trend.fit(X_scaled, y)

# total_accuracy, class_accuracies, class_denominators, class_numerators, total_profit_percentage = validation(test_data, model_big_trend, preprocessor, close_scaler, cols, lookahead, percentage, price_related_features, robust_features)
total_accuracy, class_accuracies, class_denominators, class_numerators, total_profit_percentage = validation(test_data, model_big_trend, preprocessor, None, cols, lookahead, tp, sl, None, robust_features)

print(f"n_estimators: {n}, max_depth: {d}, max_leaves: {0}, learning_rate: {lr}, total_accuracy: {total_accuracy}, class_0_accuracy: {class_accuracies[0]}, class_1_accuracy: {class_accuracies[1]}, class_0_denominator: {class_denominators[0]}, class_1_denominator: {class_denominators[1]}, class_0_numerator: {class_numerators[0]}, class_1_numerator: {class_numerators[1]}, total_profit_percentage: {total_profit_percentage}")
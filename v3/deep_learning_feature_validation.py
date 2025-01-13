import sys
import os
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modules import data as modules_data
from modules import indicators as indicators
from modules import utilities as utilities

@utilities.capture_args
def calculate_volatility(df, look_back):
    df['Close_Volatility'] = df['Close'].rolling(window=look_back).std()
    df['High_Volatility'] = df['High'].rolling(window=look_back).std()
    df['Low_Volatility'] = df['Low'].rolling(window=look_back).std()
    df['Open_Volatility'] = df['Open'].rolling(window=look_back).std()

    return df

@utilities.capture_args
def calculate_bollinger_bands(df):
    df['BB_Width'] = df['Upper Band'] - df['Lower Band']
    df['BB_Pos'] = (df['Close'] - df['Lower Band']) / (df['Upper Band'] - df['Lower Band'])

    return df

@utilities.capture_args
def calculate_VWAP(df, window=20):
    df['Typical_Price'] = (df['Close'] + df['High'] + df['Low']) / 3
    df['VP'] = df['Typical_Price'] * df['Volume']

    df['Cumulative_VP'] = df['VP'].rolling(window=window).sum()
    df['Cumulative_Volume'] = df['Volume'].rolling(window=window).sum()

    df['VWAP'] = df['Cumulative_VP'] / df['Cumulative_Volume']

    return df

def customized_specific_period_col(df):
    customized_cols_infos = []
    
    # 紀錄原先的cols
    original_cols = set(df.columns)
    
    df, price_indicator_info = indicators.trend_min_max_price_indicator(df, look_back=192, is_need_return_function_info=True)
    customized_cols_infos.append(price_indicator_info)

    df, supertrend_delta_info = indicators.super_trend_delta_and_risk_v1(df, is_need_return_function_info=True)
    customized_cols_infos.append(supertrend_delta_info)

    df, bollinger_bands_info = calculate_bollinger_bands(df, is_need_return_function_info=True)
    customized_cols_infos.append(bollinger_bands_info)

    df['EMA_7'], ema_7_info = indicators.calculate_ema(df['Close'], 7, is_need_return_function_info=True)
    customized_cols_infos.append(ema_7_info)

    df['EMA_25'], ema_25_info = indicators.calculate_ema(df['Close'], 25, is_need_return_function_info=True)
    customized_cols_infos.append(ema_25_info)

    df['EMA_99'], ema_99_info = indicators.calculate_ema(df['Close'], 99, is_need_return_function_info=True)
    customized_cols_infos.append(ema_99_info)

    df, candlestick_patterns_info = indicators.calculate_candelstick_patterns(df, is_need_return_function_info=True)
    customized_cols_infos.append(candlestick_patterns_info)

    df, volatility_info = calculate_volatility(df, 20, is_need_return_function_info=True)
    customized_cols_infos.append(volatility_info)

    df, VWAP_info = calculate_VWAP(df, window=20, is_need_return_function_info=True)
    customized_cols_infos.append(VWAP_info)

    # 紀錄新的cols
    modified_cols = set(df.columns)
    # 篩選多出來的cols
    new_cols = list(modified_cols - original_cols)

    print(f'Full df: {df}')

    return df, customized_cols_infos, new_cols

@utilities.capture_args
def price_target(df, long_tp=0.015, short_tp=0.015, long_sl=0.0075, short_sl=0.0075, hold_count_limit=16):
    # 2 做多贏
    # 0 做空贏
    # 1 盤整無波動

    price_series = df['Close']
    low_series = df['Low']
    high_series = df['High']

    cases = []

    for idx in range(len(price_series) - 1):
        current_price = price_series[idx]

        hit_long_tp = False
        hit_short_tp = False

        case = 1

        check_end_index = min(len(price_series), idx + hold_count_limit + 1)
        for future_idx in range(idx + 1, check_end_index):
            future_price = price_series[future_idx]
            future_low = low_series[future_idx]
            future_high = high_series[future_idx]

            high_change_pct = indicators.calculate_percentage_change(future_high, current_price)
            low_change_pct = indicators.calculate_percentage_change(future_low, current_price)

            if high_change_pct >= long_tp:
                case = 2
                hit_long_tp = True
                break
            elif low_change_pct <= -short_tp:
                case = 0
                hit_short_tp = True
                break

        if not hit_long_tp and not hit_short_tp:
            case = 1

        cases.append(case)

    cases.append(1)  # 最後一個數據沒有未來的數據可以比較，所以直接設定為0
    return cases

@utilities.capture_args
def set_y_label(df, lookahead=8, long_percentage=None, short_percentage=None):
    # set_y_label

    for i in range(len(df)):
        current_close = df['Close'].iloc[i]

        if long_percentage is not None:
            if i == len(df) - 1:
                df.at[i, 'y'] = 0
                df.at[i, 'up_percentage'] = current_close * (1 + long_percentage)
                continue
            window_end_index = min(i + lookahead + 1, len(df))
            future_high_window = df['High'].iloc[i+1:window_end_index]
            max_future_price = future_high_window.max()
            up_price = current_close * (1 + long_percentage)
            if max_future_price >= up_price:
                df.at[i, 'y'] = 1
            else:
                df.at[i, 'y'] = 0
            df.at[i, 'up_percentage'] = up_price

        elif short_percentage is not None:
            if i == len(df) - 1:
                df.at[i, 'y'] = 0
                df.at[i, 'down_percentage'] = current_close * (1 - short_percentage)
                continue
            window_end_index = min(i + lookahead + 1, len(df))
            future_low_window = df['Low'].iloc[i+1:window_end_index]
            min_future_price = future_low_window.min()
            down_price = current_close * (1 - short_percentage)
            if min_future_price <= down_price:
                df.at[i, 'y'] = 1
            else:
                df.at[i, 'y'] = 0
            df.at[i, 'down_percentage'] = down_price

        df['lookahead'] = lookahead

    return df

end_time_string = "2023-12-31 23:59:59"

symbol = "ETHUSDT"
interval = "15m"
look_back = 144 #使用回看n根數據
epochs = 300
batch_size = 128
total_klines = 100000
get_local_file_name = 'ETHUSDT_15m_2023-12-31_23-59-59_150000_calculated.csv'
input_model_infos = []

df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
df, customized_cols_infos, new_cols = customized_specific_period_col(df)
df, target_function_info = set_y_label(df, lookahead=20, long_percentage=0.015, is_need_return_function_info=True)

target_counts = df['y'].value_counts()
print(target_counts)

X = df

cols = X.columns
cols_to_remove = ['datetime', 'Current_Max_date', 'Current_Min_date']
cols = cols[~cols.isin(cols_to_remove)]

X_features_only = X[cols]

X = X_features_only

corr = X.corr()

focus_features = ['y']

top_correlations = pd.DataFrame()
for feature in focus_features:
    top_features = corr[feature].sort_values(ascending=False).index[1:101]
    top_correlations[feature] = corr.loc[top_features, feature]
    print(top_correlations[feature].to_string())

plt.figure(figsize=(16, 20))
sns.heatmap(top_correlations, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation of Features with Target')
plt.show()

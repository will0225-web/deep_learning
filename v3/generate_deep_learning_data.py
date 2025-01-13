import sys
import os
from datetime import datetime

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules import data as modules_data
from modules import signals as signals
from modules import utilities as utilities
from modules import indicators as indicators

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

def customized_specific_period_col(df):
    customized_cols_infos = []

    # 紀錄原先的cols
    original_cols = set(df.columns)

    df = indicators.trend_min_max_price_indicator(df, look_back=192)

    df = indicators.super_trend_delta_and_risk_v1(df)

    df = calculate_bollinger_bands(df)

    df['EMA_7'] = indicators.calculate_ema(df['Close'], 7)

    df['EMA_25'] = indicators.calculate_ema(df['Close'], 25)

    df['EMA_99'] = indicators.calculate_ema(df['Close'], 99)

    df = indicators.calculate_candelstick_patterns(df)

    df = calculate_volatility(df, 20)

    df = calculate_VWAP(df, window=20)

    # Only klines with significant volume will be used as input data
    # Set threshold as 0 to include all klines
    df['Significant Kline'] = df['Volume'].apply(lambda x: 1 if x >= 0 else 0)
    siginificant_kline_counts = df['Significant Kline'].value_counts()
    print(siginificant_kline_counts)

    # 紀錄新的cols
    modified_cols = set(df.columns)
    # 篩選多出來的cols
    new_cols = list(modified_cols - original_cols)

    print(f'Full df: {df}')

    return df, customized_cols_infos, new_cols

symbol = "ETHUSDT"
interval = "15m"
total_klines = 300000
# end_time_string = '2023-12-31 23:59:59'
# get_local_file_name = 'ETHUSDT_5m_2023-12-31_23-59-59_450000_calculated.csv'
get_local_file_name = ''
file_name_tag = 'spot_full'

end_time = int(datetime.timestamp(datetime.now())) * 1000
end_time_seconds = end_time / 1000
end_datetime = datetime.fromtimestamp(end_time_seconds)
end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
temp_end_time_string = end_time_string.replace(':', '-').replace(' ', '_')

df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, 
                                              is_calculated_need_save=False, is_need_calculated=True, data_source='spot')
df, _, _ = customized_specific_period_col(df)
df = set_y_label(df, lookahead=288, long_percentage=0.04)
if file_name_tag:
    df.to_csv(f'local_data/{symbol}_{interval}_{temp_end_time_string}_{total_klines}_{file_name_tag}_calculated.csv')
else:
    df.to_csv(f'local_data/{symbol}_{interval}_{temp_end_time_string}_{total_klines}_calculated.csv')
target_counts = df['y'].value_counts()
print(target_counts)

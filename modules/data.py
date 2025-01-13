import os
import sys
import pandas as pd
import numpy as np
import datetime
import requests
from sklearn.preprocessing import MinMaxScaler
from modules import signals as signals
from modules import indicators as indicators
import pytz

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.join(current_path, '..')

local_data_path = os.path.join(parent_path, 'will_lee')

def get_binance_klines(symbol, interval, start_time_string, end_time_string, data_source='futures', get_local_file_name='', is_need_save_original_data=False, is_need_calculated=True, is_calculated_need_save=False, is_need_recalculate=False, file_name_tag=''):
    all_klines = []
    df = pd.DataFrame()  # 默认值为空的 DataFrame

    # 获取本地文件数据
    if get_local_file_name != '':
        get_local_file_path = os.path.join(local_data_path, get_local_file_name)
        if is_need_calculated:
            if os.path.exists(get_local_file_path):
                df = get_binance_klines_local(get_local_file_path, start_time=start_time_string, end_time=end_time_string)
                if is_need_recalculate:
                    df = recalculate_df_all_data(df)
                return df
            else:
                raise FileNotFoundError(f"File not found: {get_local_file_path}, Please regenerate the file.")
        else:
            if os.path.exists(get_local_file_path):
                all_klines = pd.read_csv(get_local_file_path)
                df = convert_binance_klines_to_df(all_klines)
                return df
            else:
                raise FileNotFoundError(f"File not found: {get_local_file_path}, Please regenerate the file.")
    else:
        # 选择 API 源
        url = 'https://api.binance.com/api/v3/klines' if data_source == 'spot' else 'https://fapi.binance.com/fapi/v1/klines'

        # 将 start_time 和 end_time 转换为毫秒时间戳
        start_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(start_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
        end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000

        # 请求数据，直到获取到所有在时间范围内的数据
        while start_time < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time,
                "limit": 500  # 每次最多获取500条
            }
            response = requests.get(url, params=params)
            data = response.json()

            if not data:
                # 如果返回的列表是空的，中断循环
                break

            all_klines.extend(data)  # 添加新数据到 all_klines
            start_time = data[-1][0] + 1  # 更新 start_time 为上次拿到的最后一个 K 线的结束时间+1毫秒

        # 文件保存路径
        temp_end_time_string = end_time_string.replace(':', '-').replace(' ', '_')
        calculate_file_path = os.path.join(local_data_path, f'{symbol}_{interval}_{temp_end_time_string}_calculated.csv' if file_name_tag == '' else f'{symbol}_{interval}_{temp_end_time_string}_{file_name_tag}_calculated.csv')

        # 保存数据
        if is_need_save_original_data:
            save_binance_data(all_klines, calculate_file_path)
        
        df = convert_binance_klines_to_df(all_klines)
        if is_need_calculated:
            df = calculate_df_all_data(df, calculate_file_path, is_calculated_need_save)

    return df


# Fetch Data from Binance
def get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name='', is_need_save_original_data=False, is_need_calculated=True, is_calculated_need_save=False, is_need_recalculate=False, file_name_tag='', data_source='futures'):
    all_klines = []
    df = pd.DataFrame()  # 默认值为空的DataFrame
     
    if get_local_file_name != '':
        get_local_file_path = os.path.join(local_data_path, get_local_file_name)
        if is_need_calculated:
            if os.path.exists(get_local_file_path):
                df = get_binance_klines_local(get_local_file_path, total_klines, end_time_string)
                if is_need_recalculate:
                    df = recalculate_df_all_data(df)
                return df
            else:
                raise FileNotFoundError(f"File not found: {get_local_file_path}, Please regenerate the file.")
        else:
            if os.path.exists(get_local_file_path):
                all_klines = pd.read_csv(get_local_file_path)
                df = convert_binance_klines_to_df(all_klines)
                return df
            else:
                raise FileNotFoundError(f"File not found: {get_local_file_path}, Please regenerate the file.")
    else:
        if data_source == 'spot':
            url = 'https://api.binance.com/api/v3/klines'
        else:
            url = 'https://fapi.binance.com/fapi/v1/klines'
        # Binance限制每次請求只能取得500筆K線
        limit = min(500, total_klines)
        num_requests = total_klines // limit

        end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
        for _ in range(num_requests):
            params = {
                "symbol": symbol,
                "interval": interval,
                "endTime": end_time,
                "limit": limit
            }
            response = requests.get(url, params=params)
            data = response.json()

            if not data:
                # 如果返回的列表是空的，中斷循環
                break
            
            all_klines = data + all_klines  # 添加新數據到all_klines的前面
            
            # 更新end_time為此次拿到的第一根K線的時間-1毫秒
            end_time = data[0][0] - 1

        temp_end_time_string = end_time_string.replace(':', '-').replace(' ', '_')
        
        if file_name_tag == '':
            calculate_file_path = os.path.join(local_data_path, f'{symbol}_{interval}_{temp_end_time_string}_{total_klines}_calculated.csv')
        else:
            calculate_file_path = os.path.join(local_data_path, f'{symbol}_{interval}_{temp_end_time_string}_{total_klines}_{file_name_tag}_calculated.csv')

        if is_need_save_original_data:
            save_binance_data(all_klines, calculate_file_path)
        df = convert_binance_klines_to_df(all_klines)
        if is_need_calculated:
            df = calculate_df_all_data(df, calculate_file_path, is_calculated_need_save)
    return df

def save_binance_data(klines, original_file_path):
    df = convert_binance_klines_to_df(klines)
    df.to_csv(original_file_path)

def recalculate_df_all_data(df):
    df = default_columes(df)
    return df

def default_columes(df):
    ### 這邊為default欄位, 基本上不要去改動, 要想創造客製化的cols, 另外設立function
    ### 例如要RSI 24長度的話, 欄位新增一個RSI_24

    # 提取時間
    # trading/interval-predict.py 已預先計算 datetime 欄位，但仍須將 dtype 轉換為 datetime
    if 'Open time' in df:
        df['datetime'] = pd.to_datetime(df['Open time'], unit='ms')
    elif 'datetime' in df:
        df['datetime'] = pd.to_datetime(df['datetime'])
    # 設置原始時間戳為UTC時區
    # df['datetime'] = df['datetime'].dt.tz_localize('UTC')

    # 轉換為台北時區 (UTC+8)
    # df['datetime'] = df['datetime'].dt.tz_convert('Asia/Taipei')

    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday

    d_time_temp = df['datetime']

    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'year', 'month', 'day', 'hour', 'weekday']].astype(float)
    df['kline_color'] = signals.is_green_red_kline(df['Close'], df['Open'])
    df['percentage'] = indicators.calculate_percentage_change(df['Close'], df['Open'])

    # 計算macd
    df['MACD'], df['Signal'], df['Hist'] = indicators.calculate_MACD(df['Close'], 12, 26)
    df['MACD_Long_Short'] = signals.is_macd_short_long(df['Hist'])
    df['MACD_Cross'] = signals.check_macd_hist_cross(df['Hist'])
    df['MACD_Stronger'] = signals.is_macd_stronger(df['Hist'])

    df['K'], df['D'] = indicators.calculate_stochastic_oscillator(df, 14)

    df['SMA_200'] = indicators.calculate_sma(df['Close'], 200)
    df['SMA_55'] = indicators.calculate_sma(df['Close'], 55)
    df['SMA_27'] = indicators.calculate_sma(df['Close'], 27)
    df['SMA_21'] = indicators.calculate_sma(df['Close'], 21)
    df['SMA_12'] = indicators.calculate_sma(df['Close'], 12)

    # EMA
    df['EMA_12'] = indicators.calculate_ema(df['Close'], 12)
    df['EMA_26'] = indicators.calculate_ema(df['Close'], 26)
    df['EMA_50'] = indicators.calculate_ema(df['Close'], 50)

    # TR, ATR
    df['TR'] = indicators.calculate_tr(df, 14)
    df['ATR'] = indicators.calculate_atr(df['TR'], 14)

    # 計算RSI
    df['RSI'] = indicators.calculate_RSI(df['Close'], 14)

    # 計算super trend
    df['Up Trend'], df['Down Trend'], df['Super Trend'] = indicators.calculate_supertrend(df, 10, 3)

    # 計算BB
    df['Middle Band'], df['Upper Band'], df['Lower Band'] = indicators.compute_bollinger_bands(df['Close'], 20 , 2)

    df = indicators.add_three_klines(df, 0.0045)
    # df = indicators.calculate_macd_hist_continuity(df)
    df = indicators.add_rsi_overbought_oversold(df, 80, 25)

    # 轉換float
    df = df.astype(float)

    df['datetime'] = d_time_temp

    return df

def set_target(df):
    # target = signals.set_rsi_target(df['Close'], df['RSI'], df['Low'], df['High'], 0.01, 0.01, 0.005, 60)
    # target = signals.set_target(df, 0.015, 0.005, 15)

    # target = signals.set_rsi_macd_target(df, 0.007, 0.007, 0.004, 0.006, 12, 48) # best

    # target = signals.set_rsi_macd_target(df, 0.005, 0.005, 0.005, 0.006, 15, 35)

    # target = signals.set_future_long_short_target(df, 0.015, 8)
    
    # target, target_function_info = signals.set_bb_specific_profit_strategy(df, 0.008, 0.008, 0.012, 0.005, 16, is_need_return_function_info=True)
    # target, target_function_info = signals.set_bb_trailing_strategy(df, 0.006, 0.006, 0.012, 0, 12, is_need_return_function_info=True)
    # target, target_function_info = signals.super_trend_strategy_v2(df, 0.008, 0.008, 0.004, 20, is_need_return_function_info=True)
    # target, target_function_info = signals.super_trend_strategy_v1(df, 0.008, 0.008, 0, 20, is_need_return_function_info=True) # best

    
    # if not is_macd_cross:
    #     target = signals.set_win_loss_target(df, 0.017, 0.012, 0.007, 16)
    # elif is_macd_cross:
    #     target = signals.set_target_macd_cross(df, 67, 25, 1.5)

    # target = signals.set_macd_thesame_trend_target(df, 0.015, 0.015, 0.005)
    # target = signals.set_macd_entry_trend_target(df, 0.015, 0.015, 0.005)
    
    # target = signals.set_rsi_targetA(df, 0.015, 0.015, 0.005, 60)
    
    # (max_highs, max_lows), target_function_info = signals.max_high_and_max_low_target(df, 36, is_need_return_function_info=True)
    # target, target_function_info = signals.super_trend_strategy_v3(df, 0.008, 0.008, 0.003, 12, is_need_return_function_info=True)。
    target, target_function_info = signals.super_trend_strategy_v4(df, 0.008, 0.008, 0.005, 12, is_need_return_function_info=True)
    df['Target'] = target
    # df['Max_High'] = max_highs
    # df['Max_Low'] = max_lows
    return df, target_function_info


def calculate_df_all_data(df, calculate_file_path, is_need_save=False):
    df = default_columes(df)

    if is_need_save:
        df.to_csv(calculate_file_path)
    return df

def convert_binance_klines_to_df(data):
    df = pd.DataFrame(data, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base', 'Taker buy quote', 'Ignore'])
    return df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]


def get_binance_klines_local(file_path, data_length, end_time=''):
    df = pd.read_csv(file_path, index_col=0)

    if not end_time == '':
        end_index = df[df.datetime <= end_time].index[-1]
        start_index = max(end_index - data_length + 1, 0)
        df = df[start_index: end_index + 1]
    else:
        df = df[-data_length:]

    df.reset_index(drop=True, inplace=True)
    return df

# Data Normalization
def normalize(df, cols):
    norm_df = df[cols].copy()
    min_max_scaler = MinMaxScaler()

    for col in cols:
        norm_df[col] = min_max_scaler.fit_transform(df[f'{col}'].values.reshape(-1,1))
    
    return norm_df

def denormalize(df, norm_data):
    min_max_scaler = MinMaxScaler()

    min_max_scaler.fit_transform(df)
    denorm_data = min_max_scaler.inverse_transform(norm_data)

    return denorm_data


# Dataset Preparation
def prepare_dataset_sliding(df, time_frame):
    nparray = df.to_numpy()

    result = []
    for i in range(len(nparray) - (time_frame - 1) - 1):
        result.append(nparray[i: i + (time_frame + 1)])
    result = np.array(result)
    
    X = result[:, :-1]
    y = result[:, -1]
    
    # print(result)
    # print(f'X: {X}')
    # print(f'y: {y}')

    return X, y

def prepare_data_multiple(df, time_frame, predictions):
    nparray = df.to_numpy()

    result = []
    for i in range(len(nparray) - (time_frame - 1) - predictions):
        result.append(nparray[i: i + (time_frame + predictions)])
    result = np.array(result)
    
    X = result[:, :-predictions]
    y = result[:, -predictions:][:, :, -1]
    
    # print(f'result: {result}')
    # print(f'X: {X}')
    # print(f'y: {result[:, -predictions:][:, :, -1]}')
    # print(f'y2: {result[:, -predictions:][:, :, -1]}')

    return X, y

def create_look_back_dataset(X, y, look_back=1):
    dataX, dataY = [], []
    for i in range(1, len(X) - look_back + 1):
    # for i in range(len(X) - look_back):
        dataX.append(X[i:(i + look_back)])
        dataY.append(y[i + look_back - 1])  # y的索引需要小心调整
    return np.array(dataX), np.array(dataY)

def clean_data(df):
    """
    处理DataFrame中的NaN和Inf值，填充NaN值、替换Inf值并进行插值处理。
    
    参数:
    df : pandas.DataFrame
        要处理的DataFrame。
        
    返回:
    df : pandas.DataFrame
        处理后的DataFrame。
    """
    
    # 1. 填充 NaN 值为 0
    df.fillna(0, inplace=True)
    
    # 2. 将 Inf 替换为 NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 3. 使用线性插值法处理 NaN 值
    df.interpolate(method='linear', inplace=True)
    
    # 4. 选择数值类型的列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 5. 检查 NaN 值的数量
    nan_counts = df[numeric_cols].isna().sum()
    print("NaN Counts:\n", nan_counts)
    
    # 6. 检查 Inf 值的数量
    inf_counts = np.isinf(df[numeric_cols]).sum()
    print("Infinity Counts:\n", inf_counts)
    
    # 7. 打印 NaN 和 Inf 总数量
    total_nan_inf = nan_counts + inf_counts
    print("Total NaN and Infinity Counts:\n", total_nan_inf)
    
    # 8. 检查所有列中 NaN 值的行
    nan_rows = df[df[numeric_cols].isna().any(axis=1)]
    inf_rows = df[np.isinf(df[numeric_cols]).any(axis=1)]
    print("Rows with NaN columns:\n", nan_rows)
    print("Rows with Inf columns:\n", inf_rows)
    
    # 9. 重置索引
    df.reset_index(drop=True, inplace=True)
    
    return df

def generate_kline(length, lowest, highest, lowestIndex, highestIndex, first_close):
    if not (0 <= lowestIndex < length and 0 <= highestIndex < length):
        raise ValueError("lowestIndex and highestIndex must be within the range of length")
    
    # 初始化空的K線數據
    opens = np.zeros(length)
    closes = np.zeros(length)
    highs = np.zeros(length)
    lows = np.zeros(length)
    volume = np.zeros(length)
    
    # 第一根 K 線的開盤價和收盤價
    opens[0] = np.random.uniform(lowest, first_close)
    closes[0] = first_close
    highs[0] = max(opens[0], closes[0], np.random.uniform(opens[0], highest))
    lows[0] = min(opens[0], closes[0], np.random.uniform(lowest, opens[0]))
    volume[0] = np.random.uniform(1000, 20000)

    # 隨機生成價格，並確保最高和最低價格在指定的index
    for i in range(1, length):
        if i == lowestIndex:
            low = lowest
            high = np.random.uniform(low, highest)
        elif i == highestIndex:
            high = highest
            low = np.random.uniform(lowest, high)
        else:
            low = np.random.uniform(lowest, highest)
            high = np.random.uniform(low, highest)

        # 開盤價一定是上一根的收盤價
        open_price = closes[i - 1]
        close_price = np.random.uniform(low, high)

        # 設置每根K線的價格，並確保 high 是最高，low 是最低
        opens[i] = open_price
        closes[i] = close_price
        highs[i] = max(open_price, close_price, high)
        lows[i] = min(open_price, close_price, low)
        volume[i] = np.random.uniform(1000, 20000)

    # 將數據打包到 DataFrame 中以便觀察
    kline_data = pd.DataFrame({
        'Open': opens,
        'Close': closes,
        'High': highs,
        'Low': lows,
        'Volume': volume
    })

    return kline_data
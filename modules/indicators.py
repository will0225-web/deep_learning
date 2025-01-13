import numpy as np
import pandas as pd
from modules import utilities as utilities

def calculate_percentage_change(future_price, entry_price):
    return ((future_price - entry_price) / entry_price)
@utilities.capture_args
def add_price_max_min_indicator(df, look_back=14):
    # 计算look_back期间的最高High和最低Low
    df['max_high_look_back'] = df['High'].rolling(window=look_back).max()
    df['min_low_look_back'] = df['Low'].rolling(window=look_back).min()
    
    return df

@utilities.capture_args
def calculate_sma(data, window=20):
    return data.rolling(window=window).mean()

@utilities.capture_args
# MACD計算函數
def calculate_MACD(data, short_window=12, long_window=26):
    # Short term EMA
    ShortEMA = data.ewm(span=short_window, adjust=True).mean()
    # Long term EMA
    LongEMA = data.ewm( span=long_window, adjust=True).mean()
    # Calculate MACD line
    MACD = ShortEMA - LongEMA
    # Calculate signal line
    signal = MACD.ewm(span=9, adjust=True).mean()

    hist = MACD - signal
    return MACD, signal, hist

@utilities.capture_args
# RSI
def calculate_RMA(data, period):
    sma = calculate_sma(data, period)
    data_series = pd.Series(data)
    sma_series = pd.Series(sma)
    
    # 使用SMA的前period筆資料作為RMA的初始值
    rma = pd.concat([sma_series.iloc[:period], data_series.iloc[period:]])
    
    alpha = 1 / period
    rma = rma.ewm(alpha=alpha, adjust=False).mean()
    
    return rma

@utilities.capture_args
def calculate_ema(data, period):
    sm_factor = 2 / (period + 1)
    return data.ewm(alpha=sm_factor, adjust=False).mean()

@utilities.capture_args
def calculate_stochastic_oscillator(df, period=14):
    # 计算%K
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    k = ((df['Close'] - low_min) / (high_max - low_min)) * 100

    # 计算%D（%K的3期简单移动平均）
    d = k.rolling(window=3).mean()
    return k, d

@utilities.capture_args
def calculate_RSI(data, window):
    change = data.astype(float).diff(1).fillna(0)

    up = calculate_RMA(change.apply(lambda x: max(x, 0)), window)
    down = calculate_RMA(change.apply(lambda x: -min(x, 0)), window)
    
    rsi = pd.Series(0.0, index=data.index)
    
    for i in range(window, len(data)):
        if down[i] == 0:
            rsi[i] = 100
        elif up[i] == 0:
            rsi[i] = 0
        else:
            rsi[i] = 100 - (100 / (1 + up[i] / down[i]))
    
    return rsi

@utilities.capture_args
# 計算 TR
def calculate_tr(df, period):
    tr = np.max([df['High'] - df['Low'], abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())], axis=0)

    return tr

@utilities.capture_args
# 計算 ATR
def calculate_atr(tr, period):
    atr = calculate_RMA(tr, period)
    
    return atr
    
@utilities.capture_args
# 計算 Supertrend
def calculate_supertrend(df, period, multiplier, changeATR=True):
    atr = 0.0
    
    if changeATR:
        tr = calculate_tr(df, period)
        tr_series = pd.Series(tr)
        atr = calculate_atr(tr_series, period)
    else:
        tr = calculate_tr(df, period)
        tr_series = pd.Series(tr)
        atr = calculate_sma(tr_series, period)

    src = (df['High'] + df['Low']) / 2
    close_prev = df['Close'].shift(1)

    atr_list = atr.tolist()
    src_list = src.tolist()

    up_list = [None]
    # 從第二筆資料開始計算
    for i in range(1, len(df)):
        up = (src_list[i] - multiplier * atr_list[i])

        if up_list[i - 1] is not None:
            up1 = up_list[i - 1]
        else:
            up1 = up
        
        if close_prev.iloc[i] > up1:
            up = max(up, up1)
        else:
            up = up

        up_list.append(up)

    # 將計算出來的 up 值指派回 df
    df['Up Trend'] = up_list
    

    # 從第二筆資料開始計算
    dn_list = [None]
    for i in range(1, len(df)):
        dn = (src_list[i] + multiplier * atr_list[i])

        if dn_list[i - 1] is not None:
            dn1 = dn_list[i - 1]
        else:
            dn1 = dn
        
        if close_prev.iloc[i] < dn1:
            dn = min(dn, dn1)
        else:
            dn = dn

        dn_list.append(dn)

    # 將計算出來的 up 值指派回 df
    df['Down Trend'] = dn_list

    supertend_list = [1]
    # # 從第二行開始進行計算，因為第一行沒有前一行的數據
    for i in range(1, len(df)):
        if supertend_list[i - 1] == -1 and df.at[i, 'Close'] > (dn_list[i - 1] if dn_list[i - 1] is not None else 0.0):
            supertend_list.append(1)
        elif supertend_list[i - 1] == 1 and df.at[i, 'Close'] < (up_list[i - 1] if up_list[i - 1] is not None else 0.0):
            supertend_list.append(-1)
        else:
            supertend_list.append(supertend_list[i - 1])

    return up_list, dn_list, supertend_list

@utilities.capture_args
def compute_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    
    return rolling_mean, rolling_mean + (rolling_std * num_std), rolling_mean - (rolling_std * num_std)

@utilities.capture_args
def add_price_indicator(df, look_back=14):
    # 计算look_back期间的最高High和最低Low
    df['max_high_look_back'] = df['High'].rolling(window=look_back).max()
    df['min_low_look_back'] = df['Low'].rolling(window=look_back).min()
    
    # 计算斐波那契回撤水平
    df['fibonacci_0.382'] = df['min_low_look_back'] + 0.382 * (df['max_high_look_back'] - df['min_low_look_back'])
    df['fibonacci_0.5'] = df['min_low_look_back'] + 0.5 * (df['max_high_look_back'] - df['min_low_look_back'])
    df['fibonacci_0.618'] = df['min_low_look_back'] + 0.618 * (df['max_high_look_back'] - df['min_low_look_back'])
    
    return df

@utilities.capture_args
def trend_min_max_price_indicator(df, look_back=96):
    # 初始化支撑点和压力点列
    df['Current_Max_High'] = np.nan
    df['Current_Min_Low'] = np.nan
    df['Current_Max_date'] = np.nan
    df['Current_Min_date'] = np.nan

    # 查找支撑点和压力点
    for i in range(look_back, len(df)):
        # 查找支撑点
        current_min_price = df['Low'].iloc[i-look_back:i+1].min()
        current_min_index = df['Low'].iloc[i-look_back:i+1].idxmin()
        
        expansion_factor = 1

        while True:
            if current_min_index == i:
                # 如果找到的最低值索引与当前检查的K线索引相同
                # 则逐步扩大查找范围
                end_index = current_min_index
                start_index = max(0, end_index - look_back * expansion_factor)
                new_min_price = df['Low'].iloc[start_index:end_index+1].min()
                new_min_index = df['Low'].iloc[start_index:end_index+1].idxmin()
                if new_min_price < current_min_price:
                    current_min_price = new_min_price
                    current_min_index = new_min_index
                else:
                    expansion_factor += 1
                    if start_index == 0:
                        current_min_price = df['Low'].iloc[i]
                        current_min_index = i
                        break
            else:
                # 从当前最小值位置继续往前look_back查找
                end_index = current_min_index
                start_index = max(0, end_index - look_back)
                new_min_price = df['Low'].iloc[start_index:end_index+1].min()
                new_min_index = df['Low'].iloc[start_index:end_index+1].idxmin()
                if new_min_price < current_min_price:
                    current_min_price = new_min_price
                    current_min_index = new_min_index
                else:
                    break

        df.loc[df.index[i], 'Current_Min_Low'] = current_min_price
        df.loc[df.index[i], 'Current_Min_date'] = df.loc[current_min_index, 'datetime']
        df.loc[df.index[i], 'Current_Min_Index'] = current_min_index

        # 查找压力点
        current_max_price = df['High'].iloc[i-look_back:i+1].max()
        current_max_index = df['High'].iloc[i-look_back:i+1].idxmax()

        expansion_factor = 1

        while True:
            if current_max_index == i:
                # 如果找到的最高值索引与当前检查的K线索引相同
                # 则逐步扩大查找范围
                end_index = current_max_index
                start_index = max(0, end_index - look_back * expansion_factor)
                new_max_price = df['High'].iloc[start_index:end_index+1].max()
                new_max_index = df['High'].iloc[start_index:end_index+1].idxmax()
                if new_max_price > current_max_price:
                    current_max_price = new_max_price
                    current_max_index = new_max_index
                else:
                    expansion_factor += 1
                    if start_index == 0:
                        current_max_price = df['High'].iloc[i]
                        current_max_index = i
                        break
            else:
                # 从当前最大值位置继续往前look_back查找
                end_index = current_max_index
                start_index = max(0, end_index - look_back)
                new_max_price = df['High'].iloc[start_index:end_index+1].max()
                new_max_index = df['High'].iloc[start_index:end_index+1].idxmax()
                if new_max_price > current_max_price:
                    current_max_price = new_max_price
                    current_max_index = new_max_index
                else:
                    break

        df.loc[df.index[i], 'Current_Max_High'] = current_max_price
        df.loc[df.index[i], 'Current_Max_date'] = df.loc[current_max_index, 'datetime']
        df.loc[df.index[i], 'Current_Max_Index'] = current_max_index

        # 在最后进行检查并更新
        if current_max_index < current_min_index:
            current_min_price = df['Low'].iloc[current_max_index:current_min_index+1].min()
            current_min_index = df['Low'].iloc[current_max_index:current_min_index+1].idxmin()
        elif current_min_index < current_max_index:
            current_max_price = df['High'].iloc[current_min_index:current_max_index+1].max()
            current_max_index = df['High'].iloc[current_min_index:current_max_index+1].idxmax()

        df.loc[df.index[i], 'Current_Max_High'] = current_max_price
        df.loc[df.index[i], 'Current_Max_date'] = df.loc[current_max_index, 'datetime']
        df.loc[df.index[i], 'Current_Max_Index'] = current_max_index
        
        df.loc[df.index[i], 'Current_Min_Low'] = current_min_price
        df.loc[df.index[i], 'Current_Min_date'] = df.loc[current_min_index, 'datetime']
        df.loc[df.index[i], 'Current_Min_Index'] = current_min_index
    
    df = calculate_fibonacci(df)

    return df

@utilities.capture_args
def calculate_ahead_high_low(df, percentage=0.001):
    df['Future_High'] = df['Close'] * (1 + percentage)
    df['Future_Low'] = df['Close'] * (1 - percentage)
    return df

@utilities.capture_args
def calculate_fibonacci(df):
    max_index = df['Current_Max_Index']
    min_index = df['Current_Min_Index']
    current_max_high = df['Current_Max_High']
    current_min_low = df['Current_Min_Low']

    # 使用 np.where 进行向量化操作
    high = np.where(max_index < min_index, current_max_high, current_min_low)
    low = np.where(max_index < min_index, current_min_low, current_max_high)
    

    df['fibonacci_0'] = low
    df['fibonacci_1'] = high
    df['fibonacci_0.382'] = low + 0.382 * (high - low)
    df['fibonacci_0.5'] = low + 0.5 * (high - low)
    df['fibonacci_0.618'] = low + 0.618 * (high - low)
    return df

@utilities.capture_args
def calculate_candelstick_patterns(df):

    df['Open_Close_pct'] = (df['Close'] - df['Open']) / df['Open']
    df['High_Low_pct'] = (df['High'] - df['Low']) / df['Low']

    df['Up_Shadow_pct'] = np.where(df['Open'] > df['Close'], (df['High'] - df['Open']) / df['Open'], (df['High'] - df['Close']) / df['Close'])
    df['Down_Shadow_pct'] = np.where(df['Open'] > df['Close'], (df['Close'] - df['Low']) / df['Close'], (df['Open'] - df['Low']) / df['Open'])
    # df['Close_High_pct'] = (df['High'] - df['Close']) / df['Close']
    # df['Close_Low_pct'] = (df['Close'] - df['Low']) / df['Low']
    # df['Open_Low_pct'] = (df['Open'] - df['Low']) / df['Open']
    
    return df

@utilities.capture_args
def add_24h_min_max_price(df, look_back=14):
    # 计算look_back期间的最高High和最低Low
    df['max_high_look_back_24h'] = df['High'].rolling(window=look_back).max()
    df['min_low_look_back_24h'] = df['Low'].rolling(window=look_back).min()

    return df

@utilities.capture_args
def add_three_klines(df, percentage_threshold=0.005):
    # 创建用于识别模式的辅助列
    df['kline_color_shift_1'] = df['kline_color'].shift(1)
    df['kline_color_shift_2'] = df['kline_color'].shift(2)

    # 创建用于识别模式的辅助列
    df['close_shift_2'] = df['Close'].shift(2)

    df['close_change'] = (df['Close'] - df['close_shift_2']) / df['close_shift_2']


    # 红三兵
    df['red_three_soldiers'] = (
        (df['kline_color'] == 0) & 
        (df['kline_color_shift_1'] == 0) & 
        (df['kline_color_shift_2'] == 0) &
        ((df['close_change'] <= -percentage_threshold) & (df['close_change'] > -0.08))
    ).astype(int)

    # 绿三兵
    df['green_three_soldiers'] = (
        (df['kline_color'] == 1) & 
        (df['kline_color_shift_1'] == 1) & 
        (df['kline_color_shift_2'] == 1) &
        ((df['close_change'] >= percentage_threshold) & (df['close_change'] < 0.08))
    ).astype(int)


    # 删除辅助列
    df.drop(['kline_color_shift_1', 'kline_color_shift_2', 'close_shift_2', 'close_change'], axis=1, inplace=True)

    return df

@utilities.capture_args
def calculate_supertrend_risk(df):
    # 假设 df 已经有了 'Close', 'Low', 'Up Trend' 这些列
    
    # 1. close与up trend的距离%差
    close_up_diff_pct = ((df['Close'] - df['Up Trend']) / df['Up Trend']).abs()
    # 将其转换为对应的百分比范围内的值（0-3% 对应 0-45%）
    close_up_score = close_up_diff_pct.clip(upper=0.03) / 0.03 * 45
    
    # 2. low与up trend的距离%差
    low_up_diff_pct = ((df['Low'] - df['Up Trend']) / df['Up Trend']).abs()
    # 将其转换为对应的百分比范围内的值（0-2% 对应 0-45%）
    low_up_score = low_up_diff_pct.clip(upper=0.02) / 0.02 * 45
    
    # 3. close与low的距离%差
    close_low_diff_pct = ((df['Close'] - df['Low']) / df['Low']).abs()
    # 将其转换为对应的百分比范围内的值（0-1% 对应 0-10%）
    close_low_score = close_low_diff_pct.clip(upper=0.01) / 0.01 * 10
    
    # 综合得分
    df['Long Supertrend Score'] = close_up_score + low_up_score + close_low_score
    # 确保总分不超过100%
    df['Long Supertrend Score'] = np.where(df['Super Trend'] == 1, df['Long Supertrend Score'].clip(upper=100) / 100, 0)


    # 1. close与up trend的距离%差
    close_down_diff_pct = ((df['Down Trend'] - df['Close']) / df['Down Trend']).abs()
    # 将其转换为对应的百分比范围内的值（0-3% 对应 0-45%）
    close_down_score = close_down_diff_pct.clip(upper=0.03) / 0.03 * 45
    
    # 2. low与up trend的距离%差
    high_down_diff_pct = ((df['Down Trend'] - df['High']) / df['Down Trend']).abs()
    # 将其转换为对应的百分比范围内的值（0-2% 对应 0-45%）
    high_down_score = high_down_diff_pct.clip(upper=0.02) / 0.02 * 45
    
    # 3. close与low的距离%差
    close_high_diff_pct = ((df['High'] - df['Close']) / df['High']).abs()
    # 将其转换为对应的百分比范围内的值（0-1% 对应 0-10%）
    close_high_score = close_high_diff_pct.clip(upper=0.01) / 0.01 * 10
    
    # 综合得分
    df['Short Supertrend Score'] = close_down_score + high_down_score + close_high_score
    # 确保总分不超过100%
    df['Short Supertrend Score'] = np.where(df['Super Trend'] == -1, df['Short Supertrend Score'].clip(upper=100) / 100, 0)
    
    return df

@utilities.capture_args
def super_trend_delta_and_risk(df):
    df['Delta_Low_Up_Percentage'] = df.apply(lambda x: (x['Low'] - x['Up Trend']) / x['Up Trend'] if x['Super Trend'] == 1 else 0, axis=1)
    
    df['Delta_High_Down_Percentage'] = df.apply(lambda x: (x['Down Trend'] - x['High']) / x['Down Trend'] if x['Super Trend'] == -1 else 0, axis=1)

    df['Delta_Long_Close_Trend_Percentage'] = df.apply(lambda x: (x['Close'] - x['Up Trend']) / x['Up Trend'] if x['Super Trend'] == 1 else 0, axis=1)

    df['Delta_Short_Close_Trend_Percentage'] = df.apply(lambda x: (x['Down Trend'] - x['Close'])  / x['Down Trend'] if x['Super Trend'] == -1 else 0, axis=1)
    return df

@utilities.capture_args
def super_trend_delta_and_risk_v1(df):
    df['Delta_Low_Up'] = df.apply(lambda x: x['Low'] - x['Up Trend'] if x['Super Trend'] == 1 else 0, axis=1)
    
    df['Delta_High_Down'] = df.apply(lambda x: x['Down Trend'] - x['High'] if x['Super Trend'] == -1 else 0, axis=1)

    df['Delta_Close_Trend'] = df.apply(lambda x: x['Close'] - x['Up Trend'] if x['Super Trend'] == 1 else x['Close'] - x['Down Trend'], axis=1)
    return df


@utilities.capture_args
def add_lower_low_higher_high(df, higher_or_lower_threshold=0.05, look_back=14):
    # 计算期间开始和结束时的High和Low
    df['start_high'] = df['High'].shift(look_back - 1)
    df['end_high'] = df['High']
    df['start_low'] = df['Low'].shift(look_back - 1)
    df['end_low'] = df['Low']


    # 判断High和Low的趋势是否向下，并且期间内的最高和最低差距是否达到threshold
    df['lower_low'] = (
        (df['end_high'] < df['start_high']) &
        (df['end_low'] < df['start_low']) &
        ((-(df['end_low'] - df['start_high']) / df['start_high']) >= higher_or_lower_threshold)
    ).astype(int)

    # 判断High和Low趋势是否向上，并且期间内的最高和最低High差距是否达到threshold
    df['higher_high'] = (
        (df['end_high'] > df['start_high']) &
        (df['end_low'] > df['start_low']) &
        ((df['end_high'] - df['start_low']) / df['start_low'] >= higher_or_lower_threshold)
    ).astype(int)

    # 删除辅助列
    df.drop(['start_high', 'end_high', 'start_low', 'end_low'], axis=1, inplace=True)
    return df

@utilities.capture_args
def add_rsi_overbought_oversold(df, overbought_threshold=70, oversold_threshold=30):
    # 檢查RSI是否超買
    df['rsi_overbought'] = (df['RSI'] > overbought_threshold).astype(int)

    # 檢查RSI是否超賣
    df['rsi_oversold'] = (df['RSI'] < oversold_threshold).astype(int)

    return df

@utilities.capture_args
def calculate_macd_hist_continuity(df):
    df['macd_hist_stronger_continuity'] = 0
    df['macd_hist_weaker_continuity'] = 0

    macd_stronger_status = df['MACD_Stronger']

    for i in range(len(df)):
        continuity_counter = 0

        if macd_stronger_status[i] == 1:  # 持續走弱
            for j in range(i, -1, -1):
                if macd_stronger_status[j] == 1:
                    continuity_counter += 1
                else:
                    break
            df.at[i, 'macd_hist_weaker_continuity'] = continuity_counter

        elif macd_stronger_status[i] == 2:  # 持續走強
            for j in range(i, -1, -1):
                if macd_stronger_status[j] == 2:
                    continuity_counter += 1
                else:
                    break
            df.at[i, 'macd_hist_stronger_continuity'] = continuity_counter
    return df

@utilities.capture_args
def calculate_volume_sma(df, period=20):
    df['Volume_SMA'] = df['Volume'].rolling(window=period).mean()

    return df


@utilities.capture_args
def tops_and_downs(df, lookahead=8):
    tops = []
    bottoms = []

    is_current_max_high_appear = False
    max_high = df['High'].iloc[0]  # 初始设置为第一根K线的最高价
    max_datetime = df['datetime'].iloc[0]

    is_current_min_low_appear = False
    min_low = df['Low'].iloc[0]  # 初始设置为第一根K线的最低价
    min_datetime = df['datetime'].iloc[0]
    

    for i in range(len(df) - lookahead):
        current_low = df['Low'].iloc[i]
        current_high = df['High'].iloc[i]
        current_datetime = df['datetime'].iloc[i]

        if current_low < min_low:
            future_lows = df['Low'].iloc[i+1:i+lookahead+1]
            if future_lows.min() >= current_low:
                min_low = current_low
                min_datetime = current_datetime
                min_index = i
                is_current_min_low_appear = True
                # 没有被更新，则确认底峰
                bottoms.append((min_index, min_low, min_datetime))
                

        if current_high > max_high:
            # 检查未来 lookahead 根 K 线是否没有更新最高值
            future_highs = df['High'].iloc[i+1:i+lookahead+1]
            if future_highs.max() <= current_high:
                max_high = current_high
                max_datetime = current_datetime
                max_index = i
                is_current_max_high_appear = True
                # 没有被更新，则确认顶峰
                tops.append((max_index, max_high, max_datetime))
        
        if is_current_max_high_appear or is_current_min_low_appear:
            is_current_max_high_appear = False
            is_current_min_low_appear = False

            max_high = df['High'].iloc[i]
            max_datetime = df['datetime'].iloc[i]
            max_index = i
            min_low = df['Low'].iloc[i]
            min_datetime = df['datetime'].iloc[i]
            min_index = i
    return tops, bottoms

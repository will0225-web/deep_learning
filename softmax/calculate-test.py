import sys
import os

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import datetime
import csv
import pytz

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import time_convert as time_convert

# units_start = int(input('輸入units開始整數:'))
# units_end = int(input('輸入units結束整數:'))

# 特徵欄位
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'SMA', 'MACD', 'Signal', 'Hist', 'Up Trend', 'Down Trend', 'Super Trend']
symbol = "ETHUSDT"
interval = "15m"
look_back = 96 #使用回看1天數據

# 當前
end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# 特定
# end_time_string = "2023-08-31 23:59:59"

# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
klines = modules_data.get_binance_klines_backward(symbol, interval, end_time, 5000)
# klines = modules_data.get_binance_klines_forward(symbol=symbol, interval=interval, start_time=end_time, total_klines=100)

# filename = "all_kline_until_end_of_Aug.csv"
# klines = pd.read_csv(filename)
# Create DataFrame
df_temp = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base', 'Taker buy quote', 'Ignore'])

df = df_temp[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

# 四捨五入
df.round(2)

# 計算macd
# df['MACD'], df['Signal'], df['Hist'] = indicators.calculate_MACD(df['Close'], 12, 29)
# df['MACD_Cross'] = signals.check_macd_hist_cross(df['Hist'])

# df['SMA'] = indicators.calculate_sma(df['Close'], 27)

# 計算RSI
df['RSI'] = indicators.calculate_RSI(df['Close'], 14)

# ATR
# df['TR'] = indicators.calculate_tr(df, 10)
# df['ATR'] = indicators.calculate_atr(df['TR'], 10)
# df['RMA'] = indicators.calculate_RMA(df['Close'], 20)

# df['Up Trend'], df['Down Trend'], df['Super Trend'] = indicators.calculate_supertrend(df, 10, 3)

# 轉換float
df = df.astype(float)

# 四捨五入
df = df.round(2)

df['Target'] = signals.set_rsi_target(df['Close'], df['RSI'])
df = pd.concat([df, pd.get_dummies(df['Target'], prefix='Target')], axis=1)

# 再拿掉前後無參考性資料
df = df[1000:-300]

df['Time'] = df_temp['Open time'].apply(time_convert.convert_to_local_time)

df.to_csv("test.csv")

# 創建X, y數據集
X, y = [], []
for i in range(look_back, len(df) + 1):
    d = df[i - look_back:i]
    dd = df.iloc[i - 1][['Target_0', 'Target_1', 'Target_2']].values
    X.append(df[i - look_back:i])
    y.append(df.iloc[i - 1][['Target_0', 'Target_1', 'Target_2']].values)


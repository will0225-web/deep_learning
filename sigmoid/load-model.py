import numpy as np
import tensorflow as tf
import requests
import datetime
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# MACD Cross
macd_long_hist = 0.26
macd_short_hist = -0.3
def check_valid_macd_hist_cross(data):
    macd_hist_cross = []

    for i in range(len(data)):
        if i == 0:
            macd_hist_cross.append(0)
        elif data[i - 1] < 0 and data[i] > macd_long_hist:
            macd_hist_cross.append(1)
        elif data[i - 1] > 0 and data[i] < macd_short_hist:
            macd_hist_cross.append(-1)
        else:
            macd_hist_cross.append(0)
    
    return macd_hist_cross

# RSI
def calculate_RMA(data, window):
    rma = pd.Series(0.0, index=data.index)
    n = len(data)
    
    for i in range(1, n):
        rma[i] = (rma[i-1] * (window - 1) + data[i]) / window
    
    return rma

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

def get_binance_klines_backward(symbol, interval, total_klines=500):
    url = "https://fapi.binance.com/fapi/v1/klines"
    all_klines = []
    
    # Binance限制每次請求只能取得500筆K線
    limit = min(500, total_klines)
    num_requests = total_klines // limit
    
    # 使用當前時間為end_time開始
    end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000
    # 定义字符串日期
    # end_time_string = "2023-09-29 02:30:00"

    # # 轉毫秒
    # end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
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
    
    return all_klines

# 定义参数
symbol = "ETHUSDT"
interval = "15m"
cols = ['Open', 'High', 'Low', 'Close', 'RSI', 'Hist', 'MACD_Cross', 'Volume']
look_back = 96 # 使用前96根的數據預測下一天

# 获取 K 线数据
klines = get_binance_klines_backward(symbol, interval, look_back)
# 假設這是你的單一K線資料
random_choice = random.choice(klines)

# 載入已經訓練好的模型
model = tf.keras.models.load_model('1695823805:units=128:loss=0.3654:accuracy=0.8402:20000K:1%')

df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base', 'Taker buy quote', 'Ignore'])
_, _, df['Hist'] = calculate_MACD(df['Close'], 12, 29)
df['MACD_Cross'] = check_valid_macd_hist_cross(df['Hist'])

df['RSI'] = calculate_RSI(df['Close'], 14)
df = df[cols].astype(float)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# 將這些數據組裝成正確的形狀
input_data = np.array([scaled_data])

# 使用模型進行預測
predicted = model.predict(input_data)

# 根据预测结果判断上涨、下跌或无法判定
# predicted_target = np.argmax(predicted) - 1  # 由于您的标签是 [-1, 0, 1]，所以需要减1
print(predicted)
if predicted >= 0.5:
    trend = "上涨"
else:
    trend = "下跌"
    

# 获取并展示最后一个数据点的开盘、收盘、最高、最低价
last_kline = df.iloc[-1]

print(f"预测的最后一根K线是：{trend}")
print("开盘价：", last_kline['Open'])
print("最高价：", last_kline['High'])
print("最低价：", last_kline['Low'])
print("收盘价：", last_kline['Close'])

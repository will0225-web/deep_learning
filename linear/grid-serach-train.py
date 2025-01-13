import sys, os
sys.path.append(os.getcwd())
from datetime import datetime
from modules.model import train_sliding_predict
from modules.data import get_binance_klines_backward

symbol = 'ETHUSDT'
interval = '15m'
cols = ['Open', 'High', 'Low', 'Close']
look_back = 20
end_time = int(datetime.timestamp(datetime.now())) * 1000

klines_length_start = 30000
klines_length_end = 50000
klines_length_step = 10000

for i in range(klines_length_start, klines_length_end + 1, klines_length_step):
    klines = get_binance_klines_backward(symbol, interval, end_time, total_klines=i)
    for l in range(24, 97, 24):
        train_sliding_predict(klines, cols=cols, kline_length=i, look_back=l)

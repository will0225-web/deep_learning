import sys, os
sys.path.append(os.getcwd())

from datetime import datetime
from modules.model import train_softmax_model
from modules.data import get_binance_klines_backward

symbol = 'ETHUSDT'
interval = '15m'
end_time = int(datetime.timestamp(datetime.now())) * 1000

klines_length_start = 35000
klines_length_end = 95000
klines_length_step = 30000

for i in range(klines_length_start, klines_length_end + 1, klines_length_step):
    klines = get_binance_klines_backward(symbol, interval, end_time, total_klines=i)
    for l in range(24, 97, 24):
        train_softmax_model(klines, look_back=l)

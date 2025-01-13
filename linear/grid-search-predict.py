import sys, os
sys.path.append(os.getcwd())
import tensorflow as tf
from datetime import datetime
from modules.predict import sliding_predict
from modules.data import get_binance_klines_backward
from modules.parameters import get_input_shape

model_path = 'linear/trained_models/23.10.13/23.10.13-1246(sliding)'
end = 20000
symbol = 'ETHUSDT'
interval = '15m'
end_time = int(datetime.timestamp(datetime.now())) * 1000

# 載入已經訓練好的模型
model = tf.keras.models.load_model(model_path)

# 讀取 model 的 input shape 作為 look back，確保不會發生 input shape 跟 data shape 不同
look_back = get_input_shape(model)

klines = get_binance_klines_backward(symbol, interval, end_time, end)

for i in range(500, end + 1, 500):
    # 檢查 data_length 是否有大於 look_back，否則資料量不夠會噴錯無法進行預測
    # 要是 look_back 大於 data_length 和 500，又剛好沒辦法被 500 整除，也可能會因資料量不夠噴錯
    data_length = max(look_back, i)
    data_length = (data_length // 500 + 1) * 500 if (data_length % 500 != 0 and data_length > 500) else data_length
    print(f'\nData Length: {data_length}\n')

    klines_cropped = klines[-data_length:]

    sliding_predict(model_path, model, data_length=data_length, look_back=look_back, klines=klines_cropped)
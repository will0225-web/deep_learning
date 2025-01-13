import sys, os
sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf
import pandas as pd
import json
from datetime import datetime
from modules.data import get_binance_klines_backward, normalize, denormalize
from modules.plot import print_predict

# 定義參數
model_path = 'linear/trained_models/23.10.12/23.10.12-1812(sliding)'
json_file = open(f'{model_path}/model-parameters.json')
model_parameters = json.load(json_file)
print(f'\nModel: {model_path[model_path.rfind("/") + 1:]} is predicting...\n')

symbol = 'ETHUSDT'
interval = '15m'
cols = model_parameters['Dataframe Columns'] if not model_parameters['Dataframe Columns'] is None else ['Open', 'High', 'Low', 'Close']

close_column_index = cols.index('Close')
end_time = int(datetime.timestamp(datetime.now())) * 1000

# 載入已經訓練好的模型
model = tf.keras.models.load_model(model_path)

# 讀取 model 的 input shape 作為 look back，確保不會發生 input shape 跟 data shape 不同
layer1 = model.get_layer(index=0)
input_shape = layer1.get_config()['batch_input_shape']
look_back = input_shape[1]
data_length = look_back if 'multi' in model_path else 96
# 檢查 data_length 是否有大於 look_back，否則資料量不夠會噴錯無法進行預測
# 要是 look_back 大於 data_length，又剛好沒辦法被 500 整除，也可能會因資料量不夠噴錯
data_length = max(look_back, data_length)
data_length = (data_length // 500 + 1) * 500 if (data_length % 500 != 0 and data_length > 500) else data_length

# 獲取 K 線數據
klines = get_binance_klines_backward(symbol, interval, end_time, data_length)

df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base', 'Taker buy quote', 'Ignore'])
df = df[cols].astype(float)
print(f'DF: {df}')
# 使用MinMaxScaler對數據進行縮放
scaled_data = normalize(df, cols)

# 使用模型進行預測
if 'multi' in model_path:
    predicted_candlesticks = model.predict(np.array([scaled_data]))
elif 'sliding' in model_path:
    future_steps = 4
    input_data = []

    for i in range(data_length - look_back + 1):
        # print(scaled_data[i: i + look_back])
        input_data.append(np.array(scaled_data[i: i + look_back]))
    # print(f'Input Data: {input_data}')
    predicted_candlesticks = model.predict(np.array(input_data))

    for i in range(future_steps - 1):
        scaled_data = np.insert(scaled_data, len(scaled_data), predicted_candlesticks[-1], axis=0)

        last_window = scaled_data[- look_back:]
        predict_candlestick = model.predict(np.array([last_window]))
        predicted_candlesticks = np.insert(predicted_candlesticks, len(predicted_candlesticks), predict_candlestick[0], axis=0)

    pd.set_option('display.precision', 2)    
    print(f'\n最近二十根K線:\n{pd.DataFrame(denormalize(df, last_window)[-20:])}\n')
    
    # with np.printoptions(precision=2, suppress=True):
    #     print(f'Predicted Candlesticks: {denormalize(df, predicted)}')
   
print_predict(model_path, df, predicted_candlesticks[- future_steps:], close_column_index)

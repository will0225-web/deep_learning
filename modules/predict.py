import sys, os
sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf
import pandas as pd
import json
from modules.data import normalize, denormalize
from modules.plot import print_predict

def sliding_predict(model_path, model, data_length, look_back, klines):
    json_file = open(f'{model_path}/model-parameters.json')
    model_parameters = json.load(json_file)
    print(f'\nModel: {model_path[model_path.rfind("/") + 1:]} is predicting...\n')

    cols = model_parameters['Dataframe Columns'] if not model_parameters['Dataframe Columns'] is None else ['Open', 'High', 'Low', 'Close']
    close_column_index = cols.index('Close')

    df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base', 'Taker buy quote', 'Ignore'])
    df = df[cols[:close_column_index + 1]].astype(float)

    if 'Change' in cols:
        df['Change'] = df.Close.diff()
        df = df.dropna()

    print(f'\nDF:\n{df}\n')

    # 使用MinMaxScaler對數據進行縮放
    scaled_data = normalize(df, cols)

    # 使用模型進行預測
    future_steps = 4
    input_data = []

    for i in range(data_length - look_back + 1):
        # print(scaled_data[i: i + look_back])
        input_data.append(np.array(scaled_data[i: i + look_back]))
    print(f'Input Data dtype: {np.array(input_data).dtype}')
    predicted_candlesticks = model.predict(np.array(input_data))

    for i in range(future_steps - 1):
        scaled_data = np.insert(scaled_data, len(scaled_data), predicted_candlesticks[-1], axis=0)

        last_window = scaled_data[- look_back:]
        predict_candlestick = model.predict(np.array([last_window]))
        predicted_candlesticks = np.insert(predicted_candlesticks, len(predicted_candlesticks), predict_candlestick[0], axis=0)

    pd.set_option('display.precision', 2)    
    
    print_predict(model_path, df, predicted_candlesticks[- future_steps:], close_column_index)
    print(f'\n最近二十根K線:\n{pd.DataFrame(denormalize(df, last_window)[-20:])}\n')

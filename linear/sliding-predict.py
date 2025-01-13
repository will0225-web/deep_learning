import sys, os
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from datetime import datetime
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from modules.data import get_binance_klines_backward, normalize, denormalize, prepare_dataset_sliding
from modules.plot import plot

# 特徵欄位
cols = ['Open', 'High', 'Low', 'Close']
symbol = "ETHUSDT"
interval = "15m"
end_time = int(datetime.timestamp(datetime.now())) * 1000

look_back = 7 #回看N根K線數據
future_steps = 4

prediction_type = 'Sliding Predict'

# Step 1: 獲取數據
klines = get_binance_klines_backward(symbol, interval, end_time, 100000)

df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base', 'Taker buy quote', 'Ignore'])
df = df[cols].astype(float)

# 使用MinMaxScaler對數據進行縮放
scaled_data = normalize(df, cols)

X, y = prepare_dataset_sliding(scaled_data, look_back)

# 2. 分割訓練和測試數據
test_size = 0.167
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# 3. 建模與訓練
model = Sequential()

model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.4))

model.add(LSTM(units=64))
model.add(Dropout(0.4))

model.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=X_train.shape[2], kernel_initializer='uniform', activation='linear'))
model.summary()

optimizer = Adam(learning_rate=0.003) 
model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

epochs = 20
batch_size = 160

callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3, min_delta=0.01)

model.fit(X_train, y_train, batch_size, epochs, callbacks=[callback])

# 保存整個模型到目錄中
dir_name = datetime.now().strftime("%y.%m.%d-%H%M") + '(sliding)'
model.save(f'linear/{dir_name}')

# 紀錄模型訓練參數
model_parameters = {
    'Train Time': datetime.now(),
    'Prediction Type': prediction_type,
    'Dataframe Columns': cols,
    'Symbol': symbol,
    'Interval': interval,
    'Look Back': look_back,
    'Future Steps': future_steps,
    'Data Length': len(df.index),
    'Test Size': test_size,
    'Model Compile Config': model.get_compile_config(),
    'Model Metrics Result': model.get_metrics_result(),
    'Batch Size': batch_size,
    'Epoch': epochs
}

for l in range(len(model.layers)):
    layer = model.get_layer(index=l)
    layer_config = layer.get_config()
    model_parameters[f'layer {l + 1}'] = layer_config

# 預測模式1：只提供最新 look back 資料
# last_window = scaled_data[-look_back:]

# predicted_candlesticks = []

# for i in range(4):
#     predicted_candlestick = model.predict(np.array([last_window]))

#     predicted_candlesticks.append(predicted_candlestick[0])

#     last_window = np.insert(last_window[1:], look_back - 1, predicted_candlestick[0], axis=0)

# with np.printoptions(precision=2, suppress=True):
#     print(f'Predicted Candlesticks: {denormalize(df, predicted_candlesticks)[:, -1]}')
#     print(f'Last Window: {denormalize(df, last_window)}')

# 預測模式2：提供一定長度的歷史資料，讓模型有上下文可以參考
predict_data_length = 4000
predict_data_length = min(predict_data_length, len(df.index))

predict_data = scaled_data[-predict_data_length:]

input_data = []

for i in range(predict_data_length - look_back + 1):
    input_data.append(np.array(predict_data[i: i + look_back]))

predicted_candlesticks = model.predict(np.array(input_data))

for i in range(future_steps - 1):
    predict_data = np.insert(predict_data, len(predict_data), predicted_candlesticks[-1], axis=0)

    last_window = predict_data[- look_back:]
    predict_candlestick = model.predict(np.array([last_window]))
    predicted_candlesticks = np.insert(predicted_candlesticks, len(predicted_candlesticks), predict_candlestick[0], axis=0)

with np.printoptions(precision=2, suppress=True):
    print(f'\nPredicted Candlesticks:\n{denormalize(df, predicted_candlesticks)[- future_steps:]}')
    print(f'\nLast 20 Klines:\n{denormalize(df, last_window)[-20:]}')

model_parameters['Predict Data Length'] = predict_data_length
model_parameters['Predicted Candlesticks'] = denormalize(df, predicted_candlesticks[- future_steps:])

# 評估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

model_parameters['Test Metrics Results'] = model.get_metrics_result()

# 儲存模型參數與結果
parameters_json = json.dumps(model_parameters, indent=4, default=str)

with open (f'linear/{dir_name}/model-parameters.json', 'w') as outfile:
    outfile.write(parameters_json)

# 以 test data 進行預測
pred = model.predict(X_test)

denorm_pred = denormalize(df, pred)
denorm_ytest = denormalize(df, y_test)

# 將預測結果與答案放進同個 dataframe 中，並匯出成 csv 檔
d = { 'pred': denorm_pred[:, -1], 'ytest': denorm_ytest[:, -1] }
df_result = pd.DataFrame(data=d)
df_result['diff'] = df_result['pred'] - df_result['ytest']
df_result['diff_abs'] = df_result['diff'].abs()
df_result.describe().to_csv(f'linear/{dir_name}/describe.csv', float_format='%.2f')
df_result.to_csv(f'linear/{dir_name}/output.csv', float_format='%.2f')

pd.set_option('display.precision', 2)
print(df_result)

# 將預測結果為 y_test 答案繪圖進行比較
plot(denorm_pred[:, -1], denorm_ytest[:, -1], dir_name)

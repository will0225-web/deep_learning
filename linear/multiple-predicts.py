import sys, os
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from datetime import datetime
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from modules.data import get_binance_klines_backward, normalize, denormalize, prepare_data_multiple
from modules.plot import plot_subplots

# 特徵欄位
cols = ['Open', 'High', 'Low', 'Close']
symbol = 'ETHUSDT'
interval = '30m'
end_time = int(datetime.timestamp(datetime.now())) * 1000

look_back = 40 #回看N根K線數據
future_steps = 8

# Step 1: 獲取數據
klines = get_binance_klines_backward(symbol, interval, end_time, 1200)

df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base', 'Taker buy quote', 'Ignore'])
df = df[cols].astype(float)

# 使用MinMaxScaler對數據進行縮放
scaled_data = normalize(df, cols)

# 2. 分割訓練和測試數據
X, y = prepare_data_multiple(scaled_data, look_back, future_steps)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.167, random_state=42)

# 3. 建模與訓練
model = Sequential()

model.add(LSTM(units=256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))

model.add(LSTM(units=256))
model.add(Dropout(0.3))

model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=future_steps, activation='linear'))
model.summary()

optimizer = Adam(learning_rate=0.003) 
model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=160)

# 保存整個模型到目錄中
model.save('trained_model_directory')

last_window = scaled_data[-look_back:]

predicted_candlesticks = model.predict(np.array([last_window]))

with np.printoptions(precision=2, suppress=True):
    print(f'Predicted Candlesticks: {denormalize(df["Close"].values.reshape(-1, 1), predicted_candlesticks).reshape(-1, 1)}')
    print(f'Last Window: {denormalize(df, last_window)[-5:]}')

# 評估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 以 test data 進行預測
pred = model.predict(X_test)
denorm_pred = denormalize(df['Close'].values.reshape(-1, 1), pred[:, -1:])
denorm_ytest = denormalize(df['Close'].values.reshape(-1, 1), y_test[:, -1:])

plot_kline_amount = min(3, future_steps)

# 將預測結果與答案放進同個 dataframe 中，並匯出成 csv 檔
d = {}
for i in range(1, plot_kline_amount + 1):
    d[f'pred{i}'] = denormalize(df['Close'].values.reshape(-1, 1), pred[:, - i].reshape(-1, 1))[:, 0]
    d[f'ytest{i}'] = denormalize(df['Close'].values.reshape(-1, 1), y_test[:, - i].reshape(-1, 1))[:, 0]
df_result = pd.DataFrame(data=d)
df_result.to_csv('trained_model_directory/output.csv', float_format='%.2f')

pd.set_option('display.precision', 2)
print(df_result)

# 將預測結果為 y_test 答案繪圖進行比較
plot_subplots(plot_kline_amount, df['Close'], pred, y_test)

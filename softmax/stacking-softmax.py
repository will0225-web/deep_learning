import sys
import os

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import datetime
import pandas as pd
import tensorflow as tf

from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, PReLU, Conv1D, MaxPooling1D, Flatten, LeakyReLU, ReLU
from keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import class_weight
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import model as model_process
from modules import custom_model_fit_indicators as custom_model_fit_indicators
from keras.callbacks import ReduceLROnPlateau
from sklearn.base import BaseEstimator, ClassifierMixin

def create_meta_model_data(models, X):
    meta_X = []
    for model in models:
        # 使用每个基础模型的预测作为特征
        predictions = model.predict(X)
        meta_X.append(predictions)
    # 将所有模型的预测堆叠成新的特征集
    meta_X = np.hstack(meta_X)
    return meta_X

def soft_voting_predict(models, X):
    final_prediction = np.zeros((len(X), 5))  # 假设有5个类别
    for model in models:
        predictions = model.predict(X)
        final_prediction += predictions
    final_prediction /= len(models)
    return np.argmax(final_prediction, axis=1)

def load_model(dir_name):
    # 獲取當前文件的絕對路徑
    current_path = os.path.abspath(os.path.dirname(__file__))
    parent_path = os.path.join(current_path, '..')
    trained_model_name = f'{parent_path}/trained_models/split/{dir_name}'
    model = tf.keras.models.load_model(trained_model_name, custom_objects={'multiclass_f1_score': multiclass_f1_score})

    return model

def multiclass_f1_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_f1_score(y_true, y_pred)

# 特徵欄位
# cols = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "Signal", "Hist", "SMA_27", "SMA_55", "SMA_200", "kline_color", "Middle Band", "Upper Band", "Lower Band"]
# cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'RSI_6', 'RSI_12', 'RSI_24', 'MACD', 'Signal', 'Hist', 'MACD_Stronger', 'MACD_Cross', 'Up Trend', 'Down Trend', 'Super Trend', 'ATR', 'SMA_12', 'SMA_21', 'SMA_27', 'SMA_55', 'SMA_200', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_12', 'EMA_26', 'EMA_50']
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'RSI_12', 'RSI_24', 'MACD', 'Signal', 'Hist', 'Up Trend', 'Down Trend', 'Super Trend', 'ATR', 'SMA_21', 'SMA_27', 'SMA_55', 'SMA_200', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_26', 'EMA_50']
symbol = "ETHUSDT"
interval = "5m"
look_back = 144 #使用回看n根數據
epochs = 200
batch_size = 128

# 當前
# end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# end_time_seconds = end_time / 1000
# end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
# end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# 特定
end_time_string = "2023-10-31 23:59:59"

# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 150000, cols, is_need_save_original_data=False, is_read_local=True, is_need_calculated=True)
# 再拿掉前後無參考性資料
df = df[1000:-300]
df.reset_index(drop=True, inplace=True)
df.to_csv('test.csv')

new_data_end_time_string = "2023-11-30 23:59:59"
df_new = modules_data.get_binance_klines_backward(symbol, interval, new_data_end_time_string, 9000, cols, is_need_save_original_data=False, is_read_local=False, is_need_calculated=True) 
df_new = df_new[1000:-300]
df_new.reset_index(drop=True, inplace=True)

# 使用MinMaxScaler對數據進行縮放
scaler = RobustScaler()
# 保持一致性的縮放
scaler.fit(df[cols])

scaled_data = scaler.fit_transform(df[cols])
scaled_new_data= scaler.fit_transform(df_new[cols])

df = pd.concat([df, pd.get_dummies(df['Target'], prefix='Target')], axis=1)
# df_new = pd.concat([df_new, pd.get_dummies(df_new['Target'], prefix='Target')], axis=1)

# 創建X, y數據集
X, y = [], []
for i in range(look_back, len(scaled_data) + 1):
    X.append(scaled_data[i - look_back:i])
    y.append(df.iloc[i - 1][['Target_0', 'Target_1', 'Target_2', 'Target_3', 'Target_4']].values)

y = [element.astype(int) for element in y]
X, y = np.array(X), np.array(y)

X_new, y_new = [], []
for new_index in range(look_back, len(scaled_new_data) + 1):
    X_new.append(scaled_new_data[new_index - look_back:new_index])
    y_new.append(df_new.iloc[new_index - 1][['Target_0', 'Target_1', 'Target_2', 'Target_3', 'Target_4']].values)
y_new = [element.astype(int) for element in y_new]
X_new, y_new = np.array(X_new), np.array(y_new)

model1 = load_model('1702627511_softmax_loss-1.6495_accuracy-0.2448_f1-0.0796')
model2 = load_model('1702628135_softmax_loss-1.5842_accuracy-0.2608_f1-0.0781')
model3 = load_model('1702629053_softmax_loss-1.6111_accuracy-0.2689_f1-0.0825')
model4 = load_model('1702630843_softmax_loss-1.7237_accuracy-0.2575_f1-0.0885')
model5 = load_model('1702632574_softmax_loss-1.6728_accuracy-0.2703_f1-0.0873')

models = [model1, model2, model3, model4, model5]
meta_X_train = create_meta_model_data(models, X)
meta_X_test = create_meta_model_data(models, X_new)


# 定义一个简单的神经网络作为元模型
meta_model = Sequential()

meta_model.add(Dense(32, input_dim=meta_X_train.shape[1]))
meta_model.add(BatchNormalization())
meta_model.add(PReLU())
meta_model.add(Dropout(0.5))

meta_model.add(Dense(16))
meta_model.add(BatchNormalization())
meta_model.add(ReLU())
meta_model.add(Dropout(0.5))

meta_model.add(Dense(5, activation='softmax'))  # 假设有5个类别

# 1. 从one-hot编码中还原y到单一标签格式
y_single_label = np.argmax(y, axis=1)
# 2. 使用compute_class_weight计算权重
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_single_label), y=y_single_label)
# 3. 创建一个与类标签匹配的权重字典
class_weights_dict = {i: weights[i] for i in range(len(weights))}

meta_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', multiclass_f1_score])

# 训练元模型
history = meta_model.fit(meta_X_train, y, class_weight=class_weights_dict, epochs=50, batch_size=64, validation_data=(meta_X_test, y_new))

# 評估模型
loss, accuracy, f1 = meta_model.evaluate(meta_X_test, y_new)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1: {f1:.4f}")

# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))

# 定義上層目錄的路徑
parent_path = os.path.join(current_path, '..')
time = int(datetime.datetime.timestamp(datetime.datetime.now()))
trained_model_name = f'{parent_path}/trained_models/{time}_softmax_loss-{loss:.4f}_accuracy-{accuracy:.4f}_f1-{f1:.4f}'
model_process.save_model(trained_model_name, meta_model, loss, accuracy)
model_process.export_epoch_info(history, trained_model_name)
model_process.export_cm(trained_model_name, meta_model, meta_X_test, y_new, [0, 1, 2, 3, 4])
model_process.export_layer_parameters(trained_model_name, meta_model, cols, symbol, interval, look_back, df, batch_size, epochs, end_time_string)


# soft_voting_predictions = soft_voting_predict(models, X_new)
# y_new_single_label = np.argmax(y_new, axis=1)
# # 計算準確率
# accuracy = accuracy_score(y_new_single_label, soft_voting_predictions)
# print(f"堆叠模型的準確率: {accuracy}")
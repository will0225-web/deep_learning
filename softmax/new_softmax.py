import sys
import os

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam

from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, PReLU, Conv1D, MaxPooling1D, Flatten, LeakyReLU, ReLU, Bidirectional, Attention, LayerNormalization, Input, Activation, RepeatVector, Permute, Multiply, Layer
from tensorflow.keras.initializers import GlorotUniform

from keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import model as model_process
from modules import custom_model_fit_indicators as custom_model_fit_indicators
from keras.callbacks import ReduceLROnPlateau, TensorBoard

from tensorflow.keras.utils import to_categorical


# 自定义注意力层
class AttentionLayer(Layer):
    def __init__(self, return_sequences=True, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # 对输入进行变换
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        if self.return_sequences:
            return output  # 返回3D张量
        else:
            return tf.keras.backend.sum(output, axis=1)  # 返回2D张量

def multiclass_f1_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_f1_score(y_true, y_pred)


# 特徵欄位
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'MACD', 'Signal', 'Hist', 'MACD_Long_Short', 'MACD_Stronger', 'MACD_Cross', 'Up Trend', 'Down Trend', 'Super Trend', 'ATR', 'TR', 'SMA_55', 'SMA_200', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_26', 'EMA_50', 'green_three_soldiers', 'red_three_soldiers', 'rsi_overbought', 'rsi_oversold', 'macd_hist_stronger_continuity', 'macd_hist_weaker_continuity']
symbol = "ETHUSDT"
interval = "15m"
look_back = 96 #使用回看n根數據
epochs = 100
batch_size = 128

# selected_cols = ['MACD_Long_Short', 'Hist', 'RSI', 'Volume', 'Close', 'Upper Band', 'Down Trend', 'Up Trend', 'MACD', 'Super Trend']
# selected_cols = ['Upper Band', 'Lower Band', 'Middle Band', 'Super Trend', 'Up Trend', 'Down Trend', 'MACD', 'Hist', 'RSI', 'ATR', 'MACD_Cross', 'MACD_Long_Short']
selected_cols = ['Open', 'Close', 'High', 'Low', 'TR', 'ATR', 'Volume']

# 當前
# end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# end_time_seconds = end_time / 1000
# end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
# end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# 特定
end_time_string = "2023-11-30 23:59:59"

# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 100000, cols, is_need_save_original_data=False, is_read_local=True, is_need_calculated=True)

# 预处理数据
X = df[cols]
y = df['Target']

X = X[selected_cols]

indicator_look_back = 282
price_look_back = 28261
X = indicators.add_lower_low_higher_high(X, 0.06, look_back=indicator_look_back)
X = indicators.add_price_indicator(X, look_back=price_look_back)


# 再拿掉前後無參考性資料
X = X[(29904 + 1600):-300]
X.reset_index(drop=True, inplace=True)
y = y[(29904 + 1600):-300]
y.reset_index(drop=True, inplace=True)

X['Target'] = y
X.to_csv('X_original.csv')
X.drop(['Target'], axis=1, inplace=True)

X.drop(['Close'], axis=1, inplace=True)
X.drop(['Open'], axis=1, inplace=True)
X.drop(['Low'], axis=1, inplace=True)
X.drop(['High'], axis=1, inplace=True)
X.drop(['min_low_look_back'], axis=1, inplace=True)


# 補齊所有欄位
cols = X.columns.to_list()

# 选择要缩放的列
# robust_features = ['Open', 'High', 'Low', 'Close', 'Up Trend', 'Down Trend', 'SMA_55', 'SMA_200', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_26', 'EMA_50', 'max_high_look_back', 'min_low_look_back', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'MACD', 'Signal', 'Hist']
# standard_features = ['Volume', 'ATR', 'TR']
# minMax_features = ['RSI']
# robust_features = ['Upper Band', 'Middle Band', 'Lower Band', 'Up Trend', 'Down Trend', 'MACD', 'Hist']
# standard_features = ['ATR']
# minMax_features = ['RSI']
robust_features = ['max_high_look_back', 'fibonacci_0.618', 'fibonacci_0.5', 'fibonacci_0.382']
standard_features = ['ATR', 'TR', 'Volume']
minMax_features = []



# # 列出每个缩放器/转换器对应的特征
preprocessor = ColumnTransformer(
    transformers=[
        ('price', RobustScaler(), robust_features),
        ('percent', StandardScaler(), standard_features),
        ('bounded', MinMaxScaler(feature_range=(0, 1)), minMax_features),
    ],
    remainder='passthrough'  # 不需要缩放的特征保持原样
)

# 对特征进行缩放
X_scaled = preprocessor.fit_transform(X)


# 為了檢查縮放是否有問題
# 将缩放后的数据转换为DataFrame
# X_scaled_df = pd.DataFrame(X_scaled, columns=cols)
# X_scaled_df = X_scaled_df[cols]
# X_scaled_df['Target'] = y
# X_scaled_df.to_csv('X_final.csv')

# 将目标变量转换为分类格式
y_categorical = to_categorical(y, num_classes=5)

# 使用look_back创建数据集
X, y = modules_data.create_look_back_dataset(X_scaled, y_categorical, look_back)

# 假设X_scaled是您的特征数据，y_categorical是您的目标数据
test_and_validation_size = 0.2
validation_ratio_of_test = 0.5  # 在测试和验证数据集中，验证集占的比例

# 首先分割出训练集和剩余集（测试集+验证集）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_and_validation_size, shuffle=False)
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_and_validation_size, stratify=y)

# 接着将剩余集分割为测试集和验证集
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_ratio_of_test, shuffle=False)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_ratio_of_test, stratify=y_temp)

# , kernel_regularizer=l1_l2(l1=0.0001, l2=0.005)
# , bias_regularizer=l1_l2(l1=0.0005, l2=0.002)
# 构建模型
model = Sequential()
# 卷积层
# model.add(Bidirectional(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
# model.add(LayerNormalization())
# model.add(Dropout(0.4))

model.add(LSTM(150, bias_regularizer=l1_l2(l1=0.0005, l2=0.001), return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LayerNormalization())
model.add(Dropout(0.4))

# 添加注意力层
# model.add(AttentionLayer())

model.add(LSTM(150, bias_regularizer=l1_l2(l1=0.0005, l2=0.001), return_sequences=False))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(150, bias_regularizer=l1_l2(l1=0.0005, l2=0.001)))
model.add(BatchNormalization())
# model.add(LeakyReLU())
model.add(ReLU())
model.add(Dropout(0.4))

model.add(Dense(150, bias_regularizer=l1_l2(l1=0.0005, l2=0.001)))
model.add(BatchNormalization())
# model.add(LeakyReLU())
model.add(ReLU())
model.add(Dropout(0.4))


model.add(Dense(5, activation='softmax')) # 5个类别

# optimizer = Adam(learning_rate=0.01, clipnorm=1.0)
# optimizer = Adam(learning_rate=0.0005, clipnorm=1)
optimizer = Adam(learning_rate=0.001)

# 學習率
# 初始化ReduceLROnPlateau回調
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # 監控驗證集的損失
                              factor=0.3,          # 學習率被減少的因子 (new_lr = lr * factor)
                              patience=5,         # 沒有進步的時期數，在這之後學習率會被減少
                              min_lr=0.00001,      # 學習率的下限
                              verbose=1,
                              mode='min')           # 信息展示模式

# 權重
y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)

# class_weights[0] *= 1
class_weights[1] *= 2
class_weights[2] *= 2
class_weights[3] *= 1.5
class_weights[4] *= 1.5

class_weight_dict = dict(enumerate(class_weights))

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', multiclass_f1_score])

# 训练模型
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min')

# 设置 TensorBoard 日志目录
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), class_weight=class_weight_dict, callbacks=[callback, reduce_lr, tensorboard_callback])

# 评估模型
loss, accuracy, f1_score = model.evaluate(X_test, y_test, batch_size=batch_size)
print(f"Test Accuracy: {accuracy}, F1 Score: {f1_score}")

# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))

# 定義上層目錄的路徑
parent_path = os.path.join(current_path, '..')
time = int(datetime.datetime.timestamp(datetime.datetime.now()))
trained_model_name = f'{parent_path}/trained_models/{time}_softmax_loss-{loss:.4f}_accuracy-{accuracy:.4f}_f1-{f1_score:.4f}'
model_process.save_model(trained_model_name, model, loss, accuracy)
model_process.export_epoch_info(history, trained_model_name)
model_process.export_cm(trained_model_name, model, X_test, y_test, [0, 1, 2, 3, 4], batch_size=batch_size)
model_process.export_layer_parameters(trained_model_name, model, cols, symbol, interval, look_back, df, batch_size, epochs, end_time_string, indicator_look_back)
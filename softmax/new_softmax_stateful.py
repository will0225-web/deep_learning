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
from keras.callbacks import ReduceLROnPlateau, TensorBoard, Callback

from tensorflow.keras.utils import to_categorical

class ResetStatesCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_states()

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
batch_size = 1

# selected_cols = ['MACD_Long_Short', 'Hist', 'RSI', 'Volume', 'Close', 'Upper Band', 'Down Trend', 'Up Trend', 'MACD', 'Super Trend']
selected_cols = ['Upper Band', 'Lower Band', 'Middle Band', 'Super Trend', 'Up Trend', 'Down Trend', 'MACD', 'Hist', 'RSI', 'ATR', 'MACD_Cross', 'MACD_Long_Short']

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
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 52560, cols, is_need_save_original_data=False, is_read_local=True, is_need_calculated=True)

# 预处理数据
X = df[cols]
y = df['Target']

# indicator_look_back = 672
# price_look_back = 1440
# X = indicators.add_lower_low_higher_high(X, 0.05, look_back=indicator_look_back)
# X = indicators.add_price_indicator(X, look_back=price_look_back)

# 再拿掉前後無參考性資料
X = X[1600:-300]
X.reset_index(drop=True, inplace=True)
y = y[1600:-300]
y.reset_index(drop=True, inplace=True)

X = X[selected_cols]

X['Target'] = y
X.to_csv('X_original.csv')
# X.drop(['Target'], axis=1, inplace=True)



# 補齊所有欄位
cols = X.columns.to_list()

# 选择要缩放的列
# robust_features = ['Open', 'High', 'Low', 'Close', 'Up Trend', 'Down Trend', 'SMA_55', 'SMA_200', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_26', 'EMA_50', 'max_high_look_back', 'min_low_look_back', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'MACD', 'Signal', 'Hist']
# standard_features = ['Volume', 'ATR', 'TR']
# minMax_features = ['RSI']
robust_features = ['Upper Band', 'Middle Band', 'Lower Band', 'Up Trend', 'Down Trend', 'MACD', 'Hist']
standard_features = ['ATR']
minMax_features = ['RSI']

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

X, y = modules_data.create_look_back_dataset(X_scaled, y_categorical, look_back)
# 假设X_scaled是您的特征数据，y_categorical是您的目标数据
test_and_validation_size = 0.3
validation_ratio_of_test = 0.5  # 在测试和验证数据集中，验证集占的比例

# 首先分割出训练集和剩余集（测试集+验证集）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_and_validation_size, shuffle=False)

# 接着将剩余集分割为测试集和验证集
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_ratio_of_test, shuffle=False)


# 计算可以整除128的最大整数部分
num_batches = len(X_train) // batch_size
# 重新计算训练集的长度，使其可以整除128
new_train_length = num_batches * batch_size
# 重新切割训练集
X_train = X_train[:new_train_length]
y_train = y_train[:new_train_length]

# 同样的方式处理验证集和测试集
num_batches_val = len(X_val) // batch_size
new_val_length = num_batches_val * batch_size
X_val = X_val[:new_val_length]
y_val = y_val[:new_val_length]

num_batches_test = len(X_test) // batch_size
new_test_length = num_batches_test * batch_size
X_test = X_test[:new_test_length]
y_test = y_test[:new_test_length]

model = Sequential()

model.add(LSTM(50, return_sequences=False, stateful=True, batch_input_shape=(batch_size, look_back, X_train.shape[2])))
model.add(Dropout(0.3))

# model.add(LSTM(50, return_sequences=True, stateful=True))
# model.add(Dropout(0.3))

# model.add(LSTM(50, return_sequences=False, stateful=True))
# model.add(Dropout(0.3))

# 第一个Dense层
# model.add(Dense(256))
# model.add(ReLU())
# model.add(Dropout(0.3))

# 第二个Dense层
model.add(Dense(128))
model.add(ReLU())
model.add(Dropout(0.3))

model.add(Dense(5, activation='softmax')) # 5个类别

optimizer = Adam(learning_rate=0.0001)

# 學習率
# 初始化ReduceLROnPlateau回調
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # 監控驗證集的損失
                              factor=0.3,          # 學習率被減少的因子 (new_lr = lr * factor)
                              patience=5,         # 沒有進步的時期數，在這之後學習率會被減少
                              min_lr=0.00001,      # 學習率的下限
                              verbose=1,
                              mode='min')           # 信息展示模式

# 训练模型
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min')

# 權重
y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weight_dict = dict(enumerate(class_weights))

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', multiclass_f1_score])

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), class_weight=class_weight_dict, callbacks=[callback, reduce_lr, ResetStatesCallback()], shuffle=False)


model.reset_states()  # 在评估前重置状态
loss, accuracy, f1_score = model.evaluate(X_test, y_test, batch_size=batch_size)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}, F1 Score: {f1_score}')
import sys
import os

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
from keras_tuner.tuners import BayesianOptimization
from keras_tuner import HyperModel, Objective

from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, PReLU, Conv1D, MaxPooling1D, Flatten, LeakyReLU, ReLU, Bidirectional, Attention, LayerNormalization, Input, Activation, RepeatVector, Permute, Multiply, Layer
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
from keras.callbacks import ReduceLROnPlateau

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
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'MACD', 'Signal', 'Hist', 'MACD_Long_Short', 'MACD_Stronger', 'MACD_Cross', 'Up Trend', 'Down Trend', 'Super Trend', 'ATR', 'TR', 'SMA_27', 'SMA_55', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_26', 'EMA_50', 'three_green_klines', 'three_red_klines', 'rsi_overbought', 'rsi_oversold', 'macd_hist_stronger_continuity', 'macd_hist_weaker_continuity']
symbol = "ETHUSDT"
interval = "15m"
look_back = 288 #使用回看n根數據
epochs = 100
batch_size = 128

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
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 40000, cols, is_need_save_original_data=False, is_read_local=True, is_need_calculated=True)

# 预处理数据
X = df[cols]
y = df['Target']


indicator_look_back = 1440
X = indicators.add_price_indicator(X, look_back=indicator_look_back)
X = indicators.add_lower_low_higher_high(X, look_back=indicator_look_back)
X = indicators.combine_lower_low_higher_high_and_kline(X, look_back=indicator_look_back)



# 再拿掉前後無參考性資料
X = X[1600:-300]
X.reset_index(drop=True, inplace=True)
y = y[1600:-300]
y.reset_index(drop=True, inplace=True)

# 補齊所有欄位
cols = X.columns.to_list()

# 选择要缩放的列
# robust_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'MACD', 'Signal', 'Hist', 'Up Trend', 'Down Trend', 'SMA_27', 'SMA_55', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_26', 'EMA_50']
robust_features = ['Open', 'High', 'Low', 'Close',  'Up Trend', 'Down Trend', 'SMA_27', 'SMA_55', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_26', 'EMA_50', 'max_high_look_back', 'min_low_look_back', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618']
standard_features = ['MACD', 'Signal', 'Hist', 'Volume', 'ATR', 'TR', 'macd_hist_stronger_continuity', 'macd_hist_weaker_continuity']
minMax_features = []

# 列出每个缩放器/转换器对应的特征
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

# 将目标变量转换为分类格式
y_categorical = to_categorical(y, num_classes=5)

# 使用look_back创建数据集
X, y = modules_data.create_look_back_dataset(X_scaled, y_categorical, look_back)

# 假设X_scaled是您的特征数据，y_categorical是您的目标数据
train_size = 0.7
test_and_validation_size = 0.3
validation_ratio_of_test = 0.5  # 在测试和验证数据集中，验证集占的比例

# 首先分割出训练集和剩余集（测试集+验证集）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_and_validation_size, shuffle=False)

# 接着将剩余集分割为测试集和验证集
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_ratio_of_test, shuffle=False)


def build_model(hp):
    model = Sequential()
    # 神经元数量的基准值
    num_conv_layers = hp.Int('conv_layers', 0, 2)
    num_lstm_layers = hp.Int('num_lstm_layers', 1, 3)
    num_dense_layer = hp.Int('dense_layers', 1, 2)

    base_conv_filters = hp.Int('base_conv_filters', min_value=16, max_value=128, step=16)
    base_conv_kernel_size = hp.Int('base_conv_kernel_size', min_value=3, max_value=9, step=2)
    base_lstm_units = hp.Int('base_lstm_units', min_value=32, max_value=128, step=32)
    base_dense_units = hp.Int('base_lstm_units', min_value=64, max_value=256, step=64)

    # 确定是保持不变、递增还是递减
    neuron_conv_filter_adjustment = hp.Choice('neuron_conv_filter_adjustment', ['fixed', 'increasing', 'decreasing'])
    neuron_lstm_adjustment = hp.Choice('neuron_lstm_adjustment', ['fixed', 'increasing', 'decreasing'])
    neuron_dense_adjustment = hp.Choice('neuron_dense_adjustment', ['fixed', 'increasing', 'decreasing'])

    conv_dropout = hp.Float('conv_dropout', min_value=0.2, max_value=0.6, step=0.1)
    lstm_dropout = hp.Float('lstm_dropout', min_value=0.2, max_value=0.6, step=0.1)
    dense_dropout = hp.Float('dense_dropout', min_value=0.2, max_value=0.6, step=0.1)

    conv_need_norm = hp.Boolean('conv_norm')
    lstm_need_norm = hp.Boolean('lstm_need_norm')
    dense_need_norm = hp.Boolean('dense_need_norm')

    # 卷积层
    filters = base_conv_filters
    for i in range(num_conv_layers):
        if i == 0:
            model.add(Conv1D(
                filters=filters,
                kernel_size=base_conv_kernel_size
            ))
        else:
            model.add(Conv1D(
                filters=filters,
                kernel_size=base_conv_kernel_size,
                input_shape=(X_train.shape[1], X_train.shape[2])
            ))
        if conv_need_norm:
            model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(conv_dropout))

        # 调整filter数量
        if neuron_conv_filter_adjustment == 'increasing':
            filters *= 2
        elif neuron_conv_filter_adjustment == 'decreasing':
            filters = max(16, filters // 2)

    if num_conv_layers != 0:
        model.add(MaxPooling1D(pool_size=2))

    # LSTM层
    units = base_lstm_units
    for i in range(num_lstm_layers):
        if num_conv_layers == 0 and i == 1:
            model.add(LSTM(
                units=units,
                return_sequences=True if i < num_lstm_layers - 1 else False,
                kernel_regularizer=l1_l2(l1=0.000001, l2=0.00005),
                input_shape=(X_train.shape[1], X_train.shape[2])
            )) 
        else:
            model.add(LSTM(
                units=units,
                return_sequences=True if i < num_lstm_layers - 1 else False,
                kernel_regularizer=l1_l2(l1=0.000001, l2=0.00005)
            )) 
        if lstm_need_norm:
            if num_lstm_layers - 1 == i:
                model.add(BatchNormalization())
            else:
                model.add(LayerNormalization())
        model.add(Dropout(lstm_dropout))

        if num_lstm_layers - 2 == i:
            model.add(AttentionLayer())

        # 调整units数量
        if neuron_lstm_adjustment == 'increasing':
            units = min(512, units * 2)
        elif neuron_lstm_adjustment == 'decreasing':
            units = max(32, units // 2)

    # 密集层
    dense_units = base_dense_units
    for i in range(num_dense_layer):
        model.add(Dense(
            units=dense_units,
        ))
        if dense_need_norm:
            model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(dense_dropout))

        # 调整units数量
        if neuron_dense_adjustment == 'increasing':
            dense_units *= 2
        elif neuron_dense_adjustment == 'decreasing':
            dense_units = max(32, dense_units // 2)

    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', multiclass_f1_score])

    return model

# 实例化贝叶斯优化器
tuner = BayesianOptimization(
    build_model,
    objective=Objective('val_multiclass_f1_score', direction='max'),
    max_trials=30,
    directory='new_softmax_macd',
    project_name='new_softmax_macd_project'
)

# 初始化ReduceLROnPlateau回調
reduce_lr = ReduceLROnPlateau(monitor='val_multiclass_f1_score',  # 監控驗證集的損失
                              factor=0.2,          # 學習率被減少的因子 (new_lr = lr * factor)
                              patience=3,         # 沒有進步的時期數，在這之後學習率會被減少
                              min_lr=0.00001,      # 學習率的下限
                              verbose=1,
                              mode='max')           # 信息展示模式

# 權重
y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weight_dict = dict(enumerate(class_weights))
# 执行搜索
callback = tf.keras.callbacks.EarlyStopping(monitor='val_multiclass_f1_score', patience=12, restore_best_weights=True, mode='max')
tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), class_weight=class_weight_dict, callbacks=[callback, reduce_lr])

# 获取最佳模型
best_model = tuner.get_best_models(num_models=1)[0]

# 评估最佳模型
loss, accuracy, f1_score = best_model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Test Accuracy: {accuracy}, F1 Score: {f1_score}")

# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))

# 定義上層目錄的路徑
parent_path = os.path.join(current_path, '..')
time = int(datetime.datetime.timestamp(datetime.datetime.now()))
trained_model_name = f'{parent_path}/trained_models/{time}_softmax_loss-{loss:.4f}_accuracy-{accuracy:.4f}_f1-{f1_score:.4f}'
model_process.save_model(trained_model_name, best_model, loss, accuracy)
# model_process.export_epoch_info(history, trained_model_name)
model_process.export_cm(trained_model_name, best_model, X_test, y_test, [0, 1, 2, 3, 4], batch_size=batch_size)
model_process.export_layer_parameters(trained_model_name, best_model, cols, symbol, interval, look_back, df, batch_size, epochs, end_time_string, indicator_look_back)
import sys
import os
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import datetime
import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Activation, concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import model as model_process
from modules import utilities as utilities
from modules import custom_model_fit_indicators as custom_model_fit_indicators
from keras.callbacks import ReduceLROnPlateau


def multiclass_f1_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_f1_score(y_true, y_pred)

def set_tops_and_bottoms(df, tops, bottoms, interval):
    tops_df = pd.DataFrame(tops, columns=['index', 'price', 'datetime'])
    bottoms_df = pd.DataFrame(bottoms, columns=['index', 'price', 'datetime'])

    df['datetime'] = pd.to_datetime(df['datetime'])
    tops_df['datetime'] = pd.to_datetime(tops_df['datetime'])
    bottoms_df['datetime'] = pd.to_datetime(bottoms_df['datetime'])

    tops = df.apply(get_past_tops_and_bottoms, args=(tops_df, interval), axis=1)
    bottoms = df.apply(get_past_tops_and_bottoms, args=(bottoms_df, interval), axis=1)

    return tops, bottoms

def get_past_tops_and_bottoms(row, tops_or_bottoms, interval):
    start_date = row['datetime'] - pd.Timedelta(days=interval)
    past_days_tops_or_bottoms = tops_or_bottoms[(tops_or_bottoms['datetime'] >= start_date) & (tops_or_bottoms['datetime'] < row['datetime'])]

    return past_days_tops_or_bottoms['price'].tolist()

def transform_and_data_info(df, y, cols, robust_features, standard_features, minMax_features, front_drop_count, back_drop_count, look_back, original_X=[]):
    data = df[cols + ['Significant Kline']]
    print(f'data: {data}')

    # 列出每個縮放器/轉換器對應的特徵
    preprocessor = ColumnTransformer(
        transformers=[
            ('price', RobustScaler(), robust_features),
            ('percent', StandardScaler(), standard_features),
            ('bounded', MinMaxScaler(feature_range=(0, 1)), minMax_features),
        ],
        remainder='passthrough'  # 不需要縮放的特徵保持原樣
    )
    # 對特徵進行縮放
    if len(original_X) == 0:
        scaled = preprocessor.fit_transform(data)
    else:
        preprocessor.fit(original_X[cols])
        # 對特徵進行縮放
        scaled = preprocessor.transform(data)

    if look_back != 0:
        scaled, y = create_look_back_dataset(scaled, y, look_back)
        scaled = scaled[:, :, :-1]

    if front_drop_count == 0 and back_drop_count == 0:
        scaled = scaled
        y = y
    elif front_drop_count != 0:
        scaled = scaled[front_drop_count:]
        y = y[front_drop_count:]
    elif back_drop_count == 0:
        scaled = scaled[:back_drop_count]
        y = y[:back_drop_count]

    if look_back == 0:
        data_X = []
        data_y = []
        for i in range(len(scaled)):
            if scaled[i][-1] == 1:
                data_X.append(scaled[i])
                data_y.append(y[i])
        data_X = np.array(data_X)
        data_y = np.array(data_y)
        scaled = data_X[:, :-1]
        y = data_y

    # 加上 customized cols 資訊
    transform_data_info = {
        'cols': cols,
        'robust_features': robust_features,
        'standard_features': standard_features,
        'minMax_features': minMax_features,
        'front_drop_count': front_drop_count,
        'back_drop_count': back_drop_count,
        'look_back': look_back
    }
    return scaled, y, transform_data_info

def create_look_back_dataset(X, y, look_back=1):
    dataX, dataY = [], []
    for i in range(1, len(X) - look_back + 1):
        # dataX.append(X[i: (i + look_back)])
        # dataY.append(y[i + look_back - 1])
        sequence_X = X[i: (i + look_back)]
        sequence_y = y[i + look_back - 1]

        if sequence_X[-1, -1] == 1:
            dataX.append(sequence_X)
            dataY.append(sequence_y)

    return np.array(dataX), np.array(dataY)

@utilities.capture_args
def price_target(df, long_tp=0.015, short_tp=0.015, long_sl=0.0075, short_sl=0.0075, hold_count_limit=16):
    # 2 做多贏
    # 0 做空贏
    # 1 盤整無波動

    price_series = df['Close']
    low_series = df['Low']
    high_series = df['High']

    cases = []

    for idx in range(len(price_series) - 1):
        current_price = price_series[idx]

        hit_long_tp = False
        hit_short_tp = False

        case = 1

        check_end_index = min(len(price_series), idx + hold_count_limit + 1)
        for future_idx in range(idx + 1, check_end_index):
            future_price = price_series[future_idx]
            future_low = low_series[future_idx]
            future_high = high_series[future_idx]

            high_change_pct = indicators.calculate_percentage_change(future_high, current_price)
            low_change_pct = indicators.calculate_percentage_change(future_low, current_price)

            if high_change_pct >= long_tp:
                case = 2
                hit_long_tp = True
                break
            elif low_change_pct <= -short_tp:
                case = 0
                hit_short_tp = True
                break

        if not hit_long_tp and not hit_short_tp:
            case = 1

        cases.append(case)

    cases.append(1)  # 最後一個數據沒有未來的數據可以比較，所以直接設定為0
    return cases

@utilities.capture_args
def price_target_binary(df, long_tp=None, short_tp=None, hold_count_limit=12):
    # 1 有漲/跌到X%
    # 0 沒有漲/跌到X%

    if long_tp is None and short_tp is None:
        raise ValueError('long_tp and short_tp cannot be None at the same time')

    price_series = df['Close']
    low_series = df['Low']
    high_series = df['High']

    cases = []

    for idx in range(len(price_series) - 1):
        current_price = price_series[idx]

        case = 0

        check_end_index = min(len(price_series), idx + hold_count_limit + 1)
        for future_idx in range(idx + 1, check_end_index):
            future_price = price_series[future_idx]
            future_low = low_series[future_idx]
            future_high = high_series[future_idx]


            if long_tp is not None:
                high_change_pct = indicators.calculate_percentage_change(future_high, current_price)
                if high_change_pct >= long_tp:
                    case = 1
                    break
            elif short_tp is not None:
                low_change_pct = indicators.calculate_percentage_change(future_low, current_price)
                if -low_change_pct >= short_tp:
                    case = 1
                    break

        cases.append(case)

    cases.append(0)  # 最後一個數據沒有未來的數據可以比較，所以直接設定為0
    return cases

def customized_specific_period_col(df):
    customized_cols_infos = []
    
    # 紀錄原先的cols
    original_cols = set(df.columns)
    
    df, price_indicator_info = indicators.trend_min_max_price_indicator(df, look_back=192, is_need_return_function_info=True)
    customized_cols_infos.append(price_indicator_info)

    df, supertrend_delta_info = indicators.super_trend_delta_and_risk_v1(df, is_need_return_function_info=True)
    customized_cols_infos.append(supertrend_delta_info)

    # df['EMA_7'], ema_7_info = indicators.calculate_ema(df['Close'], 7, is_need_return_function_info=True)
    # customized_cols_infos.append(ema_7_info)

    # df['EMA_25'], ema_25_info = indicators.calculate_ema(df['Close'], 25, is_need_return_function_info=True)
    # customized_cols_infos.append(ema_25_info)

    # (tops, bottoms), tops_and_downs_info = indicators.tops_and_downs(df, 6, is_need_return_function_info=True)
    # df['Tops'], df['Bottoms'] = set_tops_and_bottoms(df, tops, bottoms, 7)
    # customized_cols_infos.append(tops_and_downs_info)

    # Only klines with significant volume will be used as input data
    # Set threshold as 0 to include all klines
    df['Significant Kline'] = df['Volume'].apply(lambda x: 1 if x >= 10000 else 0)
    siginificant_kline_counts = df['Significant Kline'].value_counts()
    print(siginificant_kline_counts)

    # 紀錄新的cols
    modified_cols = set(df.columns)
    # 篩選多出來的cols
    new_cols = list(modified_cols - original_cols)

    print(f'Full df: {df}')

    return df, customized_cols_infos, new_cols


symbol = "ETHUSDT"
interval = "15m"
look_back = 144 #使用回看n根數據
epochs = 300
batch_size = 128
total_klines = 100000
get_local_file_name = 'ETHUSDT_15m_2023-12-31_23-59-59_150000_calculated.csv'
# get_local_file_name = 'ETHUSDT_5m_2023-12-31_23-59-59_500000_calculated.csv'
input_model_infos = []


# 當前
# end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# end_time_seconds = end_time / 1000
# end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
# end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# 特定
end_time_string = "2023-12-31 23:59:59"


# Step 1: 獲取數據
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
df, customized_cols_infos, new_cols = customized_specific_period_col(df)
df['Target'], target_function_info = price_target(df, is_need_return_function_info=True)

df.to_csv('./local_data/v3_calculated_data.csv')

target_counts = df['Target'].value_counts()
print(target_counts)

# 拿掉前後無參考性資料
drop_front_data_count = 1000
drop_back_data_count = 300
df = df[drop_front_data_count:-drop_back_data_count]
df.reset_index(drop=True, inplace=True)

# 'Target'列不重複的數值當作有的target
target_types = sorted(df['Target'].unique().tolist())

# print(target_types)
X = df
y = df['Target']
# 将目标变量转换为分类格式

y_categorical = to_categorical(y, num_classes=len(target_types))
# y_categorical = y

# X['Target'] = y
# X.to_csv('X_original.csv')
# X.drop(['Target'], axis=1, inplace=True)

# 價格的狀態
ochl_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'Hist', 'RSI', 'Upper Band', 'Lower Band', 'Middle Band', 'SMA_200', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618']
ochl_robust_features = []
ochl_standard_features = ['Close', 'High', 'Low', 'Open', 'Volume', 'Hist', 'RSI', 'Upper Band', 'Lower Band', 'Middle Band', 'SMA_200', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618']
ochl_minMax_features = []
X_ochl_scaled, X_ochl_y, ochl_data_info = transform_and_data_info(X, y_categorical, ochl_cols, ochl_robust_features, ochl_standard_features, ochl_minMax_features, 0, 0, look_back)
input_model_infos.append(ochl_data_info)

# # Support and Resistance
# sr_cols = ['fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'Close', 'High', 'Low']
# sr_robust_features = []
# sr_standard_features = ['fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'Close', 'High', 'Low']
# sr_minMax_features = []
# # 這個要把look_back取走才會跟接下來X_ochl搭到, 因為X_ochl會從look_back的index開始抓資料
# X_sr_scaled, X_sr_y, sr_data_info = transform_and_data_info(X, y_categorical, sr_cols, sr_robust_features, sr_standard_features, sr_minMax_features, look_back, 0, 0)
# input_model_infos.append(sr_data_info)

# # Strategy
# strategy_cols = ['Take Profit', 'Stop Loss', 'Look Ahead']
# strategy_robust_features = []
# strategy_standard_features = ['Take Profit', 'Stop Loss', 'Look Ahead']
# strategy_minMax_features = []
# X_strategy_scaled, X_strategy_y, strategy_data_info = transform_and_data_info(X, y_categorical, strategy_cols, strategy_robust_features, strategy_standard_features, strategy_minMax_features, 0, 0, look_back)
# input_model_infos.append(strategy_data_info)

# 假设X_scaled是您的特征数据，y_categorical是您的目标数据
test_and_validation_size = 0.3
validation_ratio_of_test = 0.5  # 在测试和验证数据集中，验证集占的比例

# 首先分割出训练集和剩余集（测试集+验证集）
X_ochl_train, X_ochl_temp, y_ochl_train, y_ochl_temp = train_test_split(X_ochl_scaled, X_ochl_y, test_size=test_and_validation_size, shuffle=False)
# X_sr_train, X_sr_temp, y_sr_train, y_sr_temp = train_test_split(X_sr_scaled, X_sr_y, test_size=test_and_validation_size, shuffle=False)
# X_strategy_train, X_strategy_temp, y_strategy_train, y_strategy_temp = train_test_split(X_strategy_scaled, X_strategy_y, test_size=test_and_validation_size, shuffle=False)


# 接着将剩余集分割为测试集和验证集
X_ochl_val, X_ochl_test, y_ochl_val, y_ochl_test = train_test_split(X_ochl_temp, y_ochl_temp, test_size=validation_ratio_of_test, shuffle=False)
# X_sr_val, X_sr_test, y_sr_val, y_sr_test = train_test_split(X_sr_temp, y_sr_temp, test_size=validation_ratio_of_test, shuffle=False)
# X_strategy_val, X_strategy_test, y_strategy_val, y_strategy_test = train_test_split(X_strategy_temp, y_strategy_temp, test_size=validation_ratio_of_test, shuffle=False)

# , kernel_regularizer=l1_l2(l1=0.0001, l2=0.005)
# , bias_regularizer=l1_l2(l1=0.0005, l2=0.002)
# 构建模型

# 定义输入层

# 單純學習價格的上漲或者下跌的input
ochl_input = Input(shape=(look_back, X_ochl_train.shape[2]), name='ochl_input')
# sr_input = Input(shape=(X_sr_train.shape[1],), name='sr_input')
# strategy_input = Input(shape=(look_back, X_strategy_train.shape[2]), name='strategy_input')

# 建立一個ochl_input單純學習價格上漲或者下跌的layer，幫助最後的softmax學習的程式
# LSTM层，处理时间序列数据
ochl_lstm = LSTM(128, return_sequences=True)(ochl_input)
ochl_dropout = Dropout(0.3)(ochl_lstm)
ochl_lstm2 = LSTM(64, return_sequences=False)(ochl_dropout)
ochl_dropout2 = Dropout(0.3)(ochl_lstm2)

# 内部趋势学习层
# trend_layer = Dense(64, activation='relu')(dropout_layer)  # 用于学习趋势的中间层
# trend_dropout = Dropout(0.5)(trend_layer)  # 防止过拟合

merged_dense = Dense(16)(ochl_dropout2)
bn1 = BatchNormalization()(merged_dense)
merged_dense = Activation('relu')(bn1)
merged_dense = Dropout(0.3)(merged_dense)
output_layer = Dense(y_ochl_train.shape[1], activation='softmax')(merged_dense)

model = Model(inputs=[ochl_input], outputs=output_layer)

# CNN层，用于提取时间序列中的局部特征
# conv1 = Conv1D(filters=256, kernel_size=3, activation='relu')(ochl_input)
# pool1 = MaxPooling1D(pool_size=2)(conv1)
# conv2 = Conv1D(filters=128, kernel_size=3)(pool1)
# bn1 = BatchNormalization()(conv2)
# act1 = Activation('relu')(bn1)
# pool2 = MaxPooling1D(pool_size=2)(act1)
# pattern_out = Flatten()(pool2)
# pattern_out = Dense(256)(pattern_out)
# bn2 = BatchNormalization()(pattern_out)
# pattern_out = Activation('relu')(bn2)
# pattern_out = Dropout(0.3)(pattern_out)


# 处理更多指标的LSTM层，带注意力机制
# bi_lstm1 = Bidirectional(LSTM(512, return_sequences=True))(ochl_lstm_input)
# bi_lstm1_ln = LayerNormalization()(bi_lstm1)
# bi_lstm1_dropout = Dropout(0.5)(bi_lstm1_ln)
# # attention_layer = Attention()([bi_lstm1_dropout, bi_lstm1_dropout])
# # attention_output = concatenate([bi_lstm1_dropout, attention_layer])
# bi_lstm2 = Bidirectional(LSTM(256, return_sequences=True))(bi_lstm1_dropout)
# bi_lstm2_ln = LayerNormalization()(bi_lstm2)
# bi_lstm2_dropout = Dropout(0.3)(bi_lstm2_ln)
# bi_lstm3 = Bidirectional(LSTM(128, return_sequences=False))(bi_lstm2_dropout)
# bi_lstm3_ln = LayerNormalization()(bi_lstm3)
# lstm_out = Dropout(0.3)(bi_lstm3_ln)
# lstm_layer = Dense(64)(lstm_out)
# bn3 = BatchNormalization()(lstm_layer)
# lstm_out = Activation('relu')(bn3)
# lstm_out = Dropout(0.5)(lstm_out)

# sr_dense = Dense(256)(sr_input)
# sr_ln = BatchNormalization()(sr_dense)
# sr_relu = Activation('relu')(sr_ln)
# sr_dropout = Dropout(0.2)(sr_relu)
# sr_dense2 = Dense(128)(sr_dropout)
# sr_ln2 = BatchNormalization()(sr_dense2)
# sr_relu2 = Activation('relu')(sr_ln2)
# sr_dropout2 = Dropout(0.2)(sr_relu2)

# strategy_dense = Dense(256)(strategy_input)
# strategy_ln = BatchNormalization()(strategy_dense)
# strategy_relu = Activation('relu')(strategy_ln)
# strategy_dropout = Dropout(0.2)(strategy_relu)
# strategy_dense2 = Dense(128)(strategy_dropout)
# strategy_ln2 = BatchNormalization()(strategy_dense2)
# strategy_relu2 = Activation('relu')(strategy_ln2)
# strategy_dropout2 = Dropout(0.2)(strategy_relu2)

# # 合并两个输入处理路径的输出
# merged_layer = concatenate([ochl_dropout2, sr_dropout2])

# # 后续的Dense层
# merged_dense = Dense(512)(merged_layer)
# bn1 = BatchNormalization()(merged_dense)
# merged_dense = Activation('relu')(bn1)
# merged_dense = Dropout(0.2)(merged_dense)
# merged_dense = Dense(256)(merged_dense)
# bn2 = BatchNormalization()(merged_dense)
# merged_dense = Activation('relu')(bn2)
# merged_dense = Dropout(0.2)(merged_dense)
# output_layer = Dense(y_ochl_train.shape[1], activation='softmax')(merged_dense)

# 构建模型
# model = Model(inputs=[ochl_input, sr_input], outputs=output_layer)

optimizer = Adam(learning_rate=0.001)

# 權重
y_train_labels = np.argmax(y_ochl_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)

class_weight_dict = dict(enumerate(class_weights))

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', multiclass_f1_score])

# # 學習率
# # 初始化ReduceLROnPlateau回調
# reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # 監控驗證集的損失
#                               factor=0.5,          # 學習率被減少的因子 (new_lr = lr * factor)
#                               patience=3,          # 沒有進步的時期數，在這之後學習率會被減少
#                               min_lr=0.000001,     # 學習率的下限
#                               verbose=1,
#                               mode='min')          # 信息展示模式

# 训练模型
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min')

# 设置 TensorBoard 日志目录
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [callback]

history = model.fit(X_ochl_train, y_ochl_train, epochs=epochs, batch_size=batch_size, validation_data=(X_ochl_val, y_ochl_val), class_weight=class_weight_dict, callbacks=callbacks)

# 评估模型
evaluation_results = model.evaluate(X_ochl_test, y_ochl_test, batch_size=batch_size, return_dict=True)
loss = evaluation_results['loss']
accuracy = evaluation_results['accuracy']
f1_score = evaluation_results['multiclass_f1_score']
for key, value in evaluation_results.items():
    print(f"{key}, {value}")


# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))

# 定義上層目錄的路徑
parent_path = os.path.join(current_path, '..')
time = int(datetime.datetime.timestamp(datetime.datetime.now()))
trained_model_name = f'{parent_path}/trained_models/{time}_softmax_loss-{loss:.4f}_accuracy-{accuracy:.4f}_f1-{f1_score:.4f}'
model_process.save_model(trained_model_name, model, loss, accuracy)
model_process.export_epoch_info(history, trained_model_name)
model_process.export_cm(trained_model_name, model, X_ochl_test, y_ochl_test, target_types, threshold=0.5, batch_size=batch_size)
model_process.export_layer_parameters(trained_model_name, model, symbol, interval, look_back, df, batch_size, epochs, end_time_string, evaluation_results, test_and_validation_size, validation_ratio_of_test, class_weight_dict, get_local_file_name)
model_process.export_connection_info(trained_model_name, batch_size, symbol, interval, look_back, end_time_string, total_klines, target_function_info, input_model_infos, target_types, customized_cols_infos, drop_front_data_count, drop_back_data_count, get_local_file_name, new_cols)

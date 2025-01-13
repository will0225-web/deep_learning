import sys
import os
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import tensorflow as tf
import itertools, random
import csv
from datetime import datetime

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

def multiclass_f05_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_fbeta_score(y_true, y_pred, beta=0.5)

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
def set_y_label(df, lookahead=8, long_percentage=None, short_percentage=None):
    # set_y_label

    for i in range(len(df)):
        current_close = df['Close'].iloc[i]

        if long_percentage is not None:
            if i == len(df) - 1:
                df.at[i, 'y'] = 0
                df.at[i, 'up_percentage'] = current_close * (1 + long_percentage)
                continue
            window_end_index = min(i + lookahead + 1, len(df))
            future_high_window = df['High'].iloc[i+1:window_end_index]
            max_future_price = future_high_window.max()
            up_price = current_close * (1 + long_percentage)
            if max_future_price >= up_price:
                df.at[i, 'y'] = 1
            else:
                df.at[i, 'y'] = 0
            df.at[i, 'up_percentage'] = up_price

        elif short_percentage is not None:
            if i == len(df) - 1:
                df.at[i, 'y'] = 0
                df.at[i, 'down_percentage'] = current_close * (1 - short_percentage)
                continue
            window_end_index = min(i + lookahead + 1, len(df))
            future_low_window = df['Low'].iloc[i+1:window_end_index]
            min_future_price = future_low_window.min()
            down_price = current_close * (1 - short_percentage)
            if min_future_price <= down_price:
                df.at[i, 'y'] = 1
            else:
                df.at[i, 'y'] = 0
            df.at[i, 'down_percentage'] = down_price

        df['lookahead'] = lookahead

    return df

@utilities.capture_args
def set_y_win_lose_label_sequence(df, lookahead=8, tp_percentage=0.008, sl_percentage=0.004):
    for i in range(len(df) - lookahead):
        current_close = df['Close'].iloc[i]
        
        # 计算多头的止盈和止损价格
        long_tp_price = current_close * (1 + tp_percentage)
        long_sl_price = current_close * (1 - sl_percentage)
        
        # 计算空头的止盈和止损价格
        short_tp_price = current_close * (1 - tp_percentage)
        short_sl_price = current_close * (1 + sl_percentage)

        y_label = 0  # 默认为没有达到止盈或止损的情况
        long_hit = False  # 标记多头是否触发止盈或止损
        short_hit = False  # 标记空头是否触发止盈或止损

        for j in range(1, lookahead + 1):
            future_high = df['High'].iloc[i + j]
            future_low = df['Low'].iloc[i + j]

            # 检查多头是否触发止盈或止损
            if not long_hit:
                if future_low <= long_sl_price:
                    y_label = 3  # 多头止损（long lose）
                    long_hit = True  # 多头已经处理
                elif future_high >= long_tp_price:
                    y_label = 1  # 多头获胜（long win）
                    long_hit = True  # 多头已经处理
                    break

            # 检查空头是否触发止盈或止损
            if not short_hit:
                if future_high >= short_sl_price:
                    y_label = 4  # 空头止损（short lose）
                    short_hit = True  # 空头已经处理
                elif future_low <= short_tp_price:
                    y_label = 2  # 空头获胜（short win）
                    short_hit = True  # 空头已经处理
                    break
                
            # 如果多头和空头都触发了止损，则标记为0
            if long_hit and short_hit:
                y_label = 0  # 两边都碰到止损，标记为0
                break

        # 标记止盈和止损价格
        df.at[i, 'y'] = y_label
        df.at[i, 'long_tp_price'] = long_tp_price
        df.at[i, 'long_sl_price'] = long_sl_price
        df.at[i, 'short_tp_price'] = short_tp_price
        df.at[i, 'short_sl_price'] = short_sl_price

    df['lookahead'] = lookahead

    return df

@utilities.capture_args
def calculate_volatility(df, look_back):
    df['Close_Volatility'] = df['Close'].rolling(window=look_back).std()
    df['High_Volatility'] = df['High'].rolling(window=look_back).std()
    df['Low_Volatility'] = df['Low'].rolling(window=look_back).std()
    df['Open_Volatility'] = df['Open'].rolling(window=look_back).std()

    return df

@utilities.capture_args
def calculate_bollinger_bands(df):
    df['BB_Width'] = df['Upper Band'] - df['Lower Band']
    df['BB_Pos'] = (df['Close'] - df['Lower Band']) / (df['Upper Band'] - df['Lower Band'])

    return df

@utilities.capture_args
def calculate_VWAP(df, window=20):
    df['Typical_Price'] = (df['Close'] + df['High'] + df['Low']) / 3
    df['VP'] = df['Typical_Price'] * df['Volume']

    df['Cumulative_VP'] = df['VP'].rolling(window=window).sum()
    df['Cumulative_Volume'] = df['Volume'].rolling(window=window).sum()

    df['VWAP'] = df['Cumulative_VP'] / df['Cumulative_Volume']

    return df

def customized_specific_period_col(df):
    customized_cols_infos = []

    # 紀錄原先的cols
    original_cols = set(df.columns)

    df, price_indicator_info = indicators.trend_min_max_price_indicator(df, look_back=192, is_need_return_function_info=True)
    customized_cols_infos.append(price_indicator_info)

    df, supertrend_delta_info = indicators.super_trend_delta_and_risk_v1(df, is_need_return_function_info=True)
    customized_cols_infos.append(supertrend_delta_info)

    df, bollinger_bands_info = calculate_bollinger_bands(df, is_need_return_function_info=True)
    customized_cols_infos.append(bollinger_bands_info)

    df['EMA_7'], ema_7_info = indicators.calculate_ema(df['Close'], 7, is_need_return_function_info=True)
    customized_cols_infos.append(ema_7_info)

    df['EMA_25'], ema_25_info = indicators.calculate_ema(df['Close'], 25, is_need_return_function_info=True)
    customized_cols_infos.append(ema_25_info)

    df['EMA_99'], ema_99_info = indicators.calculate_ema(df['Close'], 99, is_need_return_function_info=True)
    customized_cols_infos.append(ema_99_info)

    df, candlestick_patterns_info = indicators.calculate_candelstick_patterns(df, is_need_return_function_info=True)
    customized_cols_infos.append(candlestick_patterns_info)

    df, volatility_info = calculate_volatility(df, 20, is_need_return_function_info=True)
    customized_cols_infos.append(volatility_info)

    df, VWAP_info = calculate_VWAP(df, window=20, is_need_return_function_info=True)
    customized_cols_infos.append(VWAP_info)

    # Only klines with significant volume will be used as input data
    # Set threshold as 0 to include all klines
    df['Significant Kline'] = df['Volume'].apply(lambda x: 1 if x >= 0 else 0)
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
total_klines = 250000
get_local_file_name = 'ETHUSDT_15m_2024-09-27_15-30-41_300000_spot_full_calculated.csv'
# get_local_file_name = 'ETHUSDT_5m_2023-12-31_23-59-59_500000_calculated.csv'


# 當前
# end_time = int(datetime.timestamp(datetime.now())) * 1000

# end_time_seconds = end_time / 1000
# end_datetime = datetime.fromtimestamp(end_time_seconds)
# end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# 特定
end_time_string = "2023-12-31 23:59:59"


# Step 1: 獲取數據
# df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
# df, customized_cols_infos, new_cols = customized_specific_period_col(df)
df_original = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
df_original, customized_cols_infos, new_cols = customized_specific_period_col(df_original)

# df, target_function_info = set_y_label(df, lookahead=288, long_percentage=0.04, is_need_return_function_info=True)

# df.to_csv('./local_data/v3_calculated_data.csv')

# target_counts = df['y'].value_counts()
# print(target_counts)

# 拿掉前後無參考性資料
# drop_front_data_count = 1000
# drop_back_data_count = 300
# df = df[drop_front_data_count:-drop_back_data_count]
# df.reset_index(drop=True, inplace=True)

# # 'y'列不重複的數值當作有的target
# target_types = sorted(df['y'].unique().tolist())

# # print(target_types)
# X = df
# y = df['y']
# df.drop('y', axis=1, inplace=True)

# # 将目标变量转换为分类格式
# y_categorical = to_categorical(y, num_classes=len(target_types))


# Grid Search Parameters
lstm_layers_combinations = [1, 2, 2, 2, 3, 3, 3, 4]

lstm_units_max = 512
lstm_units_min = 2
l = 9

d = 7

learning_rates = [0.001, 0.0015, 0.002, 0.0025]
dropout_rates = [0.2, 0.3, 0.4]

# TODO: Add more grid search parameters
# batch_sizes = []
# look_backs = []
# y_label_lookaheads = []
# y_label_long_percentages = []


# Set LSTM units grid combinations
lstm_units = [lstm_units_max / (2 ** i) for i in range(l) if lstm_units_max / (2 ** i) >= lstm_units_min]
lstm_units = [int(i) for i in lstm_units]

# for i in range(1, 4):
#     lstm_products = []
#     lstm_products.extend(itertools.product(lstm_units, repeat=i))
#     print(f'Number of LSTM units combinations: {len(lstm_products)}')

# lstm_units_combinations = [(a, b) for a, b in itertools.product(lstm_units, repeat=2)]

dense_units = [2 ** i for i in range(1, d + 1)]

# lstm_units_combinations = [(1024, 4)]
# dense_units = [2]

dense_units_combinations = []
for r in range(1, 4):
    dense_units_combinations.extend(itertools.product(dense_units, repeat=r))
    # print(f'Number of Dense units combinations: {len(dense_units_combinations)}')

lookaheads = range(48, 288 + 48, 48)
long_percentages = np.arange(0.01, 0.035, 0.005)
long_percentages = np.round(long_percentages, 3)

# y_label_args = [(240, 0.02), (240, 0.03), (96, 0.01), (288, 0.015), (240, 0.01), (96, 0.025), (192, 0.03), (48, 0.03)]

time = int(datetime.timestamp(datetime.now()))
with open(f'{time}_training_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    # writer.writerow(['LSTM Units', 'Dense Units', 'Learning Rate', 'Dropout Rate', 'Loss', 'Accuracy', 'F0.5 Score'])
    writer.writerow(['Lookahead', 'Long Percentage', 'LSTM Units', 'Dense Units', 'Learning Rate', 'Dropout Rate', 'Loss', 'Accuracy', 'F0.5 Score'])

    for i in range(10000):
    # for lookahead, long_percentage in y_label_args:
        # lstm_layers_count = random.choice(lstm_layers_combinations)
        # lstm_units_combinations = list(itertools.product(lstm_units, repeat=lstm_layers_count))
        # lstm_combination = random.choice(lstm_units_combinations)

        # dense_combination = random.choice(dense_units_combinations)
        # learning_rate = random.choice(learning_rates)
        # dropout_rate = random.choice(dropout_rates)

        lookahead = random.choice(lookaheads)
        long_percentage = random.choice(long_percentages)

        print(f'Current Lookahead: {lookahead}')
        print(f'Current Long Percentage: {long_percentage}')

        lstm_combination = (64, 32)
        dense_combination = (128, 64)
        learning_rate = 0.001
        dropout_rate = 0.3

        # df_original = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
        df_original, target_function_info = set_y_label(df_original, lookahead=lookahead, long_percentage=long_percentage, is_need_return_function_info=True)

        target_counts = df_original['y'].value_counts()
        print(target_counts)

        df = df_original.copy()
        drop_front_data_count = 1000
        drop_back_data_count = 300
        df = df[drop_front_data_count:-drop_back_data_count]
        df.reset_index(drop=True, inplace=True)

        # 'y'列不重複的數值當作有的target
        target_types = sorted(df['y'].unique().tolist())

        X = df
        y = df['y']
        df.drop('y', axis=1, inplace=True)

        # 将目标变量转换为分类格式
        y_categorical = to_categorical(y, num_classes=len(target_types))

        print(f'Current LSTM combination: {lstm_combination}')
        print(f'Current Dense combination: {dense_combination}')
        print(f'Current Learning rate: {learning_rate}')
        print(f'Current Dropout rate: {dropout_rate}')

        input_model_infos = []

        # 價格的狀態
        ochl_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'fibonacci_1', 'fibonacci_0',
            'High_Low_pct', 'Up_Shadow_pct', 'Down_Shadow_pct', 'Open_Close_pct',
            'EMA_7', 'EMA_25', 'EMA_99', 
            'BB_Width', 'Upper Band', 'Lower Band', 'Middle Band',
            'Close_Volatility', 'ATR', 'VWAP', 'High_Volatility', 'Low_Volatility', 'Open_Volatility', 'Cumulative_VP', 'Cumulative_Volume',
            'up_percentage' # 'Current_Max_High', 'SMA_200', 'Delta_High_Down', 'Delta_Low_Up', 'Current_Max_Index', 'Current_Min_Index'
        ]
        # ochl_cols = [
        #     'Open', 'High', 'Low', 'Close', 'Volume', 'fibonacci_0.382', 
        #     'fibonacci_0.5', 'fibonacci_0.618', 'Open_Close_pct', 'High_Low_pct', 
        #     'Up_Shadow_pct', 'Down_Shadow_pct', 'EMA_7', 'EMA_25', 'EMA_99', 'BB_Width', 'Close_Volatility', 'RSI', 'ATR', 'VWAP', 'ADX', 'lookahead', 'TP'
        # ]
        ochl_robust_features = ['Volume', 'Open_Close_pct', 'High_Low_pct', 'Up_Shadow_pct', 'Down_Shadow_pct', 'ATR', 'BB_Width', 'Close_Volatility', 'High_Volatility', 'Low_Volatility', 'Open_Volatility', 'Cumulative_VP', 'Cumulative_Volume']
        ochl_standard_features = ['Close', 'High', 'Low', 'Open', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'fibonacci_1', 'fibonacci_0', 'EMA_7', 'EMA_25', 'EMA_99', 'Upper Band', 'Lower Band', 'Middle Band', 'VWAP', 'up_percentage']
        ochl_minMax_features = []
        X_ochl_scaled, X_ochl_y, ochl_data_info = transform_and_data_info(X, y_categorical, ochl_cols, ochl_robust_features, ochl_standard_features, ochl_minMax_features, 0, 0, look_back)
        input_model_infos.append(ochl_data_info)


        test_and_validation_size = 0.3
        validation_ratio_of_test = 0.5

        # 首先分割出訓練集和剩餘集(包含測試集和驗證集)
        X_ochl_train, X_ochl_temp, y_ochl_train, y_ochl_temp = train_test_split(X_ochl_scaled, X_ochl_y, test_size=test_and_validation_size, shuffle=False)

        # 接着將剩餘集分割出測試集和驗證集
        X_ochl_val, X_ochl_test, y_ochl_val, y_ochl_test = train_test_split(X_ochl_temp, y_ochl_temp, test_size=validation_ratio_of_test, shuffle=False)

        # , kernel_regularizer=l1_l2(l1=0.0001, l2=0.005)
        # , bias_regularizer=l1_l2(l1=0.0005, l2=0.002)
        # 构建模型

        # 定义输入层

        # 單純學習價格的上漲或者下跌的input
        ochl_input = Input(shape=(look_back, X_ochl_train.shape[2]), name='ochl_input')


        # 建立一個ochl_input單純學習價格上漲或者下跌的layer，幫助最後的softmax學習的程式
        # LSTM层，处理时间序列数据
        lstm_count = 1
        ochl_lstm = None
        ochl_dropout = None

        for lstm in lstm_combination:
            if len(lstm_combination) == 1:
                ochl_lstm = LSTM(lstm, return_sequences=False)(ochl_input)
                ochl_dropout = Dropout(dropout_rate)(ochl_lstm)
            elif lstm_count == 1:
                ochl_lstm = LSTM(lstm, return_sequences=True)(ochl_input)
                ochl_dropout = Dropout(dropout_rate)(ochl_lstm)
            elif lstm_count == len(lstm_combination):
                ochl_lstm = LSTM(lstm, return_sequences=False)(ochl_dropout)
                ochl_dropout = Dropout(dropout_rate)(ochl_lstm)
            else:
                ochl_lstm = LSTM(lstm, return_sequences=True)(ochl_dropout)
                ochl_dropout = Dropout(dropout_rate)(ochl_lstm)

            lstm_count += 1


        dense_count = 1
        ochl_dense = None
        bn = None
        output_layer = None

        for dense in dense_combination:
            if len(dense_combination) == 1:
                ochl_dense = Dense(dense)(ochl_dropout)
                bn = BatchNormalization()(ochl_dense)
                ochl_dense = Activation('relu')(bn)
                ochl_dense = Dropout(dropout_rate)(ochl_dense)
                output_layer = Dense(y_ochl_train.shape[1], activation='softmax')(ochl_dense)
            elif dense_count == 1:
                ochl_dense = Dense(dense)(ochl_dropout)
                bn = BatchNormalization()(ochl_dense)
                ochl_dense = Activation('relu')(bn)
                ochl_dense = Dropout(dropout_rate)(ochl_dense)
            elif dense_count == len(dense_combination):
                ochl_dense = Dense(dense)(ochl_dense)
                bn = BatchNormalization()(ochl_dense)
                ochl_dense = Activation('relu')(bn)
                ochl_dense = Dropout(dropout_rate)(ochl_dense)
                output_layer = Dense(y_ochl_train.shape[1], activation='softmax')(ochl_dense)
            else:
                ochl_dense = Dense(dense)(ochl_dense)
                bn = BatchNormalization()(ochl_dense)
                ochl_dense = Activation('relu')(bn)
                ochl_dense = Dropout(dropout_rate)(ochl_dense)

            dense_count += 1


        model = Model(inputs=[ochl_input], outputs=output_layer)
        print(model.summary())

        optimizer = Adam(learning_rate=learning_rate)

        # # 權重
        # y_train_labels = np.argmax(y_ochl_train, axis=1)
        # class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)

        # class_weight_dict = dict(enumerate(class_weights))

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', multiclass_f05_score])

        # # 學習率
        # # 初始化ReduceLROnPlateau回調
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # 監控驗證集的損失
        #                             factor=0.6,          # 學習率被減少的因子 (new_lr = lr * factor)
        #                             patience=5,          # 沒有進步的時期數，在這之後學習率會被減少
        #                             min_lr=0.000001,     # 學習率的下限
        #                             verbose=1,
        #                             mode='min')          # 信息展示模式

        # 训练模型
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, mode='min')

        callbacks = [callback]

        history = model.fit(X_ochl_train, y_ochl_train, epochs=epochs, batch_size=batch_size, validation_data=(X_ochl_val, y_ochl_val), callbacks=callbacks)

        # 評估模型
        evaluation_results = model.evaluate(X_ochl_test, y_ochl_test, batch_size=batch_size, return_dict=True)
        loss = round(evaluation_results['loss'], 4)
        accuracy = round(evaluation_results['accuracy'], 4)
        f05_score = round(evaluation_results['multiclass_f05_score'], 4)
        for key, value in evaluation_results.items():
            print(f"{key}, {value}")

        writer.writerow([lookahead, long_percentage, lstm_combination, dense_combination, learning_rate, dropout_rate, loss, accuracy, f05_score])

        # 獲取當前文件的絕對路徑
        current_path = os.path.abspath(os.path.dirname(__file__))

        # 定義上層目錄的路徑
        parent_path = os.path.join(current_path, '..')
        time = int(datetime.timestamp(datetime.now()))
        trained_model_name = f'{parent_path}/trained_models/{time}_softmax_loss-{loss:.4f}_accuracy-{accuracy:.4f}_f0.5-{f05_score:.4f}'

        model_process.save_model(trained_model_name, model, loss, accuracy)
        model_process.export_epoch_info(history, trained_model_name, use_f05_score=True)
        model_process.export_cm(trained_model_name, model, X_ochl_test, y_ochl_test, target_types, threshold=0.5, batch_size=batch_size)
        model_process.export_cm(trained_model_name, model, X_ochl_test, y_ochl_test, target_types, threshold=0.6, batch_size=batch_size)
        model_process.export_cm(trained_model_name, model, X_ochl_test, y_ochl_test, target_types, threshold=0.7, batch_size=batch_size)
        model_process.export_cm(trained_model_name, model, X_ochl_test, y_ochl_test, target_types, threshold=0.8, batch_size=batch_size)
        model_process.export_cm(trained_model_name, model, X_ochl_test, y_ochl_test, target_types, threshold=0.9, batch_size=batch_size)
        model_process.export_layer_parameters(trained_model_name, model, symbol, interval, look_back, df, batch_size, epochs, end_time_string, evaluation_results, test_and_validation_size, validation_ratio_of_test, {}, get_local_file_name)
        model_process.export_connection_info(trained_model_name, batch_size, symbol, interval, look_back, end_time_string, total_klines, target_function_info, input_model_infos, target_types, customized_cols_infos, drop_front_data_count, drop_back_data_count, get_local_file_name, new_cols)
        model_process.export_model_summary(trained_model_name, model)

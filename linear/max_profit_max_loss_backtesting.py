import sys
import os
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
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

from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, PReLU, Conv1D, MaxPooling1D, Flatten, LeakyReLU, ReLU, Bidirectional, Attention, LayerNormalization, Input, Activation, RepeatVector, Permute, Multiply, Layer, concatenate
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

def customized_specific_period_col(df):
    customized_cols_infos = []
    
    # 紀錄原先的cols
    original_cols = set(df.columns)
    # indicator_look_back = 288
    price_look_back = 480
    # df, lower_low_higher_high_info = indicators.add_lower_low_higher_high(df, 0.04, look_back=indicator_look_back, is_need_return_function_info=True)
    # customized_cols_infos.append(lower_low_higher_high_info)
    # df, supertrend_delta_info = indicators.super_trend_delta_and_risk(df, is_need_return_function_info=True)
    # customized_cols_infos.append(supertrend_delta_info)
    # df, supertrend_delta_info = indicators.super_trend_delta_and_riskv1(df, is_need_return_function_info=True)
    # customized_cols_infos.append(supertrend_delta_info)

    df, price_indicator_info = indicators.add_price_indicator(df, look_back=price_look_back, is_need_return_function_info=True)
    customized_cols_infos.append(price_indicator_info)

    # 紀錄新的cols
    modified_cols = set(df.columns)
    # 篩選多出來的cols
    new_cols = list(modified_cols - original_cols)
    return df, customized_cols_infos, new_cols

def transform_and_data_info(df, y, cols, robust_features, standard_features, minMax_features, front_drop_count, back_drop_count, look_back, original_X=[]):
    data = df[cols]

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
    if len(original_X) == 0:
        scaled = preprocessor.fit_transform(data)
    else:
        preprocessor.fit(original_X[cols])
        # 对特征进行缩放
        scaled = preprocessor.transform(data)

    if look_back != 0:
        scaled, y = modules_data.create_look_back_dataset(scaled, y, look_back)
    
    if front_drop_count == 0 and back_drop_count == 0:
        scaled = scaled
        y = y
    elif front_drop_count != 0:
        scaled = scaled[front_drop_count:]
        y = y[front_drop_count:]
    elif back_drop_count == 0:
        scaled = scaled[:back_drop_count]
        y = y[:back_drop_count]

    # 加上customized cols資訊
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

def get_original_data(model_connection_info):
    symbol = model_connection_info['Symbol']
    interval = model_connection_info['Interval']
    get_local_file_name = model_connection_info['Get Local File Name']
    total_klines = model_connection_info['Get Original Klines Count']
    end_time_string = model_connection_info['End Time']
    drop_front_data_count = model_connection_info['Drop Front Data Count']
    drop_back_data_count = model_connection_info['Drop Back Data Count']

    original_df, _ = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
    original_df, customized_cols_infos, new_cols = customized_specific_period_col(original_df)

    original_df = original_df[drop_front_data_count:-drop_back_data_count]
    original_df.reset_index(drop=True, inplace=True)

    return original_df, original_df['Max_High']


# 載入已經訓練好的模型
# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.join(current_path, '..')
trained_model_path = os.path.join(parent_path, 'trained_models/1712814536_softmax_loss-0.0001')
try:
    model = tf.keras.models.load_model(trained_model_path)
except TypeError as e:
    model = tf.keras.models.load_model(trained_model_path, compile=False)
    model.compile(optimizer='adam', loss='mean_squared_error')



model_connection_info = {}
# 构建完整的文件路径
model_info_path = os.path.join(trained_model_path, 'model_connection_info.json')

# 检查文件是否存在
if os.path.exists(model_info_path):
    # 这里可以添加读取或处理文件的代码
    # 例如，读取JSON文件
    import json
    with open(model_info_path, 'r') as file:
        model_connection_info = json.load(file)
else:
    raise FileNotFoundError(f"File not found: {model_info_path}")
# 拿原始資料
original_X, _ = get_original_data(model_connection_info)
    

symbol = model_connection_info['Symbol']
interval = model_connection_info['Interval']
look_back = model_connection_info['Look Back'] #使用回看n根數據
batch_size = model_connection_info['Batch Size']
total_klines = 10000
get_local_file_name = ''
input_model_infos = []

# 特定
end_time_string = "2024-03-31 23:59:59"


# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df, target_function_info = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
df, customized_cols_infos, new_cols = customized_specific_period_col(df)

# 拿掉前後無參考性資料
drop_front_data_count = 1000
drop_back_data_count = 300
df = df[drop_front_data_count:]
df.reset_index(drop=True, inplace=True)


X = df
max_profit_y = df['Max_High']
max_loss_y = df['Max_Low']

# 选择要缩放的列
ochl_cols = ["Close", "High", "Low", "Open", "MACD", "Signal", "Hist", "Middle Band", "Lower Band", "Upper Band"]

ochl_robust_features = ["MACD", "Signal", "Hist"]
ochl_standard_features = ["Close", "High", "Low", "Open", "Middle Band", "Lower Band", "Upper Band"]
ochl_minMax_features = []
X_ochl_scaled, X_look_back_max_profit, ochl_data_info = transform_and_data_info(X, max_profit_y, ochl_cols, ochl_robust_features, ochl_standard_features, ochl_minMax_features, 0, 0, look_back, original_X=original_X)
_, X_look_back_max_loss, _ = transform_and_data_info(X, max_loss_y, ochl_cols, ochl_robust_features, ochl_standard_features, ochl_minMax_features, 0, 0, look_back, original_X=original_X)

                
current_cols = ['Super Trend', 'Volume', 'ATR', 'RSI', "red_three_soldiers", "green_three_soldiers", 'MACD_Stronger', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618']
current_robust_features = []
current_standard_features = ['Volume', 'ATR', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618']
current_minMax_features = []

# 這個要把look_back取走才會跟接下來X_ochl搭到, 因為X_ochl會從look_back的index開始抓資料
X_current_scaled, X_current_max_profit, current_data_info = transform_and_data_info(X, max_profit_y, current_cols, current_robust_features, current_standard_features, current_minMax_features, look_back, 0, 0, original_X=original_X)

evaluation_results = model.evaluate([X_ochl_scaled, X_current_scaled], [X_look_back_max_profit, X_look_back_max_loss], batch_size=batch_size, return_dict=True)
predicted = model.predict([X_ochl_scaled, X_current_scaled], batch_size=batch_size)

X = X[-len(X_look_back_max_profit):]

X['Predicted_Max_Profit'] = predicted[0]
X['Predicted_Max_Loss'] = predicted[1]


print(predicted)

output_module_name = f'softmax'
X.to_csv(f'{trained_model_path}/{output_module_name}_output.csv', index=False)
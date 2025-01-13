import sys
import os
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import supertrend_backtesting_v1
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential, Model
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from keras.callbacks import ReduceLROnPlateau, TensorBoard

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import model as model_process
from modules import custom_model_fit_indicators as custom_model_fit_indicators

from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, PReLU, Conv1D, MaxPooling1D, Flatten, LeakyReLU, ReLU, Bidirectional, Attention, LayerNormalization, Input, Activation, RepeatVector, Permute, Multiply, Layer, concatenate

def multiclass_f1_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_f1_score(y_true, y_pred)

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
    df, supertrend_risk_score_info = indicators.calculate_supertrend_risk(df, is_need_return_function_info=True)
    customized_cols_infos.append(supertrend_risk_score_info)

    # df = indicators.add_24h_min_max_price(df, look_back=96)
    ### export customized col
    # df['EMA_9'], ema_9_function_info = indicators.calculate_ema(df['Close'], 9, is_need_return_function_info=True)
    # customized_cols_infos.append(ema_9_function_info)
    # df['EMA_21'], ema_21_function_info = indicators.calculate_ema(df['Close'], 21, is_need_return_function_info=True)
    # customized_cols_infos.append(ema_21_function_info)
    # df['ATR_35'], atr_35_function_info  = indicators.calculate_atr(df['TR'], 35, is_need_return_function_info=True)
    # customized_cols_infos.append(atr_35_function_info)

    # df['EMA_14'], ema_14_function_info  = indicators.calculate_ema(df['Close'], 14, is_need_return_function_info=True)
    # customized_cols_infos.append(ema_14_function_info)

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

symbol = "ETHUSDT"
interval = "15m"
look_back = 144 #使用回看n根數據
epochs = 300
batch_size = 128
input_model_infos = []
total_klines = 100000
get_local_file_name = ''

customized_cols_infos = []
new_cols = []

drop_front_data_count = 1000
drop_back_data_count = 300

# 當前
end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

end_time_seconds = (end_time / 1000) + 1
end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
# 特定
# end_time_string = "2023-12-31 23:59:59"

df, trained_model_path, target_y, X, predicted = supertrend_backtesting_v1.get_predict(total_klines, end_time_string, get_local_file_name)

# 假设 target_y 是 one-hot 编码，我们需要转换为标签
target_labels = np.argmax(target_y, axis=1)
# 假设 predicted 也是概率数组，我们需要转换为预测的标签
predicted_labels = np.argmax(predicted, axis=1)
# 比较 target_labels 和 predicted_labels，不一样则为0，否则为1
is_predicted_correct = (target_labels == predicted_labels).astype(int)


# 初始化一个与target_labels相同长度的数组，用于存储新的编码
# is_predicted_correct = np.zeros_like(target_labels)

# 对于每个标签，判断是否正确，以及它属于哪个组
# for i in range(len(target_labels)):
#     # if target_labels[i] in [0, 2, 4, 7]:
#     if target_labels[i] in [0, 2, 4, 6]:
#         # 如果属于0, 2, 4, 7组
#         is_predicted_correct[i] = 0 if target_labels[i] != predicted_labels[i] else 1
#     else:
#         # 如果属于1, 3, 5, 6组
#         is_predicted_correct[i] = 2 if target_labels[i] != predicted_labels[i] else 3

# for i in range(len(target_labels)):
#     # if target_labels[i] in [0, 2, 4, 7]:
#     if target_labels[i] in [0, 2, 4]:
#         # 如果属于0, 2, 4组
#         is_predicted_correct[i] = 0 if target_labels[i] == 6 else 1
#     elif target_labels[i] in [1, 3, 5]:
#         # 如果属于1, 3, 5组
#         is_predicted_correct[i] = 2 if target_labels[i] == 7 else 3
#     elif target_labels[i] in [6]:
#         is_predicted_correct[i] = 1 if target_labels[i] != predicted_labels[i] else 0
#     elif target_labels[i] in [7]:
#         is_predicted_correct[i] = 3 if target_labels[i] != predicted_labels[i] else 2
# for i in range(len(predicted_labels)):
# # if target_labels[i] in [0, 2, 4, 7]:
#     if predicted_labels[i] in [0, 2, 4]:
#         # 如果属于0, 2, 4组
#         is_predicted_correct[i] = 0 if target_labels[i] == 6 else 1
#     elif predicted_labels[i] in [1, 3, 5]:
#         # 如果属于1, 3, 5组
#         is_predicted_correct[i] = 0 if target_labels[i] == 7 else 1
#     elif predicted_labels[i] in [6]:
#         is_predicted_correct[i] = 1 if target_labels[i] != predicted_labels[i] else 0
#     elif predicted_labels[i] in [7]:
#         is_predicted_correct[i] = 1 if target_labels[i] != predicted_labels[i] else 0


# for i in range(len(predicted_labels)):
#     if predicted_labels[i] in [0, 2, 4]:
#         # 如果属于0, 2, 4组
#         is_predicted_correct[i] = 0 if target_labels[i] == 6 else 1
#     elif predicted_labels[i] in [1, 3, 5]:
#         # 如果属于1, 3, 5组
#         is_predicted_correct[i] = 2 if target_labels[i] == 7 else 3
#     elif predicted_labels[i] in [6, 7]:
#         is_predicted_correct[i] = 4 if target_labels[i] != predicted_labels[i] else 5

# for i in range(len(predicted_labels)):
#     if predicted_labels[i] in [0, 2, 4]:
#         # 如果属于0, 2, 4组
#         is_predicted_correct[i] = 0 if target_labels[i] != predicted_labels[i] else 1
#     elif predicted_labels[i] in [1, 3, 5]:
#         # 如果属于1, 3, 5组
#         is_predicted_correct[i] = 2 if target_labels[i] != predicted_labels[i] else 3
#     elif predicted_labels[i] in [6, 7]:
#         is_predicted_correct[i] = 4 if target_labels[i] != predicted_labels[i] else 5


predicted_confidence = np.max(predicted, axis=1)

X['is_predicted_correct'] = is_predicted_correct
X['Predicted'] = predicted_labels
X['Confidence'] = predicted_confidence
# target_y整理成 假設target_y和predicted不一樣就是0 不然就是1
# 把他加到X['is_predicted_correct']
# 也要加上X['Predicted'], X['Confidence']
# 動態靜態欄位不變, input加上predicted target和confidence
y = X['is_predicted_correct']
# 将目标变量转换为分类格式
target_types = sorted(y.unique().tolist())
# y_categorical = to_categorical(y, num_classes=len(target_types))
y_categorical = y.values

# 选择要缩放的列
ochl_cols = ["Close", "High", "Low", "Open", "Volume", "ATR", "Super Trend", "Down Trend", "Up Trend", 'K', 'D', 'MACD', 'Signal', 'Hist', 'MACD_Stronger']

ochl_robust_features = ["MACD", "Signal", "Hist"]
ochl_standard_features = ["Close", "High", "Low", "Open", "Volume", "ATR", 'Down Trend', 'Up Trend']
ochl_minMax_features = []
X_ochl_scaled, X_ochl_y, ochl_data_info = transform_and_data_info(X, y_categorical, ochl_cols, ochl_robust_features, ochl_standard_features, ochl_minMax_features, 0, 0, look_back)
input_model_infos.append(ochl_data_info)


current_cols = ['Super Trend', 'Delta_High_Down', 'Delta_Low_Up', 'Delta_Close_Trend', 'percentage', 'kline_color', "MACD_Cross", "Volume", "ATR"]
current_robust_features = []
current_standard_features = ['Delta_High_Down', 'Delta_Low_Up', 'Delta_Close_Trend', 'Volume', 'ATR']
current_minMax_features = []
# 這個要把look_back取走才會跟接下來X_ochl搭到, 因為X_ochl會從look_back的index開始抓資料
X_current_scaled, X_current_y, current_data_info = transform_and_data_info(X, y_categorical, current_cols, current_robust_features, current_standard_features, current_minMax_features, look_back, 0, 0)
input_model_infos.append(current_data_info)


predictred_cols = ['Volume', 'kline_color', 'ATR', 'RSI', 'percentage', 'MACD_Cross', 'MACD_Stronger', 'Predicted', 'Confidence']
predictred_robust_features = []
predictred_standard_features = ['Volume', 'ATR']
predictred_minMax_features = []
# 這個要把look_back取走才會跟接下來X_ochl搭到, 因為X_ochl會從look_back的index開始抓資料
X_predictred_scaled, X_predictred_y, predictred_data_info = transform_and_data_info(X, y_categorical, predictred_cols, predictred_robust_features, predictred_standard_features, predictred_minMax_features, look_back, 0, 0)
input_model_infos.append(predictred_data_info)

# 假设X_scaled是您的特征数据，y_categorical是您的目标数据
test_and_validation_size = 0.2
validation_ratio_of_test = 0.5  # 在测试和验证数据集中，验证集占的比例

# 首先分割出训练集和剩余集（测试集+验证集）
X_train, X_temp, y_train, y_temp = train_test_split(X_ochl_scaled, X_ochl_y, test_size=test_and_validation_size, shuffle=False)
X_current_train, X_current_temp, y_current_train, y_current_temp = train_test_split(X_current_scaled, X_current_y, test_size=test_and_validation_size, shuffle=False)
X_predictred_train, X_predictred_temp, y_predictred_train, y_predictred_temp = train_test_split(X_predictred_scaled, X_predictred_y, test_size=test_and_validation_size, shuffle=False)


# 接着将剩余集分割为测试集和验证集
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_ratio_of_test, shuffle=False)
X_current_val, X_current_test, y_current_val, y_current_test = train_test_split(X_current_temp, y_current_temp, test_size=validation_ratio_of_test, shuffle=False)
X_predictred_val, X_predictred_test, y_predictred_val, y_predictred_test = train_test_split(X_predictred_temp, y_predictred_temp, test_size=validation_ratio_of_test, shuffle=False)


# 定义输入层
ochl_input = Input(shape=(look_back, X_train.shape[2]), name='ochl_input')  # OCHL数据
current_input = Input(shape=(X_current_train.shape[1],), name='current_input')  # 当前周期指标数据
predicted_input = Input(shape=(X_predictred_train.shape[1],), name='predicted_input') 


# CNN层，用于提取时间序列中的局部特征
conv1 = Conv1D(filters=128, kernel_size=3, activation='relu')(ochl_input)
pool1 = MaxPooling1D(pool_size=2)(conv1)
conv2 = Conv1D(filters=64, kernel_size=3, activation='relu')(pool1)
pool2 = MaxPooling1D(pool_size=2)(conv2)
# 通过Flatten层将卷积层的输出扁平化，以便与LSTM层连接
cnn_out = Flatten()(pool2)

# OCHL数据的LSTM层
lstm_layer = LSTM(units=150, return_sequences=True)(ochl_input)
lstm_dropout = Dropout(0.3)(lstm_layer)
lstm_layer = LSTM(units=150, return_sequences=False)(lstm_dropout)
lstm_dropout = Dropout(0.3)(lstm_layer)
lstm_layer = Dense(100, activation='relu')(lstm_dropout)
lstm_out  = Dropout(0.3)(lstm_layer)

# 当前周期指标数据的Dense层
current_dense = Dense(150, activation='relu')(current_input)
current_droupout = Dropout(0.3)(current_dense)
current_dense = Dense(100, activation='relu')(current_droupout)
current_out = Dropout(0.3)(current_dense)

# 合并两个输入处理路径的输出
merged_layer = concatenate([cnn_out, lstm_out, current_out])

# 后续的Dense层
dense = Dense(150, activation='relu')(merged_layer)
# dense = Dense(100, activation='relu')(dense)
dense = Dropout(0.4)(dense)

predicted_dense = Dense(256, bias_regularizer=l1_l2(l1=0.0005, l2=0.001), kernel_regularizer=l1_l2(l1=0.0001, l2=0.002), activation='relu')(predicted_input)
predictedt_droupout = Dropout(0.5)(predicted_dense)
predicted_dense = Dense(256, activation='relu')(predictedt_droupout)
predicted_out = Dropout(0.5)(predicted_dense)

merged_layer2 = concatenate([dense, predicted_out])

final_dense = Dense(512, bias_regularizer=l1_l2(l1=0.001, l2=0.002), kernel_regularizer=l1_l2(l1=0.001, l2=0.005), activation='relu')(merged_layer2)
final_dense = Dropout(0.5)(final_dense)
final_dense = Dense(256, activation='relu')(final_dense)
final_dense = Dropout(0.5)(final_dense)
# output_layer = Dense(y_train.shape[1], activation='softmax')(final_dense)
output_layer = Dense(1, activation='sigmoid')(final_dense)  # 注意这里变化

# 构建模型
model = Model(inputs=[ochl_input, current_input, predicted_input], outputs=output_layer)

optimizer = Adam(learning_rate=0.001)

# 權重
# y_train_labels = np.argmax(y_train, axis=1)
# class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

class_weight_dict = dict(enumerate(class_weights))

# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', multiclass_f1_score])
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', custom_model_fit_indicators.f1_score])

# monitor_string = 'val_multiclass_f1_score'
monitor_string = 'val_loss'
# 學習率
# 初始化ReduceLROnPlateau回調
reduce_lr = ReduceLROnPlateau(monitor=monitor_string,  # 監控驗證集的損失
                            factor=0.3,          # 學習率被減少的因子 (new_lr = lr * factor)
                            patience=5,         # 沒有進步的時期數，在這之後學習率會被減少
                            min_lr=0.00001,      # 學習率的下限
                            verbose=1,
                            mode='min')           # 信息展示模式
# 训练模型
callback = tf.keras.callbacks.EarlyStopping(monitor=monitor_string, patience=26, restore_best_weights=True, mode='min')

# 设置 TensorBoard 日志目录
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [reduce_lr, callback]

history = model.fit([X_train, X_current_train, X_predictred_train], y_train, epochs=epochs, batch_size=batch_size, validation_data=([X_val, X_current_val, X_predictred_val], y_val), class_weight=class_weight_dict, callbacks=callbacks)


# 评估模型
evaluation_results = model.evaluate([X_test, X_current_test, X_predictred_test], y_test, batch_size=batch_size, return_dict=True)
loss = evaluation_results['loss']
accuracy = evaluation_results['accuracy']
f1_score = evaluation_results['f1_score']
# f1_score = evaluation_results['multiclass_f1_score']
for key, value in evaluation_results.items():
    print(f"{key}, {value}")


# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))

# 定義上層目錄的路徑
parent_path = os.path.join(current_path, '..')
time = int(datetime.datetime.timestamp(datetime.datetime.now()))
trained_model_name = f'{parent_path}/trained_models/{time}_softmax_loss-{loss:.4f}_accuracy-{accuracy:.4f}_f1-{f1_score:.4f}'
model_process.save_model(trained_model_name, model, loss, accuracy)
# model_process.export_epoch_info(history, trained_model_name)
# model_process.export_cm(trained_model_name, model, [X_test, X_current_test, X_predictred_test], y_test, target_types, batch_size=batch_size)
model_process.export_epoch_sigmoid_info(history, trained_model_name)
model_process.export_sigmoid_cm(trained_model_name, model, [X_test, X_current_test, X_predictred_test], y_test, target_types, threshold=0.5, batch_size=batch_size)
model_process.export_layer_parameters(trained_model_name, model, symbol, interval, look_back, df, batch_size, epochs, end_time_string, evaluation_results, test_and_validation_size, validation_ratio_of_test, class_weight_dict, get_local_file_name)
model_process.export_connection_info(trained_model_name, batch_size, symbol, interval, look_back, end_time_string, total_klines, None, input_model_infos, target_types, customized_cols_infos, drop_front_data_count, drop_back_data_count, get_local_file_name, new_cols)
import os, json, time
import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from keras.utils import to_categorical

from modules import custom_model_fit_indicators
from modules.data import create_look_back_dataset

# Load Model
def load_model(model_path, custom_objects):
    model = tf.keras.models.load_model(model_path, custom_objects)
    return model

def multiclass_f1_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_f1_score(y_true, y_pred)

def load_model_params(model_path, target_file):
    model_params_json = ''

    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.startswith(target_file):
                model_params_json = file

    if not model_params_json == '':
        json_file = open(f'{model_path}/{model_params_json}')
        model_params = json.load(json_file)
        return model_params
    else:
        raise FileNotFoundError(f"File not found: {target_file} not in {model_path}")  

def set_input_model_layers(df, y, original_X, input_model_infos):
    input_model_X_datas = []
    input_model_Y_datas = []
    
    for input_model_info in input_model_infos:
        cols = input_model_info['cols']
        robust_features = input_model_info['robust_features']
        standard_features = input_model_info['standard_features']
        minMax_features = input_model_info['minMax_features']
        front_drop_count = input_model_info['front_drop_count']
        back_drop_count = input_model_info['back_drop_count']
        look_back = input_model_info['look_back']

        scaled, target_y, _ = transform_and_data_info(df, y, cols, robust_features, standard_features, minMax_features, front_drop_count, back_drop_count, look_back, original_X)
        input_model_X_datas.append(scaled)
        input_model_Y_datas.append(target_y)
        
    return input_model_X_datas, input_model_Y_datas

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
    if len(original_X) == 0:
        scaled = preprocessor.fit_transform(data)
    else:
        preprocessor.fit(original_X[cols])
        # 对特征进行缩放
        scaled = preprocessor.transform(data)
    
    if look_back != 0:
        scaled, y = create_look_back_dataset(scaled, y, look_back)

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

def prepare_data(model_connection_info, data, model_type=None):
    df = pd.DataFrame()
    original_X = pd.DataFrame()

    if model_connection_info:
        target_types = model_connection_info['Target Types']
        input_model_infos = model_connection_info['Input Model Infos']
        total_klines = model_connection_info['Get Original Klines Count']
        end_time_string = model_connection_info['End Time']
        drop_front_data_count = model_connection_info['Drop Front Data Count']
        drop_back_data_count = model_connection_info['Drop Back Data Count']

    original_df = data.copy()
    original_df['datetime'] = pd.to_datetime(original_df['datetime'])
    original_df = original_df[original_df['datetime'] <= end_time_string]

    original_df = original_df[-total_klines:]
    original_df = original_df[drop_front_data_count: -drop_back_data_count]
    original_df.reset_index(drop=True, inplace=True)
    print(f'original_df: {original_df}')

    original_X = original_df
    
    if model_type == 'merged':
        df = data[-20000:]
        drop_front_data_count = 5920
    else:
        df = data[-2000:]

    df = df[drop_front_data_count:]
    df.reset_index(drop=True, inplace=True)
    print(f'df: {df}')

    X = df
    y = df['Target']
    y_categorical = to_categorical(y, num_classes=len(target_types))

    input_model_X_datas, _ = set_input_model_layers(X, y_categorical, original_X, input_model_infos)

    return input_model_X_datas

def predict_result(model_path, data, model_type=None):
    load_model_start = time.time()
    model = load_model(model_path, custom_objects={'multiclass_f1_score': multiclass_f1_score})
    load_model_end = time.time()
    print(f'Time elapsed during model loading: {load_model_end - load_model_start}')

    # Read model connection info to prepare data to predict
    model_connection_info = {}
    model_connection_info = load_model_params(model_path, 'model_connection_info')  

    prepare_data_start = time.time()
    input_model_X_datas = prepare_data(model_connection_info, data, model_type)
    prepare_data_end = time.time()
    print(f'Time elapsed during data preparation: {prepare_data_end - prepare_data_start}')

    predict_start = time.time()
    pred = model.predict(input_model_X_datas)
    predict_end = time.time()
    print(f'Time elapsed during prediction: {predict_end - predict_start}')

    print(f'pred: {pred[-1]}')
    predicted_result = np.argmax(pred[-1])
    confidence = np.max(pred[-1])

    return predicted_result, confidence

import sys
import os

os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
import csv

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer


from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import model as model_process
from modules import custom_model_fit_indicators as custom_model_fit_indicators

from tensorflow.keras.utils import to_categorical

def customized_specific_period_col(df):
    customized_cols_infos = []
    
    # 紀錄原先的cols
    original_cols = set(df.columns)
    # indicator_look_back = 192
    # price_look_back = 480
    # df = indicators.add_lower_low_higher_high(df, 0.06, look_back=indicator_look_back)
    # df = indicators.add_price_indicator(df, look_back=price_look_back)
    # df = indicators.add_24h_min_max_price(df, look_back=96)
    ### export customized col

    
    # df['ATR_35'], atr_35_function_info  = indicators.calculate_atr(df['TR'], 35, is_need_return_function_info=True)
    # customized_cols_infos.append(atr_35_function_info)

    # df['EMA_14'], ema_14_function_info  = indicators.calculate_ema(df['Close'], 14, is_need_return_function_info=True)
    # customized_cols_infos.append(ema_14_function_info)

    # 紀錄新的cols
    modified_cols = set(df.columns)
    # 篩選多出來的cols
    new_cols = list(modified_cols - original_cols)
    return df, customized_cols_infos, new_cols

# def get_max_count(csv_path):
#     step = 0.01
#     min_confidence = 0.25
#     max_confidence = 0.5
#     df = pd.read_csv(csv_path)
#     for specific_target in [3, 5, 7]:
#         # 创建confidence的分组
#         confidence_bins = np.arange(min_confidence, max_confidence + step, step)
#         df['confidence_group'] = pd.cut(df['Confidence'], bins=confidence_bins, include_lowest=True, right=False)

#         # 筛选predicted target为1且target为特定值的数据，并计算频率
#         df_filtered_positive = df[(df['Predicted Target'] == 1) & (df['Target'] == specific_target)]
#         positive_counts = df_filtered_positive.groupby('confidence_group').size()

#         # 筛选predicted target不为1且target为特定值的数据，并计算频率
#         df_filtered_negative = df[(df['Predicted Target'] != 1) & (df['Predicted Target'] != 0) & (df['Target'] == specific_target)]
#         negative_counts = df_filtered_negative.groupby('confidence_group').size()

#         # 计算差值
#         diff_counts = positive_counts - negative_counts.fillna(0)  # 对于没有negative counts的组，使用fillna(0)

#         # 找到差值最大的分组
#         max_group = diff_counts.idxmax()
#         max_count_diff = diff_counts.max()

#         print(f"在target为{specific_target}的情况下，留下最多的'predicted target'为1减去不为1的confidence值的分组是：{max_group}，差值为：{max_count_diff}")

def enhancement_strategy(X, predicted):
    time = int(datetime.datetime.timestamp(datetime.datetime.now()))
    output_module_name = f'{time}_softmax'

    total_profit = 0
    position = 0  # 当前仓位状态，0表示无仓位，1表示多头仓位，-1表示空头仓位
    entry_price = 0  # 开仓价格
    entry_index = 0  # 开仓的索引
    entry_predicted_target = 0 #開倉predict_target
    lose_count = 0
    win_count = 0

    take_profit = 0.012
    stop_loss = 0.008
    max_profit = 0

    take_profit_3_4 = 0.008
    stop_loss_3_4 = 0.012


    entry_confidence_1 = (0.4, 1)
    entry_confidence_2 = (0.4, 1)
    entry_confidence_3 = (0.7, 1)
    entry_confidence_4 = (0.7, 1)


    

    fee = 0.0004

    hold_count_limit = 16
    for i in range(0, len(predicted)):
        # 获取原始数据集中的特定时间点的数据
        original_data = X.iloc[i]  # 假设 original_X 是DataFrame


        # 获取原始的 Close 和 RSI 值
        original_close = original_data['Close']
        original_high = original_data['High']
        original_low = original_data['Low']
        original_datetime = original_data['datetime']

        p = predicted[i]
        predicted_target = np.argmax(p)
        confidence = np.max(predicted[i])

        # real target
        target = np.argmax(target_y[i])
        reason = ''

        if position != 0:
            if position == 1 and entry_predicted_target == 1:
                if (entry_price - original_low) / entry_price >= stop_loss:
                    real_stop_loss_profit = ((original_low - entry_price) / entry_price) - fee
                    current_profit = max(real_stop_loss_profit, -stop_loss - fee)
                    lose_count += 1

                    profit = current_profit
                    total_profit += profit
                    
                    position = 0
                    max_profit = 0
                    entry_price = 0 
                else:
                    current_profit = indicators.calculate_percentage_change(original_high, entry_price)
                    max_profit = max(max_profit, current_profit)
                    
                    if current_profit >= take_profit or i - entry_index >= hold_count_limit:
                        # 結單
                        if current_profit >= take_profit:
                            # 不超過take profit的percentage
                            profit = min(current_profit - fee, take_profit - fee)
                        else:
                            # 其餘正常算
                            profit = current_profit - fee

                        total_profit += profit
                        
                        if profit <= 0:
                            lose_count += 1
                        else:
                            win_count += 1
                        
                        position = 0
                        max_profit = 0
                        entry_price = 0

            elif position == -1 and entry_predicted_target == 2:
                if (original_high - entry_price) / entry_price >= stop_loss:
                    real_stop_loss_profit = ((entry_price - original_high) / entry_price) - fee
                    current_profit = max(real_stop_loss_profit, -stop_loss - fee)
                    lose_count += 1
                    
                    profit = current_profit
                    total_profit += profit

                    position = 0
                    max_profit = 0
                    entry_price = 0
                else:
                    current_profit = -indicators.calculate_percentage_change(original_low, entry_price)
                    max_profit = max(max_profit, current_profit)

                    if current_profit >= take_profit or i - entry_index >= hold_count_limit:
                        # 結單
                        if current_profit >= take_profit:
                            # 不超過take profit的percentage
                            profit = min(current_profit - fee, take_profit - fee)
                        else:
                            # 其餘正常算
                            profit = current_profit - fee

                        total_profit += profit

                        if profit <= 0:
                            lose_count += 1
                        else:
                            win_count += 1

                        position = 0
                        max_profit = 0
                        entry_price = 0
            # if position == -1 and entry_predicted_target == 3:
            #     if (entry_price - original_low) / entry_price >= stop_loss:
            #         real_stop_loss_profit = ((original_low - entry_price) / entry_price)
            #         current_profit = max(real_stop_loss_profit, -stop_loss)
            #         current_profit = -current_profit - fee
            #         win_count += 1

            #         profit = current_profit
            #         total_profit += profit
                    
            #         position = 0
            #         max_profit = 0
            #         entry_price = 0 
            #     else:
            #         current_profit = indicators.calculate_percentage_change(original_high, entry_price)
            #         max_profit = max(max_profit, current_profit)
                    
            #         if current_profit >= stop_loss_3_4 or i - entry_index >= hold_count_limit:
            #             # 結單
            #             if current_profit >= stop_loss_3_4:
            #                 # 不超過take profit的percentage
            #                 profit = min(current_profit, stop_loss_3_4)
            #             else:
            #                 # 其餘正常算
            #                 profit = current_profit
            #             # 轉反向
            #             profit = -profit - fee

            #             total_profit += profit
            #             if profit <= 0:
            #                 lose_count += 1
            #             else:
            #                 win_count += 1
                        
            #             position = 0
            #             max_profit = 0
            #             entry_price = 0
            # if position == 1 and entry_predicted_target == 4:
            #     if (original_high - entry_price) / entry_price >= stop_loss:
            #         real_stop_loss_profit = ((entry_price - original_high) / entry_price)
            #         current_profit = max(real_stop_loss_profit, -stop_loss)
            #         current_profit = -current_profit - fee
            #         win_count += 1
                    
            #         profit = current_profit
            #         total_profit += profit

            #         position = 0
            #         max_profit = 0
            #         entry_price = 0
            #     else:
            #         current_profit = -indicators.calculate_percentage_change(original_low, entry_price)
            #         max_profit = max(max_profit, current_profit)

            #         if current_profit >= stop_loss_3_4 or i - entry_index >= hold_count_limit:
            #             # 結單
            #             if current_profit >= stop_loss_3_4:
            #                 # 不超過take profit的percentage
            #                 profit = min(current_profit, stop_loss_3_4)
            #             else:
            #                 # 其餘正常算
            #                 profit = current_profit

            #             # 轉反向
            #             profit = -profit - fee

            #             total_profit += profit

            #             if profit <= 0:
            #                 lose_count += 1
            #             else:
            #                 win_count += 1

            #             position = 0
            #             max_profit = 0
            #             entry_price = 0
        else:
            max_profit = 0
            # 开仓逻辑

            if (predicted_target == 1 and confidence >= entry_confidence_1[0]) or (predicted_target == 3 and confidence >= entry_confidence_3[0]) or (predicted_target == 3 and confidence < entry_confidence_1[0]):
                position = 1

                entry_price = original_close
                entry_index = i

                entry_predicted_target = 1
            elif (predicted_target == 2 and confidence >= entry_confidence_2[0]) or (predicted_target == 4 and confidence >= entry_confidence_4[0]) or (predicted_target == 4 and confidence < entry_confidence_2[0]):
                position = -1

                entry_price = original_close
                entry_index = i
                
                entry_predicted_target = 2
                

        with open(f'{trained_model_path}/{output_module_name}_output.csv', 'a', newline='') as file:  # 'a'表示附加模式，這樣數據將會被添加到文件而不是覆蓋它
            writer = csv.writer(file)
            # 定义标题行
            headers = ["Entry Price", "Original Close", "Original High", "Original Low", "Win Count", "Lose Count", "Total Profit", "Max Profit", "Position", "Original Datetime", "Predicted Target", "Target", "Confidence", "Reason"]

            # 检查文件是否为空，如果是空的，写入标题行
            if file.tell() == 0:
                writer.writerow(headers)

            writer.writerow([entry_price, original_close, original_high, original_low, win_count, lose_count, total_profit, max_profit, position, original_datetime, predicted_target, target, confidence, reason])
    
    model_process.export_total_profit_info(df, trained_model_path, time, f'{trained_model_path}/{output_module_name}_output.csv')
    # get_max_count(f'{trained_model_path}/{output_module_name}_output.csv')



def regular_strategy(X, predicted):
    time = int(datetime.datetime.timestamp(datetime.datetime.now()))
    output_module_name = f'{time}_softmax'

    total_profit = 0
    position = 0  # 当前仓位状态，0表示无仓位，1表示多头仓位，-1表示空头仓位
    entry_price = 0  # 开仓价格
    entry_index = 0  # 开仓的索引
    entry_predicted_target = 0 #開倉predict_target
    lose_count = 0
    win_count = 0

    take_profit = 0.012
    stop_loss = 0.008
    max_profit = 0

    take_profit_3_4 = 0.008
    stop_loss_3_4 = 0.012

    # entry_confidence = 0.4
    entry_confidence = 0

    # fee = 0.0004
    fee = 0

    hold_count_limit = 12
    for i in range(0, len(predicted)):
        # 获取原始数据集中的特定时间点的数据
        original_data = X.iloc[i]  # 假设 original_X 是DataFrame


        # 获取原始的 Close 和 RSI 值
        original_close = original_data['Close']
        original_high = original_data['High']
        original_low = original_data['Low']
        original_datetime = original_data['datetime']
        upper_band = original_data['Upper Band']
        lower_band = original_data['Lower Band']
        
        p = predicted[i]
        predicted_target = np.argmax(p)
        confidence = np.max(predicted[i])

        # real target
        target = np.argmax(target_y[i])
        reason = ''

        if position != 0:
            if position == 1 and entry_predicted_target == 1:
                if (entry_price - original_low) / entry_price >= stop_loss:
                    real_stop_loss_profit = ((original_low - entry_price) / entry_price) - fee
                    current_profit = max(real_stop_loss_profit, -stop_loss - fee)
                    lose_count += 1

                    profit = current_profit
                    total_profit += profit
                    
                    position = 0
                    max_profit = 0
                    entry_price = 0 
                    reason = 'Long stop loss'
                else:
                    current_high_profit = indicators.calculate_percentage_change(original_high, entry_price)
                    current_close_profit = indicators.calculate_percentage_change(original_close, entry_price)
                    max_profit = max(max_profit, current_high_profit)
                    
                    if current_high_profit >= take_profit or i - entry_index >= hold_count_limit:
                        # 結單
                        if i - entry_index >= hold_count_limit:
                            profit = current_close_profit - fee
                            reason = 'Long hold count limit'
                        else:
                            if current_high_profit >= take_profit:
                                # 不超過take profit的percentage
                                profit = min(current_high_profit - fee, take_profit - fee)
                            else:
                                # 其餘正常算
                                profit = current_high_profit - fee

                            reason = 'Long take profit'

                        total_profit += profit
                        
                        if profit <= 0:
                            lose_count += 1
                        else:
                            win_count += 1
                        
                        position = 0
                        max_profit = 0
                        entry_price = 0

            elif position == -1 and entry_predicted_target == 2:
                if (original_high - entry_price) / entry_price >= stop_loss:
                    real_stop_loss_profit = ((entry_price - original_high) / entry_price) - fee
                    current_profit = max(real_stop_loss_profit, -stop_loss - fee)
                    lose_count += 1
                    
                    profit = current_profit
                    total_profit += profit

                    position = 0
                    max_profit = 0
                    entry_price = 0
                    reason = 'Short stop loss'
                else:
                    current_low_profit = -indicators.calculate_percentage_change(original_low, entry_price)
                    current_close_profit = -indicators.calculate_percentage_change(original_close, entry_price)
                    max_profit = max(max_profit, current_low_profit)

                    # 結單
                    if current_low_profit >= take_profit or i - entry_index >= hold_count_limit:
                        if i - entry_index >= hold_count_limit:
                            profit = current_close_profit - fee

                            reason = 'Short hold count limit'
                        else:
                            if current_low_profit >= take_profit:
                                # 不超過take profit的percentage
                                profit = min(current_low_profit - fee, take_profit - fee)
                            else:
                                # 其餘正常算
                                profit = current_low_profit - fee
                            
                            reason = 'Short take profit'

                        total_profit += profit

                        if profit <= 0:
                            lose_count += 1
                        else:
                            win_count += 1

                        position = 0
                        max_profit = 0
                        entry_price = 0
            if position == -1 and (entry_predicted_target == 3 or entry_predicted_target == 5):
                if (original_high - entry_price) / entry_price >= stop_loss_3_4:
                    real_stop_loss_profit = ((entry_price - original_high) / entry_price) - fee
                    current_profit = max(real_stop_loss_profit, -stop_loss_3_4 - fee)
                    lose_count += 1
                    
                    profit = current_profit
                    total_profit += profit

                    position = 0
                    max_profit = 0
                    entry_price = 0
                    reason = 'Reverse Short stop loss'
                else:
                    current_low_profit = -indicators.calculate_percentage_change(original_low, entry_price)
                    current_close_profit = -indicators.calculate_percentage_change(original_close, entry_price)
                    max_profit = max(max_profit, current_low_profit)

                    # 結單
                    if current_low_profit >= take_profit_3_4 or i - entry_index >= hold_count_limit:
                        if i - entry_index >= hold_count_limit:
                            profit = current_close_profit - fee

                            reason = 'Reverse Short hold count limit'
                        else:
                            if current_low_profit >= take_profit_3_4:
                                # 不超過take profit的percentage
                                profit = min(current_low_profit - fee, take_profit_3_4 - fee)
                            else:
                                # 其餘正常算
                                profit = current_low_profit - fee
                            
                            reason = 'Reverse Short take profit'

                        total_profit += profit

                        if profit <= 0:
                            lose_count += 1
                        else:
                            win_count += 1

                        position = 0
                        max_profit = 0
                        entry_price = 0
            if position == 1 and (entry_predicted_target == 4 or entry_predicted_target == 6):
                if (entry_price - original_low) / entry_price >= stop_loss_3_4:
                    real_stop_loss_profit = ((original_low - entry_price) / entry_price) - fee
                    current_profit = max(real_stop_loss_profit, -stop_loss_3_4 - fee)
                    lose_count += 1

                    profit = current_profit
                    total_profit += profit
                    
                    position = 0
                    max_profit = 0
                    entry_price = 0 
                    reason = 'Reverse Long stop loss'
                else:
                    current_high_profit = indicators.calculate_percentage_change(original_high, entry_price)
                    current_close_profit = indicators.calculate_percentage_change(original_close, entry_price)
                    max_profit = max(max_profit, current_high_profit)
                    
                    if current_high_profit >= take_profit_3_4 or i - entry_index >= hold_count_limit:
                        # 結單
                        if i - entry_index >= hold_count_limit:
                            profit = current_close_profit - fee
                            reason = 'Reverse Long hold count limit'
                        else:
                            if current_high_profit >= take_profit_3_4:
                                # 不超過take profit的percentage
                                profit = min(current_high_profit - fee, take_profit_3_4 - fee)
                            else:
                                # 其餘正常算
                                profit = current_high_profit - fee

                            reason = 'Reverse Long take profit'

                        total_profit += profit
                        
                        if profit <= 0:
                            lose_count += 1
                        else:
                            win_count += 1
                        
                        position = 0
                        max_profit = 0
                        entry_price = 0
        else:
            max_profit = 0
            # 开仓逻辑
            if original_close > upper_band:

                if (predicted_target == 1 and confidence >= entry_confidence):
                    position = 1

                    entry_price = original_close
                    entry_index = i

                    entry_predicted_target = predicted_target
                elif predicted_target == 3:
                    position = -1

                    entry_price = original_close
                    entry_index = i

                    entry_predicted_target = predicted_target
                elif predicted_target == 5:
                    position = -1
                    entry_price = original_close
                    entry_index = i

                    entry_predicted_target = predicted_target
            elif original_close < lower_band:
                if (predicted_target == 2 and confidence >= entry_confidence):
                    # position = -1

                    entry_price = original_close
                    entry_index = i
                    
                    entry_predicted_target = predicted_target
                elif (predicted_target == 4 and confidence >= 0):
                    position = 1
                    entry_price = original_close
                    entry_index = i
                    
                    entry_predicted_target = predicted_target
                elif (predicted_target == 6 and confidence >= 0):
                    position = 1
                    entry_price = original_close
                    entry_index = i
                    
                    entry_predicted_target = predicted_target
                

        with open(f'{trained_model_path}/{output_module_name}_output.csv', 'a', newline='') as file:  # 'a'表示附加模式，這樣數據將會被添加到文件而不是覆蓋它
            writer = csv.writer(file)
            # 定义标题行
            headers = ["Entry Price", "Original Close", "Original High", "Original Low", "Win Count", "Lose Count", "Total Profit", "Max Profit", "Position", "Original Datetime", "Predicted Target", "Target", "Confidence", "Reason"]

            # 检查文件是否为空，如果是空的，写入标题行
            if file.tell() == 0:
                writer.writerow(headers)

            writer.writerow([entry_price, original_close, original_high, original_low, win_count, lose_count, total_profit, max_profit, position, original_datetime, predicted_target, target, confidence, reason])





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

    return original_df, original_df['Target']


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

        scaled, y, _ = transform_and_data_info(df, y, cols, robust_features, standard_features, minMax_features, front_drop_count, back_drop_count, look_back, original_X)
        input_model_X_datas.append(scaled)
        input_model_Y_datas.append(y)
        
    return input_model_X_datas, input_model_Y_datas



def multiclass_f1_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_f1_score(y_true, y_pred)


# 載入已經訓練好的模型
# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.join(current_path, '..')
trained_model_path = os.path.join(parent_path, 'trained_models/1709987084_softmax_loss-0.7022_accuracy-0.7377_f1-0.3519_BB_strategy')
try:
    model = tf.keras.models.load_model(trained_model_path, custom_objects={'multiclass_f1_score': multiclass_f1_score})
except TypeError as e:
    model = tf.keras.models.load_model(trained_model_path, custom_objects={'multiclass_f1_score': multiclass_f1_score}, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', multiclass_f1_score])


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
original_X, original_y = get_original_data(model_connection_info)


total_klines = 20000
get_local_file_name = ''
symbol = model_connection_info['Symbol']
interval = model_connection_info['Interval']
batch_size = model_connection_info['Batch Size']
input_model_infos = model_connection_info['Input Model Infos']
target_types = model_connection_info['Target Types']

# 當前
end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

end_time_seconds = (end_time / 1000) + 1
end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# 特定
# end_time_string = "2023-12-31 23:59:59"

# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df, target_function_info = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
df, customized_cols_infos, new_cols = customized_specific_period_col(df)

# 拿掉前後無參考性資料
drop_front_data_count = 1000
drop_back_data_count = 300
df = df[drop_front_data_count:-drop_back_data_count]
df.reset_index(drop=True, inplace=True)

X = df
y = df['Target']
y_categorical = to_categorical(y, num_classes=len(target_types))

input_model_X_datas, input_model_Y_datas = set_input_model_layers(X, y_categorical, original_X, input_model_infos)
target_y = input_model_Y_datas[0]

X = X[-len(target_y):]

# 評估模型
loss, accuracy, f1 = model.evaluate(input_model_X_datas, target_y, batch_size=batch_size)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1: {f1:.4f}")

predicted = model.predict(input_model_X_datas)
# predicted = target_y


# regular_strategy(X, predicted)
enhancement_strategy(X, predicted)
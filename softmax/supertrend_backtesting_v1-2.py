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
import pytz
import itertools

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from multiprocessing import Pool


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
    df, supertrend_delta_info = indicators.super_trend_delta_and_risk_v1(df, is_need_return_function_info=True)

    df, supertrend_delta_info = indicators.super_trend_delta_and_risk(df, is_need_return_function_info=True)
    customized_cols_infos.append(supertrend_delta_info)

    # 紀錄新的cols
    modified_cols = set(df.columns)
    # 篩選多出來的cols
    new_cols = list(modified_cols - original_cols)
    return df, customized_cols_infos, new_cols

def grid_enhancement_strategy(target_y, X, predicted_targets, confidences, grid_target, start_range, end_range, step, tp):
    # confidence_start_range = np.arange(0.4, 0.5, 0.01)
    # confidence_end_range = np.arange(0.41, 0.5, 0.01) 
    confidence_start_range = np.arange(start_range, end_range, step)
    confidence_end_range = np.arange(start_range + step, end_range, step) 
    trailing_threshold_range = np.arange(0.006, 0.014, 0.002)
    run_up_stop_activation_range = np.arange(0.005, 0.02, 0.005)
    run_up_constant_range = np.arange(10, 25, 5)


    # max_total_profit為負無窮大
    max_total_profit = float('-inf')
    max_start = 0
    max_end = 0
    for start in confidence_start_range:
        for end in confidence_end_range:
            # for trailing_threshold in trailing_threshold_range:
            #     for run_up_stop_activation in run_up_stop_activation_range:
            #         for run_up_constant in run_up_constant_range:
            if end > start:
                entry_confidence = (start, end)
                time = int(datetime.datetime.timestamp(datetime.datetime.now()))
                output_module_name = f'{time}_softmax'

                total_profit = 0
                position = 0  # 当前仓位状态，0表示无仓位，1表示多头仓位，-1表示空头仓位
                entry_price = 0  # 开仓价格
                entry_index = 0  # 开仓的索引
                entry_predicted_target = 0 #開倉predict_target
                lose_count = 0
                win_count = 0

                # 1, 2 hold_count_at_least_profit% - 1%
                # 3, 4 1% - 2.0%
                # 5, 6 2% - 3%
                # 7, 8 3%up
                # 9, 10 0.8%down


                stop_loss = 0.008
                max_profit = 0

                # take_profit_reverse = 0.008
                # stop_loss_reverse = 0.012

                # trailing_threshold = 0.01
                # trailing_return_threshold = 0.008


                fee = 0.0004

                hold_count_limit = 20

                loss_hold_count_limit = 6
                current_loss_hold_count = 0

                trailing_threshold = 0.008
                trailing_return_threshold = 0.004
                # 4根維持就出
                no_keep_update_max_profit_count_limit = 4
                no_keep_update_max_profit_count = 0 
                no_keep_trend_threshold = [0.003, trailing_threshold]

                # 2根以內close突然回調超過max_profit一半也出
                # suddenly_callback_count = no_keep_update_max_profit_count_limit
                # activation_percentage_threshold = 0.008
                # suddenly_power_exit_threshold = 0.33

                # run_up_setting
                run_up_stop_activation = 0.008
                run_up_constant = 35
            

                is_finish_order = False
                for i in range(1, len(predicted_targets)):
                    # 获取原始数据集中的特定时间点的数据
                    original_data = X.iloc[i]  # 假设 original_X 是DataFrame
                    pre_original_data = X.iloc[i - 1]

                    

                    # 获取原始的 Close 和 RSI 值
                    original_close = original_data['Close']
                    original_open = original_data['Open']
                    original_high = original_data['High']
                    original_low = original_data['Low']
                    original_datetime = original_data['datetime']
                    macd_stronger = original_data['MACD_Stronger']
                    hist = original_data['Hist']

                    upper_band = original_data['Upper Band']
                    lower_band = original_data['Lower Band']
                    
                    # pre data
                    pre_supertrend = pre_original_data['Super Trend']

                    supertrend = original_data['Super Trend']
                    volume = original_data['Volume']
                    kline_color = original_data['kline_color']
                    
                    predicted_target = predicted_targets[i]
                    confidence = confidences[i]

                    # max_high_look_back_24h = original_data['max_high_look_back_24h']
                    # min_low_look_back_24h = original_data['min_low_look_back_24h']

                    # real target
                    target = np.argmax(target_y[i])
                    reason = ''
                    

                    if position != 0:
                        current_hold_count = i - entry_index
                        take_profit = tp

                        trailing_threshold = take_profit / 1.5
                        trailing_return_threshold = trailing_threshold / 2
                        if trailing_return_threshold <= 0.004:
                            trailing_return_threshold = 0.004

                        # 4根維持就出
                        no_keep_trend_threshold = [0.003, trailing_threshold]



                        if position == 1 and (entry_predicted_target == 0 or entry_predicted_target == 2 or entry_predicted_target == 4 or entry_predicted_target == 6):
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
                                current_low_profit = indicators.calculate_percentage_change(original_low, entry_price)
                                
                                max_profit = max(max_profit, current_high_profit)

                                
                                runup_stop_price = entry_price * ((100 - stop_loss * 100) / 100) * (1 + (max_profit / run_up_constant * current_hold_count))
                                # runup_stop_price = pre_original_low * (1 - run_up_constant)
                                # runup_stop_price = max(entry_price * ((100 - stop_loss * 100) / 100) * (1 + (max_profit / run_up_constant * current_hold_count)), pre_original_low - (pre_ATR / 3))

                                if current_close_profit <= 0.0:
                                    current_loss_hold_count += 1
                                else:
                                    current_loss_hold_count = 0

                                if current_high_profit <= max_profit:
                                    no_keep_update_max_profit_count += 1
                                else:
                                    no_keep_update_max_profit_count = 0
                                # 結單
                                if max_profit >= take_profit:
                                    # 不超過take profit的percentage
                                    profit = min(max_profit - fee, take_profit - fee)
                                    reason = f'Long take profit {take_profit}' 
                                    is_finish_order = True
                                elif max_profit >= run_up_stop_activation and (original_close <= runup_stop_price or original_open <= runup_stop_price):
                                    actual_price = 0
                                    if original_open <= runup_stop_price:
                                        actual_price = original_open
                                    else:
                                        actual_price = original_close
                                    profit = indicators.calculate_percentage_change(actual_price, entry_price) - fee
                                    reason = f'Long Runup stop Triggered {profit}'
                                    is_finish_order = True
                                elif max_profit >= trailing_threshold and max_profit - current_close_profit >= trailing_return_threshold:
                                    # 一定trailing的狀況
                                    profit = min(max_profit - trailing_return_threshold - fee, take_profit - fee)
                                    reason = f'Long must be trailing stop'
                                    is_finish_order = True
                                elif no_keep_trend_threshold[0] <= max_profit <= no_keep_trend_threshold[1] and no_keep_update_max_profit_count >= no_keep_update_max_profit_count_limit:
                                    profit = current_close_profit - fee
                                    reason = f'Long no keep update max profit count'
                                    is_finish_order = True
                                # elif pre_supertrend == 1 and supertrend == -1:
                                #     profit = current_close_profit - fee
                                #     reason = f'Long Supertrend changed'
                                #     is_finish_order = True
                                # elif (no_keep_update_max_profit_count < suddenly_callback_count and max_profit >= activation_percentage_threshold and (max_profit * suddenly_power_exit_threshold) >= current_close_profit):
                                #     profit = current_close_profit - fee
                                #     reason = f'Long short time suddenly callback stop'
                                    # is_finish_order = True
                                # elif (max_profit >= trailing_threshold and current_low_profit <= max_profit - trailing_return_threshold):
                                #     # 回調的狀況先拿profit
                                #     profit = max_profit - trailing_return_threshold - fee
                                #     reason = f'Long trailing stop'
                                #     is_finish_order = True
                                # elif (max_profit >= detect_is_need_exit_threshold and current_close_profit <= if_win_at_least_profit):
                                #     profit = current_close_profit - fee
                                #     reason = f'Long at least profit stop'
                                    # is_finish_order = True
                                elif (current_loss_hold_count >= loss_hold_count_limit):
                                    profit = current_close_profit - fee
                                    reason = f'Long loss hold count'
                                    is_finish_order = True
                                elif current_hold_count >= hold_count_limit:
                                    profit = current_close_profit - fee
                                    reason = 'Long hold count limit'
                                    is_finish_order = True

                                
                                if is_finish_order:
                                    total_profit += profit
                                    
                                    if profit <= 0:
                                        lose_count += 1
                                    else:
                                        win_count += 1
                                    
                                    position = 0
                                    max_profit = 0
                                    entry_price = 0
                            


                        elif position == -1 and (entry_predicted_target == 1 or entry_predicted_target == 3 or entry_predicted_target == 5 or entry_predicted_target == 7):
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
                                current_high_profit = -indicators.calculate_percentage_change(original_high, entry_price)
                                current_low_profit = -indicators.calculate_percentage_change(original_low, entry_price)
                                current_close_profit = -indicators.calculate_percentage_change(original_close, entry_price)
                                max_profit = max(max_profit, current_low_profit)

                                runup_stop_price = entry_price * ((100 + stop_loss * 100) / 100) * (1 - (max_profit / run_up_constant * current_hold_count))
                                # runup_stop_price = pre_original_high * (1 + run_up_constant)
                                # runup_stop_price = min(entry_price * ((100 + stop_loss * 100) / 100) * (1 - (max_profit / run_up_constant * current_hold_count)), pre_original_high + (pre_ATR / 3))

                                if current_close_profit <= 0:
                                    current_loss_hold_count += 1
                                else:
                                    current_loss_hold_count = 0

                                if current_high_profit <= max_profit:
                                    no_keep_update_max_profit_count += 1
                                else:
                                    no_keep_update_max_profit_count = 0
                                # 結單
                                if max_profit >= take_profit:
                                    # 不超過take profit的percentage
                                    profit = min(max_profit - fee, take_profit - fee)
                                    reason = f'Short take profit {take_profit}'
                                    is_finish_order = True
                                elif run_up_stop_activation and (original_close >= runup_stop_price or original_open >= runup_stop_price):
                                    actual_price = 0
                                    if original_open >= runup_stop_price:
                                        actual_price = original_open
                                    else:
                                        actual_price = original_close
                                    profit = -indicators.calculate_percentage_change(actual_price, entry_price) - fee
                                    reason = f'Short Runup stop Triggered {profit}'
                                    is_finish_order = True
                                elif max_profit >= trailing_threshold and max_profit - current_close_profit >= trailing_return_threshold:
                                    # 一定trailing的狀況
                                    profit = min(max_profit - trailing_return_threshold - fee, take_profit - fee)
                                    reason = f'Short must be trailing stop'
                                    is_finish_order = True
                                elif no_keep_trend_threshold[0] <= max_profit <= no_keep_trend_threshold[1] and no_keep_update_max_profit_count >= no_keep_update_max_profit_count_limit:
                                    profit = current_close_profit - fee
                                    reason = f'Short no keep update max profit count'
                                    is_finish_order = True
                                # elif pre_supertrend == -1 and supertrend == 1:
                                #     profit = current_close_profit - fee
                                #     reason = f'Short Supertrend changed'
                                #     is_finish_order = True
                                # elif (no_keep_update_max_profit_count < suddenly_callback_count and max_profit >= activation_percentage_threshold and (max_profit * suddenly_power_exit_threshold) >= current_close_profit):
                                #     profit = current_close_profit - fee
                                #     reason = f'Short short time suddenly callback stop'
                                #     is_finish_order = True
                                # elif (max_profit >= trailing_threshold and current_high_profit <= max_profit - trailing_return_threshold):
                                #     # 回調的狀況先拿profit
                                #     profit = max_profit - trailing_return_threshold - fee
                                #     reason = f'Short trailing stop'
                                #     is_finish_order = True
                                # elif (max_profit >= detect_is_need_exit_threshold and current_close_profit <= if_win_at_least_profit):
                                #     profit = if_win_at_least_profit - fee
                                #     reason = f'Short at least profit stop'
                                    # is_finish_order = True 
                                elif (current_loss_hold_count >= loss_hold_count_limit):
                                    profit = current_close_profit - fee
                                    reason = f'Short loss hold count'
                                    is_finish_order = True
                                elif current_hold_count >= hold_count_limit:
                                    profit = current_close_profit - fee
                                    reason = 'Short hold count limit'
                                    is_finish_order = True
                                
                                
                                if is_finish_order:
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
                        current_loss_hold_count = 0
                        no_keep_update_max_profit_count = 0
                        is_finish_order = False
                        # 开仓逻辑

                        volume_threshold = 20000

                        if supertrend == 1 and predicted_target == grid_target and volume >= volume_threshold:
                            if (predicted_target == 0 and entry_confidence[0] <= confidence <= entry_confidence[1]):
                                position = 1

                                entry_price = original_close
                                entry_index = i

                                entry_predicted_target = predicted_target
                            elif (predicted_target == 2 and entry_confidence[0] <= confidence <= entry_confidence[1]):
                                position = 1

                                entry_price = original_close
                                entry_index = i

                                entry_predicted_target = predicted_target
                            elif (predicted_target == 4 and entry_confidence[0] <= confidence <= entry_confidence[1]):
                                position = 1
                                entry_price = original_close
                                entry_index = i

                                entry_predicted_target = predicted_target
                            elif (predicted_target == 6 and entry_confidence[0] <= confidence <= entry_confidence[1]):
                                position = 1
                                entry_price = original_close
                                entry_index = i

                                entry_predicted_target = predicted_target
                        if supertrend == -1 and predicted_target == grid_target and volume >= volume_threshold:
                            if (predicted_target == 1 and entry_confidence[0] <= confidence <= entry_confidence[1]):
                                position = -1

                                entry_price = original_close
                                entry_index = i
                                
                                entry_predicted_target = predicted_target
                            elif (predicted_target == 3 and entry_confidence[0] <= confidence <= entry_confidence[1]):
                                position = -1

                                entry_price = original_close
                                entry_index = i
                                
                                entry_predicted_target = predicted_target
                            elif (predicted_target == 5 and entry_confidence[0] <= confidence <= entry_confidence[1]):
                                position = -1
                                entry_price = original_close
                                entry_index = i
                                
                                entry_predicted_target = predicted_target
                            elif (predicted_target == 7 and entry_confidence[0] <= confidence <= entry_confidence[1]):
                                position = -1
                                entry_price = original_close
                                entry_index = i

                                entry_predicted_target = predicted_target
                # print(grid_target, total_profit, start, end, run_up_stop_activation, run_up_constant, trailing_threshold)

                # 配置输出CSV文件
                with open('grid_search_results.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    # 写入标题行
                    writer.writerow([grid_target, total_profit, start, end, tp])

                print(grid_target, total_profit, start, end, tp)
                if max_total_profit < total_profit:
                    max_total_profit = total_profit
                    max_start = start
                    max_end = end

                        
                # max_total_profit = max(max_total_profit, total_profit)
            #     with open(f'{trained_model_path}/{output_module_name}_output.csv', 'a', newline='') as file:  # 'a'表示附加模式，這樣數據將會被添加到文件而不是覆蓋它
            #         writer = csv.writer(file)
            #         # 定义标题行
            #         headers = ["Runup Stop Price", "Entry Price", "Original Close", "Original High", "Original Low", "Win Count", "Lose Count", "Total Profit", "Max Profit", "Position", "Original Datetime", "Predicted Target", "Target", "Confidence", "Reason"]

            #         # 检查文件是否为空，如果是空的，写入标题行
            #         if file.tell() == 0:
            #             writer.writerow(headers)


            #         # 指定原始时间戳的时区为UTC
            #         original_datetime_utc = original_datetime.tz_localize('UTC')

            #         # 转换为台湾时间（UTC+8）
            #         taiwan_datetime = original_datetime_utc.tz_convert('Asia/Taipei')

            #         # 格式化为字符串
            #         taiwan_datetime_str = taiwan_datetime.strftime('%Y-%m-%d %H:%M:%S')

            #         writer.writerow([runup_stop_price, entry_price, original_close, original_high, original_low, win_count, lose_count, total_profit, max_profit, position, taiwan_datetime_str, predicted_target, target, confidence, reason])

            # model_process.export_total_profit_info(df, trained_model_path, time, f'{trained_model_path}/{output_module_name}_output.csv')
            # get_max_count(f'{trained_model_path}/{output_module_name}_output.csv')
    # 配置输出CSV文件
    with open('grid_search_results.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # 写入标题行
        writer.writerow([grid_target, max_total_profit, max_start, max_end, tp])
        writer.writerow(['End'])
    print(max_total_profit, max_start, max_end, tp)
    return max_total_profit, max_start, max_end

def enhancement_strategy_v2(df, trained_model_path, target_y, X, predicted_targets, confidences):
    time = int(datetime.datetime.timestamp(datetime.datetime.now()))
    output_module_name = f'{time}_softmax'

    total_profit = 0
    position = 0  # 当前仓位状态，0表示无仓位，1表示多头仓位，-1表示空头仓位
    entry_price = 0  # 开仓价格
    entry_index = 0  # 开仓的索引
    entry_predicted_target = 0 #開倉predict_target
    lose_count = 0
    win_count = 0

    take_profit_0_1 = 0.025
    take_profit_2_3 = 0.03
    take_profit_4_5 = 0.03

    take_profit_6_7 = 0.008
    stop_loss_6_7 = 0.005

    stop_loss = 0.008
    max_profit = 0

    # take_profit_reverse = 0.008
    # stop_loss_reverse = 0.012

    trailing_threshold = 0.015
    trailing_return_threshold = 0.005

    detect_is_need_exit_threshold = 0.008
    if_win_at_least_profit = 0.006

    fee = 0.0004
    # fee = 0

    reverse_hold_count_limit = 20
    hold_count_limit = 20

    loss_hold_count_limit = 6
    current_loss_hold_count = 0


    # 4根維持就出
    # no_keep_update_max_profit_count_limit = 3
    # no_keep_update_max_profit_count = 0
    # no_keep_trend_threshold = 0.008

    # # 2根以內close突然回調超過max_profit一半也出
    # suddenly_callback_count = no_keep_update_max_profit_count_limit
    # activation_percentage_threshold = 0.008
    # suddenly_power_exit_threshold = 0.33

    # run_up_setting
    # activation 0.08,  每次都讓stop price設定為 最高或最低 / 2的位置
    run_up_stop_activation = 0.008
    run_up_constant = 35
    # run_up_constant = 0.0025

    
    threshold = 0.003
    
    # run_up_stop_activation = 0.012
    # run_up_constant = 25

    # adjustment
    # entry_confidence_0 = (0.7, 1)
    # entry_confidence_2 = (0.55, 0.6)
    # entry_confidence_4 = (0.6, 0.7)

    # entry_confidence_1 = (0.7, 0.75)
    # entry_confidence_3 = (0.44, 0.46)
    # entry_confidence_5 = (0.45, 0.55)

    # best
    # entry_confidence_0 = (0.65, 1)
    # entry_confidence_2 = (0.5, 0.6)
    # entry_confidence_4 = (0.6, 0.7)

    # entry_confidence_1 = (0.7, 1)
    # entry_confidence_3 = (0.4, 0.5)
    # entry_confidence_5 = (0.45, 0.55)

    entry_confidence_0 = (0.65, 0.85)
    entry_confidence_2 = (0.5, 0.6)
    entry_confidence_4 = (0.6, 0.7)

    entry_confidence_1 = (0.55, 0.75)
    entry_confidence_3 = (0.4, 0.5)
    entry_confidence_5 = (0.45, 0.55)


    # entry_confidence_0 = (0, 1)
    # entry_confidence_1 = (0, 1)
    # entry_confidence_2 = (0, 1)
    # entry_confidence_3 = (0, 1)
    # entry_confidence_4 = (0, 1)
    # entry_confidence_5 = (0, 1)

    is_finish_order = False
    for i in range(1, len(predicted_targets)):
        # 获取原始数据集中的特定时间点的数据
        original_data = X.iloc[i]  # 假设 original_X 是DataFrame
        pre_original_data = X.iloc[i - 1]

        

        # 获取原始的 Close 和 RSI 值
        original_close = original_data['Close']
        original_open = original_data['Open']
        original_high = original_data['High']
        original_low = original_data['Low']
        original_datetime = original_data['datetime']
        macd_stronger = original_data['MACD_Stronger']

        upper_band = original_data['Upper Band']
        lower_band = original_data['Lower Band']
        
        # pre data
        pre_supertrend = pre_original_data['Super Trend']
        pre_original_high = pre_original_data['High']
        pre_original_low = pre_original_data['Low']
        pre_original_close = original_data['Close']
        pre_original_open = original_data['Open']
        pre_upper_band = pre_original_data['Upper Band']
        pre_lower_band = pre_original_data['Lower Band']
        pre_ATR = pre_original_data['ATR']

        supertrend = original_data['Super Trend']
        volume = original_data['Volume']
        kline_color = original_data['kline_color']
        
        predicted_target = predicted_targets[i]
        confidence = confidences[i]

        # real target
        target = np.argmax(target_y[i])
        reason = ''
        
        
        runup_stop_price = 0
        if position != 0:
            if entry_predicted_target == 0 or entry_predicted_target == 1:
                take_profit = take_profit_0_1
            elif entry_predicted_target == 2 or entry_predicted_target == 3:
                take_profit = take_profit_2_3
            elif entry_predicted_target == 4 or entry_predicted_target == 5:
                take_profit = take_profit_4_5

            current_hold_count = i - entry_index
            
            # if max_profit >= 0.005:
            #     stop_loss = 0.005
            # elif max_profit >= 0.008:
            #     stop_loss = 0.0
            # elif max_profit >= 0.012:
            #     stop_loss = -0.005
            # else:
            #     stop_loss = 0.008

            if position == 1 and (entry_predicted_target == 0 or entry_predicted_target == 2 or entry_predicted_target == 4):
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
                    current_low_profit = indicators.calculate_percentage_change(original_low, entry_price)
                    
                    max_profit = max(max_profit, current_high_profit)

                    
                    runup_stop_price = entry_price * ((100 - stop_loss * 100) / 100) * (1 + (max_profit / run_up_constant * current_hold_count))
                    # runup_stop_price = pre_original_low * (1 - run_up_constant)
                    # runup_stop_price = max(entry_price * ((100 - stop_loss * 100) / 100) * (1 + (max_profit / run_up_constant * current_hold_count)), pre_original_low - (pre_ATR / 3))

                    if current_close_profit <= 0.0:
                        current_loss_hold_count += 1
                    else:
                        current_loss_hold_count = 0

                    if current_high_profit <= max_profit:
                        no_keep_update_max_profit_count += 1
                    else:
                        no_keep_update_max_profit_count = 0
                    # 結單
                    if max_profit >= take_profit:
                        # 不超過take profit的percentage
                        profit = min(max_profit - fee, take_profit - fee)
                        reason = f'Long take profit {take_profit}' 
                        is_finish_order = True
                    elif max_profit >= run_up_stop_activation and (original_close <= runup_stop_price or original_open <= runup_stop_price):
                        actual_price = 0
                        if original_open <= runup_stop_price:
                            actual_price = original_open
                        else:
                            actual_price = original_close
                        profit = indicators.calculate_percentage_change(actual_price, entry_price) - fee
                        reason = f'Long Runup stop Triggered {profit}'
                        is_finish_order = True
                    elif pre_supertrend == 1 and supertrend == -1:
                        profit = current_close_profit - fee
                        reason = f'Long Supertrend changed'
                        is_finish_order = True
                    # elif (no_keep_update_max_profit_count >= no_keep_update_max_profit_count_limit and (max_profit - current_low_profit > no_keep_trend_threshold)):
                    #     profit = current_close_profit - fee
                    #     reason = f'Long no keep trend stop'
                        # is_finish_order = True
                    # elif (no_keep_update_max_profit_count < suddenly_callback_count and max_profit >= activation_percentage_threshold and (max_profit * suddenly_power_exit_threshold) >= current_close_profit):
                    #     profit = current_close_profit - fee
                    #     reason = f'Long short time suddenly callback stop'
                        # is_finish_order = True
                    # elif (max_profit >= trailing_threshold and current_low_profit <= max_profit - trailing_return_threshold):
                    #     # 回調的狀況先拿profit
                    #     profit = max_profit - trailing_return_threshold - fee
                    #     reason = f'Long trailing stop'
                        # is_finish_order = True
                    # elif (max_profit >= detect_is_need_exit_threshold and current_close_profit <= if_win_at_least_profit):
                    #     profit = current_close_profit - fee
                    #     reason = f'Long at least profit stop'
                        # is_finish_order = True
                    elif (current_loss_hold_count >= loss_hold_count_limit):
                        profit = current_close_profit - fee
                        reason = f'Long loss hold count'
                        is_finish_order = True
                    elif current_hold_count >= hold_count_limit:
                        profit = current_close_profit - fee
                        reason = 'Long hold count limit'
                        is_finish_order = True

                    
                    if is_finish_order:
                        total_profit += profit
                        
                        if profit <= 0:
                            lose_count += 1
                        else:
                            win_count += 1
                        
                        position = 0
                        max_profit = 0
                        entry_price = 0
                


            elif position == -1 and (entry_predicted_target == 1 or entry_predicted_target == 3 or entry_predicted_target == 5):
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
                    current_high_profit = -indicators.calculate_percentage_change(original_high, entry_price)
                    current_low_profit = -indicators.calculate_percentage_change(original_low, entry_price)
                    current_close_profit = -indicators.calculate_percentage_change(original_close, entry_price)
                    max_profit = max(max_profit, current_low_profit)

                    runup_stop_price = entry_price * ((100 + stop_loss * 100) / 100) * (1 - (max_profit / run_up_constant * current_hold_count))
                    # runup_stop_price = pre_original_high * (1 + run_up_constant)
                    # runup_stop_price = min(entry_price * ((100 + stop_loss * 100) / 100) * (1 - (max_profit / run_up_constant * current_hold_count)), pre_original_high + (pre_ATR / 3))

                    if current_close_profit <= 0:
                        current_loss_hold_count += 1
                    else:
                        current_loss_hold_count = 0

                    if current_high_profit <= max_profit:
                        no_keep_update_max_profit_count += 1
                    else:
                        no_keep_update_max_profit_count = 0
                    # 結單
                    if max_profit >= take_profit:
                        # 不超過take profit的percentage
                        profit = min(max_profit - fee, take_profit - fee)
                        reason = f'Short take profit {take_profit}'
                        is_finish_order = True
                    elif run_up_stop_activation and (original_close >= runup_stop_price or original_open >= runup_stop_price):
                        actual_price = 0
                        if original_open >= runup_stop_price:
                            actual_price = original_open
                        else:
                            actual_price = original_close
                        profit = -indicators.calculate_percentage_change(actual_price, entry_price) - fee
                        reason = f'Short Runup stop Triggered {profit}'
                        is_finish_order = True
                    elif pre_supertrend == -1 and supertrend == 1:
                        profit = current_close_profit - fee
                        reason = f'Short Supertrend changed'
                        is_finish_order = True
                    # elif (no_keep_update_max_profit_count >= no_keep_update_max_profit_count_limit and (max_profit - current_high_profit > no_keep_trend_threshold)):
                    #     profit = current_close_profit - fee
                    #     reason = f'Short no keep trend stop'
                    #     is_finish_order = True
                    # elif (no_keep_update_max_profit_count < suddenly_callback_count and max_profit >= activation_percentage_threshold and (max_profit * suddenly_power_exit_threshold) >= current_close_profit):
                    #     profit = current_close_profit - fee
                    #     reason = f'Short short time suddenly callback stop'
                    #     is_finish_order = True
                    # elif (max_profit >= trailing_threshold and current_high_profit <= max_profit - trailing_return_threshold):
                    #     # 回調的狀況先拿profit
                    #     profit = max_profit - trailing_return_threshold - fee
                    #     reason = f'Short trailing stop'
                        # is_finish_order = True
                    # elif (max_profit >= detect_is_need_exit_threshold and current_close_profit <= if_win_at_least_profit):
                    #     profit = if_win_at_least_profit - fee
                    #     reason = f'Short at least profit stop'
                        # is_finish_order = True 
                    elif (current_loss_hold_count >= loss_hold_count_limit):
                        profit = current_close_profit - fee
                        reason = f'Short loss hold count'
                        is_finish_order = True
                    elif current_hold_count >= hold_count_limit:
                        profit = current_close_profit - fee
                        reason = 'Short hold count limit'
                        is_finish_order = True
                    
                    
                    if is_finish_order:
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
            current_loss_hold_count = 0
            no_keep_update_max_profit_count = 0
            is_finish_order = False
            # 开仓逻辑
            
            if supertrend == 1 and kline_color == 0:
                if (predicted_target == 0 and entry_confidence_0[0] <= confidence <= entry_confidence_0[1]) and volume >= 35000:
                    position = 1

                    entry_price = original_close
                    entry_index = i

                    entry_predicted_target = predicted_target
                elif predicted_target == 2 and entry_confidence_2[0] <= confidence <= entry_confidence_2[1] and volume >= 45000:
                    position = 1

                    entry_price = original_close
                    entry_index = i

                    entry_predicted_target = predicted_target
                elif predicted_target == 4 and entry_confidence_4[0] <= confidence <= entry_confidence_4[1] and volume >= 45000:
                    position = 1
                    entry_price = original_close
                    entry_index = i

                    entry_predicted_target = predicted_target
            elif supertrend == -1 and kline_color == 1:
                if (predicted_target == 1 and entry_confidence_1[0] <= confidence <= entry_confidence_1[1]) and volume >= 35000:
                    position = -1

                    entry_price = original_close
                    entry_index = i
                    
                    entry_predicted_target = 1
                elif (predicted_target == 3 and entry_confidence_3[0] <= confidence <= entry_confidence_3[1]) and volume >= 45000:
                    position = -1
                    entry_price = original_close
                    entry_index = i
                    
                    entry_predicted_target = predicted_target
                elif (predicted_target == 5 and entry_confidence_5[0] <= confidence <= entry_confidence_5[1]) and volume >= 45000:
                    position = -1
                    entry_price = original_close
                    entry_index = i
                    
                    entry_predicted_target = predicted_target
                

        with open(f'{trained_model_path}/{output_module_name}_output.csv', 'a', newline='') as file:  # 'a'表示附加模式，這樣數據將會被添加到文件而不是覆蓋它
            writer = csv.writer(file)
            # 定义标题行
            headers = ["Runup Stop Price", "Entry Price", "Original Close", "Original High", "Original Low", "Win Count", "Lose Count", "Total Profit", "Max Profit", "Position", "Original Datetime", "Predicted Target", "Target", "Confidence", "Reason"]

            # 检查文件是否为空，如果是空的，写入标题行
            if file.tell() == 0:
                writer.writerow(headers)


            # 指定原始时间戳的时区为UTC
            original_datetime_utc = original_datetime.tz_localize('UTC')

            # 转换为台湾时间（UTC+8）
            taiwan_datetime = original_datetime_utc.tz_convert('Asia/Taipei')

            # 格式化为字符串
            taiwan_datetime_str = taiwan_datetime.strftime('%Y-%m-%d %H:%M:%S')

            writer.writerow([runup_stop_price, entry_price, original_close, original_high, original_low, win_count, lose_count, total_profit, max_profit, position, taiwan_datetime_str, predicted_target, target, confidence, reason])

    model_process.export_total_profit_info(df, trained_model_path, time, f'{trained_model_path}/{output_module_name}_output.csv')
    # get_max_count(f'{trained_model_path}/{output_module_name}_output.csv')


def get_original_data(model_connection_info):
    symbol = model_connection_info['Symbol']
    interval = model_connection_info['Interval']
    get_local_file_name = model_connection_info['Get Local File Name']
    total_klines = model_connection_info['Get Original Klines Count']
    end_time_string = model_connection_info['End Time']
    drop_front_data_count = model_connection_info['Drop Front Data Count']
    drop_back_data_count = model_connection_info['Drop Back Data Count']

    original_df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
    original_df, customized_cols_infos, new_cols = customized_specific_period_col(original_df)
    original_df['Target'] = signals.super_trend_strategy_v2(original_df, 0.008, 0.008, 0, 20)

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

def get_predict(total_klines, end_time_string, get_local_file_name):
    # 載入已經訓練好的模型
    # 獲取當前文件的絕對路徑
    current_path = os.path.abspath(os.path.dirname(__file__))
    parent_path = os.path.join(current_path, '..')
    trained_model_path = os.path.join(parent_path, 'trained_models/1710835373_softmax_loss-1.2277_accuracy-0.4738_f1-0.2171_supertrend_real_best')
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

    symbol = model_connection_info['Symbol']
    interval = model_connection_info['Interval']
    batch_size = model_connection_info['Batch Size']
    input_model_infos = model_connection_info['Input Model Infos']
    target_types = model_connection_info['Target Types']

    # 轉毫秒
    # end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
    # Step 1: 獲取數據
    df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
    df['Target'] = signals.super_trend_strategy_v1(df, 0.008, 0.008, 0, 20)
    df, customized_cols_infos, new_cols = customized_specific_period_col(df)
    
    # 拿掉前後無參考性資料
    drop_front_data_count = 1000
    drop_back_data_count = 300
    df = df[drop_front_data_count:]
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

    return df, trained_model_path, target_y, X, predicted

def main():
    total_klines = 30000
    get_local_file_name = ''
    # 當前
    end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

    end_time_seconds = (end_time / 1000) + 1
    end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
    end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
    # 特定
    # end_time_string = "2023-12-31 23:59:59"


    df, trained_model_path, target_y, X, predicted = get_predict(total_klines, end_time_string, get_local_file_name)
    predicted_targets = np.argmax(predicted, axis=1)
    confidences = np.max(predicted, axis=1)


    # predicted = target_y

    # enhancement_strategy_v2(df, trained_model_path, target_y, X, predicted_targets, confidences)

    dict_target_list = []
    confidences_ragne_list = [
        [0.28, 1, 0.02], # (0.24-0.26 2%up 0.11)
        [0.28, 1, 0.02],
        [0.38, 0.9, 0.05],
        [0.35, 0.75, 0.05],
        [0.3, 0.8, 0.1],
        [0.35, 1, 0.05],
        [0.28, 1, 0.02],
        [0.28, 1, 0.02],
    ]
    # confidences_ragne_list = [
    #     [0.35, 0.45, 0.01],
    #     [0.3, 0.4, 0.01],
    #     [0.25, 0.27, 0.005],
    #     [0.23, 0.25, 0.005],
    #     [0.3, 0.38, 0.005],
    #     [0.25, 0.27, 0.005],
    #     [0.4, 0.5, 0.01],
    # ]
    for i in range(7, 8):
        for j in np.arange(0.01, 0.035, 0.005):
            confidence_range = confidences_ragne_list[i]
            start = confidence_range[0]
            end = confidence_range[1]
            step = confidence_range[2]
            max_total_profit, max_start, max_end = grid_enhancement_strategy(target_y, X, predicted_targets, confidences, i, start, end, step, j)
            dict_target_list.append({
                'target': i,
                'max_total_profit': max_total_profit,
                'max_start': max_start,
                'max_end': max_end,
                'tp': j
                # 'run_up_stop_activation': run_up_stop_activation,
                # 'run_up_constant': run_up_constant,
                # 'trailing_threshold': trailing_threshold
            })

    print(dict_target_list)

if __name__ == "__main__":
    main()



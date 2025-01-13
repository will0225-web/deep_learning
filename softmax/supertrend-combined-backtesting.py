import sys
import os

os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv, time, itertools, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime
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

def v2_customized_specific_period_col(df):
    customized_cols_infos = []
    
    # 紀錄原先的cols
    original_cols = set(df.columns)

    indicator_look_back = 288
    price_look_back = 480
    df, lower_low_higher_high_info = indicators.add_lower_low_higher_high(df, 0.04, look_back=indicator_look_back, is_need_return_function_info=True)
    customized_cols_infos.append(lower_low_higher_high_info)
    
    df, price_indicator_info = indicators.add_price_indicator(df, look_back=price_look_back, is_need_return_function_info=True)
    customized_cols_infos.append(price_indicator_info)

    df, look_back_24h_min_max_price_info = indicators.add_24h_min_max_price(df, look_back=96, is_need_return_function_info=True)
    customized_cols_infos.append(look_back_24h_min_max_price_info)

    df['EMA_9'], ema_9_info = indicators.calculate_ema(df['Close'], 9, is_need_return_function_info=True)
    customized_cols_infos.append(ema_9_info)
    df['SMA_15'], sma_15_info = indicators.calculate_sma(df['Close'], 15, is_need_return_function_info=True)
    customized_cols_infos.append(sma_15_info)
    df['SMA_30'], sma_30_info = indicators.calculate_sma(df['Close'], 30, is_need_return_function_info=True)
    customized_cols_infos.append(sma_30_info)

    # 紀錄新的cols
    modified_cols = set(df.columns)
    # 篩選多出來的cols
    new_cols = list(modified_cols - original_cols)
    return df, customized_cols_infos, new_cols

def enhancement_strategy_v2_original(df, trained_model_path, target_y, X, predicted):
    time = int(datetime.timestamp(datetime.now()))
    output_module_name = f'{time}_softmax'

    total_profit = 0
    position = 0  # 当前仓位状态，0表示无仓位，1表示多头仓位，-1表示空头仓位
    entry_price = 0  # 开仓价格
    entry_index = 0  # 开仓的索引
    entry_predicted_target = 0 #開倉predict_target
    lose_count = 0
    win_count = 0
    max_total_profit = 0
    drawdown = 0
    max_drawdown = 0
    total_winning_percentage = 0
    total_losing_percentage = 0
    sharpe_ratio = 0

    take_profit_0_1 = 0.02
    take_profit_2_3 = 0.03
    take_profit_4_5 = 0.03
    stop_loss = 0.008
    max_profit = 0

    trailing_threshold = 0.01
    trailing_return_threshold = 0.004

    detect_is_need_exit_threshold = 0.008
    if_win_at_least_profit = 0.003

    fee = 0.0010

    hold_count_limit = 20
    loss_hold_count_limit = 6
    current_loss_hold_count = 0

    # run_up_setting
    # activation 0.08,  每次都讓stop price設定為 最高或最低 / 2的位置
    run_up_stop_activation = 0.008
    run_up_constant = 35
    
    entry_confidence_0 = (0.65, 0.85)
    entry_confidence_2 = (0.5, 0.6)
    entry_confidence_4 = (0.6, 0.7)

    entry_confidence_1 = (0.55, 0.75)
    entry_confidence_3 = (0.4, 0.5)
    entry_confidence_5 = (0.45, 0.55)

    is_finish_order = False

    exit_condition_stat = {}

    for i in range(1, len(predicted)):
        # 获取原始数据集中的特定时间点的数据
        original_data = X.iloc[i]  # 假设 original_X 是DataFrame

        # 获取原始的 Close 和 RSI 值
        original_close = original_data['Close']
        original_open = original_data['Open']
        original_high = original_data['High']
        original_low = original_data['Low']
        original_datetime = original_data['datetime']

        supertrend = original_data['Super Trend']
        volume = original_data['Volume']
        kline_color = original_data['kline_color']
        
        p = predicted[i]
        predicted_target = np.argmax(p)
        confidence = np.max(predicted[i])

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

                    if current_close_profit <= 0.0:
                        current_loss_hold_count += 1
                    else:
                        current_loss_hold_count = 0

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
                    elif (max_profit >= trailing_threshold and current_close_profit <= max_profit - trailing_return_threshold):
                        # 回調的狀況先拿profit
                        profit = max_profit - trailing_return_threshold - fee
                        reason = f'Long trailing stop'
                        is_finish_order = True
                    elif (max_profit >= detect_is_need_exit_threshold and current_close_profit <= if_win_at_least_profit):
                        profit = current_close_profit - fee
                        reason = f'Long at least profit stop'
                        is_finish_order = True
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

                        if 'Runup stop Triggered' in reason:
                            string_end_index = reason.find('Triggered') + len('Triggered')
                            reason = reason[:string_end_index]
                        exit_condition_stat[reason] = exit_condition_stat.get(reason, 0) + 1

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

                    if current_close_profit <= 0:
                        current_loss_hold_count += 1
                    else:
                        current_loss_hold_count = 0

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
                    elif (max_profit >= trailing_threshold and current_close_profit <= max_profit - trailing_return_threshold):
                        # 回調的狀況先拿profit
                        profit = max_profit - trailing_return_threshold - fee
                        reason = f'Short trailing stop'
                        is_finish_order = True
                    elif (max_profit >= detect_is_need_exit_threshold and current_close_profit <= if_win_at_least_profit):
                        profit = if_win_at_least_profit - fee
                        reason = f'Short at least profit stop'
                        is_finish_order = True 
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

                        if 'Runup stop Triggered' in reason:
                            string_end_index = reason.find('Triggered') + len('Triggered')
                            reason = reason[:string_end_index]
                        exit_condition_stat[reason] = exit_condition_stat.get(reason, 0) + 1

                        if profit <= 0:
                            lose_count += 1
                            total_losing_percentage += profit * 100
                        else:
                            win_count += 1
                            total_winning_percentage += profit * 100

                        position = 0
                        max_profit = 0
                        entry_price = 0
        else:
            max_profit = 0
            current_loss_hold_count = 0
            is_finish_order = False
            # 开仓逻辑
            
            if supertrend == 1:
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
            elif supertrend == -1:
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

        max_total_profit = max(max_total_profit, total_profit)
        if max_total_profit > total_profit:
            drawdown = ((1 + max_total_profit) - (1 + total_profit)) / (1 + max_total_profit) * 100 if max_total_profit != 0 else 0
        else:
            drawdown = 0
        max_drawdown = max(max_drawdown, drawdown)

        sharpe_ratio = total_winning_percentage / abs(total_losing_percentage) if total_losing_percentage != 0 else 0                

        with open(f'{trained_model_path}/{output_module_name}_output.csv', 'a', newline='') as file:  # 'a'表示附加模式，這樣數據將會被添加到文件而不是覆蓋它
            writer = csv.writer(file)
            # 定义标题行
            headers = ["Runup Stop Price", "Entry Price", "Original Close", "Original High", "Original Low", "Win Count", "Lose Count", "Total Profit", "Max Profit", "Position", "Original Datetime", "Predicted Target", "Target", "Confidence", "Reason", "Max Total Profit", "Drawdown", "Max Drawdown", "Exit Condition Stats", "Total Winning Percentage", "Total Losing Percentage", "Sharpe Ratio"]

            # 检查文件是否为空，如果是空的，写入标题行
            if file.tell() == 0:
                writer.writerow(headers)

            # 指定原始时间戳的时区为UTC
            original_datetime_utc = original_datetime.tz_localize('UTC')

            # 转换为台湾时间（UTC+8）
            taiwan_datetime = original_datetime_utc.tz_convert('Asia/Taipei')

            # 格式化为字符串
            taiwan_datetime_str = taiwan_datetime.strftime('%Y-%m-%d %H:%M:%S')

            writer.writerow([runup_stop_price, entry_price, original_close, original_high, original_low, win_count, lose_count, total_profit, max_profit, position, taiwan_datetime_str, predicted_target, target, confidence, reason, max_total_profit, drawdown, max_drawdown, exit_condition_stat, total_winning_percentage, total_losing_percentage, sharpe_ratio])

    model_process.export_total_profit_info(df, trained_model_path, time, f'{trained_model_path}/{output_module_name}_output.csv')
    # get_max_count(f'{trained_model_path}/{output_module_name}_output.csv')

def enhancement_strategy_v2(df, trained_model_path, target_y, X, predicted, predicted_v2=None, combination=(), datetime_string=None, 
                            entry_volume_0_1_arg=None, entry_volume_2_3_arg=None, entry_volume_4_5_arg=None, hold_count_limit_arg=None,
                            toggle_limit_order=False, limit_order_rate=None, auto_cancel=None, fee_arg=None, stop_loss_arg=None,
                            entry_confidence_0_arg=None,  entry_confidence_1_arg=None, entry_confidence_2_arg=None, entry_confidence_3_arg=None, entry_confidence_4_arg=None, entry_confidence_5_arg=None,
                            toggle_entry_volume_sma=False, volume_sma_window=None, entry_volume_sma_excess=None, entry_volume_sma_excess_percentage=None,
                            grid_entry_confidence=False, save_grid_search_result=False, kline_color_check=False):
    output_module_name = f'{datetime_string}_softmax'

    total_profit = 0
    position = 0  # 目前倉位狀態，0表示無倉位，1表示多頭倉位，-1表示空頭倉位
    entry_price = 0  # 開倉價格
    entry_index = 0  # 開倉索引
    signaled_price = 0
    signaled_index = 0
    hold_position = 0
    lose_count = 0
    win_count = 0
    max_total_profit = float('-inf')
    drawdown = 0
    max_drawdown = 0
    total_winning_percentage = 0
    total_losing_percentage = 0
    sharpe_ratio = 0

    entry_volume_0_1 = entry_volume_0_1_arg if entry_volume_0_1_arg is not None else 35000
    entry_volume_2_3 = entry_volume_2_3_arg if entry_volume_2_3_arg is not None else 45000
    entry_volume_4_5 = entry_volume_4_5_arg if entry_volume_4_5_arg is not None else 45000

    take_profit = 0
    stop_loss = stop_loss_arg if stop_loss_arg is not None else 0.008
    max_profit = 0

    fee = fee_arg if fee_arg is not None else 0.0011

    hold_count_limit = hold_count_limit_arg if hold_count_limit_arg is not None else 49

    loss_hold_count_limit = 5
    current_loss_hold_count = 0

    no_keep_update_max_profit_count_limit = 4
    no_keep_update_max_profit_count = 0

    run_up_stop_activation = 0.008
    run_up_constant = 35

    entry_confidence_0 = entry_confidence_0_arg if entry_confidence_0_arg is not None else (0.65, 0.85)
    entry_confidence_2 = entry_confidence_2_arg if entry_confidence_2_arg is not None else (0.5, 0.6)
    entry_confidence_4 = entry_confidence_4_arg if entry_confidence_4_arg is not None else (0.6, 0.7)

    entry_confidence_1 = entry_confidence_1_arg if entry_confidence_1_arg is not None else (0.55, 0.75)
    entry_confidence_3 = entry_confidence_3_arg if entry_confidence_3_arg is not None else (0.4, 0.5)
    entry_confidence_5 = entry_confidence_5_arg if entry_confidence_5_arg is not None else (0.45, 0.55)

    is_finish_order = False

    for i in range(1, len(predicted)):
        # 获取原始数据集中的特定时间点的数据
        original_data = X.iloc[i]  # 假设 original_X 是DataFrame
        pre_original_data = X.iloc[i - 1]

        # 获取原始的 Close 和 RSI 值
        original_close = original_data['Close']
        original_open = original_data['Open']
        original_high = original_data['High']
        original_low = original_data['Low']
        original_datetime = pd.to_datetime(original_data['datetime'])

        pre_supertrend = pre_original_data['Super Trend']

        supertrend = original_data['Super Trend']
        volume = original_data['Volume']
        kline_color = original_data['kline_color']

        volume_sma = original_data['Volume SMA'] if toggle_entry_volume_sma == True else None

        p = predicted[i]
        predicted_target = np.argmax(p)
        confidence = np.max(predicted[i])

        # If i does not exceed the length of predicted_v2, get the predicted_v2 value
        if i < len(predicted_v2):
            p_v2 = predicted_v2[i]
            v2_target = np.argmax(p_v2)
            v2_confidence = np.max(predicted_v2[i])
        else:
            v2_target = 0
            v2_confidence = 0

        # real target
        target = np.argmax(target_y[i])
        reason = ''

        runup_stop_price = 0

        if position != 0:
            current_hold_count = i - entry_index

            if hold_position == 1:
                if examine_combination(combination, 1) and (entry_price - original_low) / entry_price >= stop_loss:
                    real_stop_loss_profit = ((original_low - entry_price) / entry_price) - fee
                    current_profit = max(real_stop_loss_profit, -stop_loss - fee)
                    profit = current_profit

                    reason = 'Long stop loss'
                    is_finish_order = True
                else:
                    current_close_profit = indicators.calculate_percentage_change(original_close, entry_price)
                    current_high_profit = indicators.calculate_percentage_change(original_high, entry_price)
                    current_low_profit = indicators.calculate_percentage_change(original_low, entry_price)

                    if max_profit >= run_up_stop_activation:
                        runup_stop_price = entry_price * ((100 - stop_loss * 100) / 100) * (1 + (max_profit / run_up_constant * current_hold_count))

                    if current_high_profit <= max_profit:
                        no_keep_update_max_profit_count += 1
                    else:
                        max_profit = max(max_profit, current_high_profit)
                        no_keep_update_max_profit_count = 0

                    if current_close_profit <= 0:
                        current_loss_hold_count += 1
                    else:
                        current_loss_hold_count = 0

                    # 結單
                    if examine_combination(combination, 2) and max_profit >= take_profit:
                        # 不超過take profit的percentage
                        profit = min(max_profit - fee, take_profit - fee)
                        reason = f'Long take profit {take_profit}' 
                        is_finish_order = True
                    elif examine_combination(combination, 3) and max_profit >= trailing_threshold and current_close_profit <= max_profit - trailing_return_threshold:
                        # 回調的狀況先拿profit
                        profit = max_profit - trailing_return_threshold - fee
                        reason = f'Long trailing stop'
                        is_finish_order = True
                    elif examine_combination(combination, 4) and no_keep_trend_threshold[0] <= max_profit <= no_keep_trend_threshold[1] and no_keep_update_max_profit_count >= no_keep_update_max_profit_count_limit:
                        profit = current_close_profit - fee
                        reason = 'Long no keep update max profit count'
                        is_finish_order = True
                    elif examine_combination(combination, 5) and current_loss_hold_count >= loss_hold_count_limit:
                        profit = current_close_profit - fee
                        reason = f'Long loss hold count'
                        is_finish_order = True
                    elif examine_combination(combination, 6) and current_hold_count >= hold_count_limit:
                        profit = current_close_profit - fee
                        reason = 'Long hold count limit'
                        is_finish_order = True
                    elif examine_combination(combination, 7) and max_profit >= run_up_stop_activation and (original_low <= runup_stop_price or original_open <= runup_stop_price):
                        actual_price = 0
                        if original_open <= runup_stop_price:
                            actual_price = original_open
                        else:
                            actual_price = original_low
                        profit = indicators.calculate_percentage_change(actual_price, entry_price) - fee
                        reason = f'Long Runup stop Triggered {profit}'
                        is_finish_order = True
                    elif examine_combination(combination, 8) and pre_supertrend == 1 and supertrend == -1:
                        profit = current_close_profit - fee
                        reason = f'Long Supertrend changed'
                        is_finish_order = True
                    # elif no_keep_update_max_profit_count < suddenly_callback_count and max_profit >= activation_percentage_threshold and (max_profit * suddenly_power_exit_threshold) >= current_close_profit:
                    #     profit = current_close_profit - fee
                    #     reason = f'Long short time suddenly callback stop'
                    #     is_finish_order = True
                    # elif (max_profit >= detect_is_need_exit_threshold and current_close_profit <= if_win_at_least_profit):
                    #     profit = current_close_profit - fee
                    #     reason = f'Long at least profit stop'
                        # is_finish_order = True

                if is_finish_order:
                    total_profit += profit

                    if profit <= 0:
                        lose_count += 1
                        total_losing_percentage += profit * 100
                    else:
                        win_count += 1
                        total_winning_percentage += profit * 100

                    position = 0
                    hold_position = 0
                    max_profit = 0
                    entry_price = 0
                    signaled_price = 0
                    take_profit = 0
                    trailing_threshold = 0
                    trailing_return_threshold = 0
                    no_keep_trend_threshold = [0, 0]
                    current_loss_hold_count = 0
                    no_keep_update_max_profit_count = 0

                    is_finish_order = False

            elif hold_position == -1:
                if examine_combination(combination, 1) and (original_high - entry_price) / entry_price >= stop_loss:
                    real_stop_loss_profit = ((entry_price - original_high) / entry_price) - fee
                    current_profit = max(real_stop_loss_profit, -stop_loss - fee)
                    profit = current_profit

                    reason = 'Short stop loss'
                    is_finish_order = True
                else:
                    current_close_profit = -indicators.calculate_percentage_change(original_close, entry_price)
                    current_high_profit = -indicators.calculate_percentage_change(original_high, entry_price)
                    current_low_profit = -indicators.calculate_percentage_change(original_low, entry_price)

                    if max_profit >= run_up_stop_activation:
                        runup_stop_price = entry_price * ((100 + stop_loss * 100) / 100) * (1 - (max_profit / run_up_constant * current_hold_count))

                    if current_low_profit <= max_profit:
                        no_keep_update_max_profit_count += 1
                    else:
                        max_profit = max(max_profit, current_low_profit)
                        no_keep_update_max_profit_count = 0

                    if current_close_profit <= 0:
                        current_loss_hold_count += 1
                    else:
                        current_loss_hold_count = 0

                    # 結單
                    if examine_combination(combination, 2) and max_profit >= take_profit:
                        # 不超過take profit的percentage
                        profit = min(max_profit - fee, take_profit - fee)
                        reason = f'Short take profit {take_profit}'
                        is_finish_order = True
                    elif examine_combination(combination, 3) and max_profit >= trailing_threshold and current_close_profit <= max_profit - trailing_return_threshold:
                        # 回調的狀況先拿profit
                        profit = max_profit - trailing_return_threshold - fee
                        reason = f'Short trailing stop'
                        is_finish_order = True
                    elif examine_combination(combination, 4) and no_keep_trend_threshold[0] <= max_profit <= no_keep_trend_threshold[1] and no_keep_update_max_profit_count >= no_keep_update_max_profit_count_limit:
                        profit = current_close_profit - fee
                        reason = 'Short no keep update max profit count'
                        is_finish_order = True
                    elif examine_combination(combination, 5) and current_loss_hold_count >= loss_hold_count_limit:
                        profit = current_close_profit - fee
                        reason = f'Short loss hold count'
                        is_finish_order = True
                    elif examine_combination(combination, 6) and current_hold_count >= hold_count_limit:
                        profit = current_close_profit - fee
                        reason = 'Short hold count limit'
                        is_finish_order = True
                    elif examine_combination(combination, 7) and max_profit >= run_up_stop_activation and (original_high >= runup_stop_price or original_open >= runup_stop_price):
                        actual_price = 0
                        if original_open >= runup_stop_price:
                            actual_price = original_open
                        else:
                            actual_price = original_high
                        profit = -indicators.calculate_percentage_change(actual_price, entry_price) - fee
                        reason = f'Short Runup stop Triggered {profit}'
                        is_finish_order = True
                    elif examine_combination(combination, 8) and pre_supertrend == -1 and supertrend == 1:
                        profit = current_close_profit - fee
                        reason = f'Short Supertrend changed'
                        is_finish_order = True
                    # elif (no_keep_update_max_profit_count < suddenly_callback_count and max_profit >= activation_percentage_threshold and (max_profit * suddenly_power_exit_threshold) >= current_close_profit):
                    #     profit = current_close_profit - fee
                    #     reason = f'Short short time suddenly callback stop'
                    #     is_finish_order = True
                    # elif (max_profit >= detect_is_need_exit_threshold and current_close_profit <= if_win_at_least_profit):
                    #     profit = if_win_at_least_profit - fee
                    #     reason = f'Short at least profit stop'
                        # is_finish_order = True 

                if is_finish_order:
                    total_profit += profit

                    if profit <= 0:
                        lose_count += 1
                        total_losing_percentage += profit * 100
                    else:
                        win_count += 1
                        total_winning_percentage += profit * 100

                    position = 0
                    hold_position = 0
                    max_profit = 0
                    entry_price = 0
                    signaled_price = 0
                    take_profit = 0
                    trailing_threshold = 0
                    trailing_return_threshold = 0
                    no_keep_trend_threshold = [0, 0]
                    current_loss_hold_count = 0
                    no_keep_update_max_profit_count = 0

                    is_finish_order = False

            elif hold_position == 0:
                since_signal_count = i - signaled_index
                if since_signal_count > auto_cancel:
                    position = 0
                    entry_price = 0
                    take_profit = 0
                    trailing_threshold = 0
                    trailing_return_threshold = 0
                    no_keep_trend_threshold = [0, 0]
                else:
                    if position == 1:
                        if original_low <= entry_price * (1 - stop_loss):
                            real_stop_loss_profit = ((original_low - entry_price) / entry_price) - fee
                            current_profit = max(real_stop_loss_profit, -stop_loss - fee)
                            profit = current_profit

                            reason = 'Long immediate stop loss'

                            total_profit += profit
                            lose_count += 1
                            total_losing_percentage += profit * 100

                            position = 0
                            max_profit = 0
                            entry_price = 0
                            signaled_price = 0
                            take_profit = 0
                            trailing_threshold = 0
                            trailing_return_threshold = 0
                            no_keep_trend_threshold = [0, 0]

                        elif original_low <= entry_price:
                            hold_position = 1
                            entry_index = i

                    elif position == -1:
                        if original_high >= entry_price * (1 + stop_loss):
                            real_stop_loss_profit = ((entry_price - original_high) / entry_price) - fee
                            current_profit = max(real_stop_loss_profit, -stop_loss - fee)
                            profit = current_profit

                            reason = 'Short immediate stop loss'

                            total_profit += profit
                            lose_count += 1
                            total_losing_percentage += profit * 100

                            position = 0
                            max_profit = 0
                            entry_price = 0
                            signaled_price = 0
                            take_profit = 0
                            trailing_threshold = 0
                            trailing_return_threshold = 0
                            no_keep_trend_threshold = [0, 0]

                        elif original_high >= entry_price:
                            hold_position = -1
                            entry_index = i

        else:
            # 開倉邏輯
            if supertrend == 1 and check_kline_color(kline_color_check, 'Long', kline_color):
                if (predicted_target == 0 and entry_confidence_0[0] <= confidence <= entry_confidence_0[1]) and check_entry_volume(volume, entry_volume_0_1, toggle_entry_volume_sma, volume_sma, entry_volume_sma_excess, entry_volume_sma_excess_percentage):
                    position = 1
                elif (predicted_target == 2 and entry_confidence_2[0] <= confidence <= entry_confidence_2[1]) and check_entry_volume(volume, entry_volume_2_3, toggle_entry_volume_sma, volume_sma, entry_volume_sma_excess, entry_volume_sma_excess_percentage):
                    position = 1
                elif (predicted_target == 4 and entry_confidence_4[0] <= confidence <= entry_confidence_4[1]) and check_entry_volume(volume, entry_volume_4_5, toggle_entry_volume_sma, volume_sma, entry_volume_sma_excess, entry_volume_sma_excess_percentage):
                    position = 1
            elif supertrend == -1 and check_kline_color(kline_color_check, 'Short', kline_color):
                if (predicted_target == 1 and entry_confidence_1[0] <= confidence <= entry_confidence_1[1]) and check_entry_volume(volume, entry_volume_0_1, toggle_entry_volume_sma, volume_sma, entry_volume_sma_excess, entry_volume_sma_excess_percentage):
                    position = -1
                elif (predicted_target == 3 and entry_confidence_3[0] <= confidence <= entry_confidence_3[1]) and check_entry_volume(volume, entry_volume_2_3, toggle_entry_volume_sma, volume_sma, entry_volume_sma_excess, entry_volume_sma_excess_percentage):
                    position = -1
                elif (predicted_target == 5 and entry_confidence_5[0] <= confidence <= entry_confidence_5[1]) and check_entry_volume(volume, entry_volume_4_5, toggle_entry_volume_sma, volume_sma, entry_volume_sma_excess, entry_volume_sma_excess_percentage):
                    position = -1

            # Set take profit and trailing stop accordingly
            if position != 0:
                if toggle_limit_order == False:
                    hold_position = position
                    entry_price = original_close
                    entry_index = i
                else:
                    signaled_price = original_close
                    signaled_index = i
                    entry_price = signaled_price * (1 - limit_order_rate) if position == 1 else signaled_price * (1 + limit_order_rate)

                take_profit = set_hybrid_tp(predicted_target, v2_target, v2_confidence)

                trailing_threshold = take_profit / 1.5
                trailing_return_threshold = trailing_threshold / 2
                if trailing_return_threshold <= 0.004:
                    trailing_return_threshold = 0.004
                elif trailing_return_threshold >= 0.008:
                    trailing_return_threshold = 0.008

                no_keep_trend_threshold = [0.003, trailing_threshold]

        max_total_profit = max(max_total_profit, total_profit)
        if max_total_profit > total_profit:
            drawdown = ((1 + max_total_profit) - (1 + total_profit)) / (1 + max_total_profit) * 100 if max_total_profit != 0 else 0
        else:
            drawdown = 0
        max_drawdown = max(max_drawdown, drawdown)

        total_trade = win_count + lose_count
        win_rate = win_count / total_trade * 100 if (win_count + lose_count) != 0 else 0

        sharpe_ratio = total_winning_percentage / abs(total_losing_percentage) if total_losing_percentage != 0 else 0

        if save_grid_search_result == False:
            with open(f'{trained_model_path}/{output_module_name}_output.csv', 'a', newline='') as file:  # 'a'表示附加模式，這樣數據將會被添加到文件而不是覆蓋它
                writer = csv.writer(file)
                # 定义标题行
                headers = ["Original Datetime", "Position", "Hold Position", "Entry Price", "Reason", "Runup Stop Price", "Max Profit", "Original Close", "Original High", "Original Low", "Volume", "Total Profit", "Max Total Profit", "Drawdown", "Max Drawdown", "Predicted Target", "Target", "Confidence", "Win Count", "Lose Count", "Total trade", "Win Rate", "Total Winning Percentage", "Total Losing Percentage", "Sharpe Ratio"]

                # 检查文件是否为空，如果是空的，写入标题行
                if file.tell() == 0:
                    writer.writerow(headers)


                # 指定原始时间戳的时区为UTC
                original_datetime_utc = original_datetime.tz_localize('UTC')

                # 转换为台湾时间（UTC+8）
                taiwan_datetime = original_datetime_utc.tz_convert('Asia/Taipei')

                # 格式化为字符串
                taiwan_datetime_str = taiwan_datetime.strftime('%Y-%m-%d %H:%M:%S')

                writer.writerow([taiwan_datetime_str, position, hold_position, entry_price, reason, runup_stop_price, max_profit, original_close, original_high, original_low, volume, total_profit, max_total_profit, drawdown, max_drawdown, predicted_target, target, confidence, win_count, lose_count, total_trade, win_rate, total_winning_percentage, total_losing_percentage, sharpe_ratio])

    # 配置输出CSV文件
    if save_grid_search_result == True:
        with open(f'{datetime_string}_grid_search_results.csv', 'a', newline='') as file:
            writer = csv.writer(file)

            if file.tell() == 0:
                # Essential performance metrics
                headers = ['Total Profit', 'Max Total Profit', 'Drawdown', 'Max Drawdown', 'Total Trade', 'Win Rate']

                # Add different entry volume conditions to headers
                if toggle_entry_volume_sma == False and (entry_volume_0_1_arg is not None or entry_volume_2_3_arg is not None or entry_volume_4_5_arg is not None):
                    headers = ['entry_volume_0_1', 'entry_volume_2_3', 'entry_volume_4_5'] + headers
                elif toggle_entry_volume_sma == True and entry_volume_sma_excess is not None:
                    headers = ['volume_sma_window', 'entry_volume_sma_excess'] + headers
                elif toggle_entry_volume_sma == True and entry_volume_sma_excess_percentage is not None:
                    headers = ['volume_sma_window', 'entry_volume_sma_excess_percentage'] + headers

                # Add hold count limit to headers
                index = headers.index('Total Profit')
                headers.insert(index, 'Hold Count Limit')

                # Add stop loss to headers
                if stop_loss_arg is not None:
                    index = headers.index('Total Profit')
                    headers.insert(index, 'Stop Loss')

                # Add entry confidence conditions to headers
                if grid_entry_confidence == True:
                    index = headers.index('Total Profit')
                    headers[index:1] = ['entry_confidence_0', 'entry_confidence_1', 'entry_confidence_2', 'entry_confidence_3', 'entry_confidence_4', 'entry_confidence_5']

                if toggle_limit_order == True:
                    index = headers.index('Total Profit')
                    headers.insert(index, 'Limit Order Rate')
                    headers.insert(index + 1, 'Auto Cancel')

                writer.writerow(headers)

            # Add essential performance metrics and entry volume conditions to output string and row
            if toggle_entry_volume_sma == False and (entry_volume_0_1_arg is not None or entry_volume_2_3_arg is not None or entry_volume_4_5_arg is not None):
                output_string = f'entry_volume_0_1: {entry_volume_0_1}, entry_volume_2_3: {entry_volume_2_3}, entry_volume_4_5: {entry_volume_4_5}, Total Profit: {total_profit}, Max Total Profit: {max_total_profit}, Drawdown: {drawdown}, Max Drawdown: {max_drawdown}, Total Trade: {total_trade}, Win Rate: {win_rate}'
                row = [entry_volume_0_1, entry_volume_2_3, entry_volume_4_5, total_profit, max_total_profit, drawdown, max_drawdown, total_trade, win_rate]
            elif toggle_entry_volume_sma == True and entry_volume_sma_excess is not None:
                output_string = f'volume_sma_window: {volume_sma_window}, entry_volume_sma_excess: {entry_volume_sma_excess}, Total Profit: {total_profit}, Max Total Profit: {max_total_profit}, Drawdown: {drawdown}, Max Drawdown: {max_drawdown}, Total Trade: {total_trade}, Win Rate: {win_rate}'
                row = [volume_sma_window, entry_volume_sma_excess, total_profit, max_total_profit, drawdown, max_drawdown, total_trade, win_rate]
            elif toggle_entry_volume_sma == True and entry_volume_sma_excess_percentage is not None:
                output_string = f'volume_sma_window: {volume_sma_window}, entry_volume_sma_excess_percentage: {entry_volume_sma_excess_percentage}, Total Profit: {total_profit}, Max Total Profit: {max_total_profit}, Drawdown: {drawdown}, Max Drawdown: {max_drawdown}, Total Trade: {total_trade}, Win Rate: {win_rate}'
                row = [volume_sma_window, entry_volume_sma_excess_percentage, total_profit, max_total_profit, drawdown, max_drawdown, total_trade, win_rate]

            # Add hold count limit to output string and row
            parts = output_string.split(', ')
            total_profit_index = next(i for i, part in enumerate(parts) if 'Total Profit' in part)
            parts.insert(total_profit_index, f'Hold Count Limit: {hold_count_limit}')
            output_string = ', '.join(parts)

            row.insert(total_profit_index, hold_count_limit)

            if stop_loss_arg is not None:
                parts = output_string.split(', ')
                total_profit_index = next(i for i, part in enumerate(parts) if 'Total Profit' in part)
                parts.insert(total_profit_index, f'Stop Loss: {stop_loss}')
                output_string = ', '.join(parts)

                row.insert(total_profit_index, stop_loss)

            # Add entry confidence conditions to output string and row
            if grid_entry_confidence == True:
                parts = output_string.split(', ')
                total_profit_index = next(i for i, part in enumerate(parts) if 'Total Profit' in part)
                parts.insert(total_profit_index, f'entry_confidence_0: {entry_confidence_0}, entry_confidence_1: {entry_confidence_1}, entry_confidence_2: {entry_confidence_2}, entry_confidence_3: {entry_confidence_3}, entry_confidence_4: {entry_confidence_4}, entry_confidence_5: {entry_confidence_5}')
                output_string = ', '.join(parts)

                row.insert(total_profit_index, entry_confidence_0)
                row.insert(total_profit_index + 1, entry_confidence_1)
                row.insert(total_profit_index + 2, entry_confidence_2)
                row.insert(total_profit_index + 3, entry_confidence_3)
                row.insert(total_profit_index + 4, entry_confidence_4)
                row.insert(total_profit_index + 5, entry_confidence_5)

            if toggle_limit_order == True:
                parts = output_string.split(', ')
                total_profit_index = next(i for i, part in enumerate(parts) if 'Total Profit' in part)
                parts.insert(total_profit_index, f'Limit Order Rate: {limit_order_rate}, Auto Cancel: {auto_cancel}')
                output_string = ', '.join(parts)

                row.insert(total_profit_index, limit_order_rate)
                row.insert(total_profit_index + 1, auto_cancel)

            print(output_string)

            writer.writerow(row)

    elif save_grid_search_result == False:
        model_process.export_total_profit_info(df, trained_model_path, datetime_string, f'{trained_model_path}/{output_module_name}_output.csv')

def grid_enhancement_strategy_original(target_y, X, predicted_targets, confidences, grid_target, start_range=None, end_range=None, 
                                       step=None, tp=None, datetime_string=None):
    max_total_profit = float('-inf')

    total_profit = 0
    position = 0  # 当前仓位状态，0表示无仓位，1表示多头仓位，-1表示空头仓位
    entry_price = 0  # 開倉價格
    entry_index = 0  # 開倉索引
    entry_predicted_target = 0 # 開倉predict_target
    lose_count = 0
    win_count = 0
    max_total_profit = 0
    drawdown = 0
    max_drawdown = 0
    total_winning_percentage = 0
    total_losing_percentage = 0

    take_profit_0_1 = 0.025
    take_profit_2_3 = 0.03
    take_profit_4_5 = 0.03

    stop_loss = 0.008
    max_profit = 0

    fee = 0.0004

    hold_count_limit = 20

    loss_hold_count_limit = 6
    current_loss_hold_count = 0

    run_up_stop_activation = 0.008
    run_up_constant = 35

    entry_confidence_0 = (0.65, 0.85)
    entry_confidence_2 = (0.5, 0.6)
    entry_confidence_4 = (0.6, 0.7)

    entry_confidence_1 = (0.55, 0.75)
    entry_confidence_3 = (0.4, 0.5)
    entry_confidence_5 = (0.45, 0.55)

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

        pre_supertrend = pre_original_data['Super Trend']

        supertrend = original_data['Super Trend']
        volume = original_data['Volume']
        kline_color = original_data['kline_color']
        
        predicted_target = predicted_targets[i]
        confidence = confidences[i]

        runup_stop_price = 0

        if position != 0:
            if entry_predicted_target == grid_target:
                take_profit = tp
            else:
                if entry_predicted_target == 0 or entry_predicted_target == 1:
                    take_profit = take_profit_0_1
                elif entry_predicted_target == 2 or entry_predicted_target == 3:
                    take_profit = take_profit_2_3
                elif entry_predicted_target == 4 or entry_predicted_target == 5:
                    take_profit = take_profit_4_5

            current_hold_count = i - entry_index

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

                    if current_close_profit <= 0:
                        current_loss_hold_count += 1
                    else:
                        current_loss_hold_count = 0

                    if current_low_profit <= max_profit:
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
                            total_losing_percentage += profit * 100
                        else:
                            win_count += 1
                            total_winning_percentage += profit * 100

                        position = 0
                        max_profit = 0
                        entry_price = 0
        else:
            max_profit = 0
            current_loss_hold_count = 0
            no_keep_update_max_profit_count = 0
            is_finish_order = False
            # 开仓逻辑
            
            if supertrend == 1:
                if (predicted_target == 0 and entry_confidence_0[0] <= confidence <= entry_confidence_0[1]) and volume >= 35000 and kline_color == 0:
                    position = 1

                    entry_price = original_close
                    entry_index = i

                    entry_predicted_target = predicted_target
                elif predicted_target == 2 and entry_confidence_2[0] <= confidence <= entry_confidence_2[1] and volume >= 45000 and kline_color == 0:
                    position = 1

                    entry_price = original_close
                    entry_index = i

                    entry_predicted_target = predicted_target
                elif predicted_target == 4 and entry_confidence_4[0] <= confidence <= entry_confidence_4[1] and volume >= 45000 and kline_color == 0:
                    position = 1
                    entry_price = original_close
                    entry_index = i

                    entry_predicted_target = predicted_target
            elif supertrend == -1:
                if (predicted_target == 1 and entry_confidence_1[0] <= confidence <= entry_confidence_1[1]) and volume >= 35000 and kline_color == 1:
                    position = -1
                    entry_price = original_close
                    entry_index = i
                    
                    entry_predicted_target = predicted_target
                elif (predicted_target == 3 and entry_confidence_3[0] <= confidence <= entry_confidence_3[1]) and volume >= 45000 and kline_color == 1:
                    position = -1
                    entry_price = original_close
                    entry_index = i
                    
                    entry_predicted_target = predicted_target
                elif (predicted_target == 5 and entry_confidence_5[0] <= confidence <= entry_confidence_5[1]) and volume >= 45000 and kline_color == 1:
                    position = -1
                    entry_price = original_close
                    entry_index = i
                    
                    entry_predicted_target = predicted_target
    # print(grid_target, total_profit, start, end, run_up_stop_activation, run_up_constant, trailing_threshold)

    # 配置输出CSV文件
    with open(f'{datetime_string}_grid_search_results.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        headers = ['Grid Target', 'Total Profit', 'TP']

        if file.tell() == 0:
            writer.writerow(headers)

        writer.writerow([grid_target, total_profit, tp])

    print(f'grid_target: {grid_target}, total_profit: {total_profit}, tp: {tp}')
    if max_total_profit < total_profit:
        max_total_profit = total_profit

    return max_total_profit

def grid_enhancement_strategy(target_y, X, predicted_targets, confidences, grid_target=None, start_range=None, end_range=None, 
                              step=None, tp=None, datetime_string=None, predicted_targets_v2=None, 
                              confidences_v2=None, hold_count_limit=None, no_keep_update_max_profit_count_limit=None):
    print(f'predicted_length: {len(predicted_targets)}')
    print(f'predicted_v2_length: {len(predicted_targets_v2)}')

    total_profit = 0
    position = 0  # 当前仓位状态，0表示无仓位，1表示多头仓位，-1表示空头仓位
    entry_price = 0  # 開倉價格
    entry_index = 0  # 開倉索引
    entry_predicted_target = 0 # 開倉predict_target
    entry_v2_target = 0
    entry_v2_confidence = 0
    lose_count = 0
    win_count = 0
    max_total_profit = float('-inf')
    drawdown = 0
    max_drawdown = 0
    total_winning_percentage = 0
    total_losing_percentage = 0

    take_profit_0_1 = 0.025
    take_profit_2_3 = 0.03
    take_profit_4_5 = 0.03

    take_profit = 0
    stop_loss = 0.008
    max_profit = float('-inf')

    fee = 0.0004

    hold_count_limit = 49

    loss_hold_count_limit = 5
    current_loss_hold_count = 0

    no_keep_update_max_profit_count_limit = no_keep_update_max_profit_count_limit
    no_keep_update_max_profit_count = 0

    entry_confidence_0 = (0.65, 0.85)
    entry_confidence_2 = (0.5, 0.6)
    entry_confidence_4 = (0.6, 0.7)

    entry_confidence_1 = (0.55, 0.75)
    entry_confidence_3 = (0.4, 0.5)
    entry_confidence_5 = (0.45, 0.55)

    is_finish_order = False
    for i in range(1, len(predicted_targets)):
        # 获取原始数据集中的特定时间点的数据
        original_data = X.iloc[i]  # 假设 original_X 是DataFrame
        pre_original_data = X.iloc[i - 1]

        # 获取原始的 Close 和 RSI 值
        original_close = original_data['Close']
        original_high = original_data['High']
        original_low = original_data['Low']

        supertrend = original_data['Super Trend']
        volume = original_data['Volume']
        kline_color = original_data['kline_color']
        
        predicted_target = predicted_targets[i]
        confidence = confidences[i]

        # If i does not exceed the length of predicted_v2, get the predicted_v2 value
        if i < len(predicted_targets_v2):
            v2_target = predicted_targets_v2[i]
            v2_confidence = confidences_v2[i]
        else:
            v2_target = 0
            v2_confidence = 0

        if position != 0:
            if entry_predicted_target == grid_target:
                take_profit = tp
            else:
                if entry_v2_target == 1:
                    if 0.39 <= entry_v2_confidence <= 0.44:
                        take_profit = 0.015 # 或者0.01
                    elif 0.23 <= entry_v2_confidence <= 0.26:
                        take_profit = 0.03
                elif entry_v2_target == 2: # 需調整
                    # if 0.2 <= entry_v2_confidence <= 0.24:
                    #     take_profit = 0.015
                    # elif 0.32 <= entry_v2_confidence <= 0.37:
                    #     take_profit = 0.02
                    if 0.28 <= entry_v2_confidence <= 0.29:
                        take_profit = 0.015 # 或者0.01
                    elif 0.32 <= entry_v2_confidence <= 0.33:
                        take_profit = 0.02
                    elif 0.2 <= entry_v2_confidence <= 0.23:
                        take_profit = 0.025
                elif entry_v2_target == 3:
                    # if 0.252 <= entry_v2_confidence <= 0.28:
                    #     take_profit = 0.015
                    if 0.256 <= entry_v2_confidence <= 0.26:
                        take_profit = 0.015 # 或者0.01
                    elif 0.274 <= entry_v2_confidence <= 0.278:
                        take_profit = 0.02
                    elif 0.262 <= entry_v2_confidence <= 0.27:
                        take_profit = 0.025
                elif entry_v2_target == 4 and not(0.255 <= entry_v2_confidence <= 0.26):
                    if 0.245 <= entry_v2_confidence <= 0.25:
                        take_profit = 0.015
                    elif 0.255 <= entry_v2_confidence <= 0.265:
                        take_profit = 0.01
                    elif 0.315 <= entry_v2_confidence <= 0.32:
                        take_profit = 0.015
                elif entry_v2_target == 5: # 需調整
                    # if 0.34 <= entry_v2_confidence <= 0.37:
                    #     take_profit = 0.025
                    if 0.36 <= entry_v2_confidence <= 0.37:
                        take_profit = 0.015
                    elif 0.32 <= entry_v2_confidence <= 0.33:
                        take_profit = 0.02
                    elif 0.34 <= entry_v2_confidence <= 0.35:
                        take_profit = 0.025
                elif entry_v2_target == 6:
                    if 0.243 <= entry_v2_confidence <= 0.265:
                        take_profit = 0.015
                    elif 0.251 <= entry_v2_confidence <= 0.253:
                        take_profit = 0.02
                elif entry_v2_target == 7 and not(0.47 <= entry_v2_confidence <= 0.475 or 0.39 <= entry_v2_confidence <= 0.44): # 7有大回撤點
                    # if 0.37 <= entry_v2_confidence <= 0.49:
                    #     take_profit = 0.01 # 或0.01(0.01比較保險)
                    # elif 0.45 <= entry_v2_confidence <= 0.55:
                    #     take_profit = 0.025
                    if 0.46 <= entry_v2_confidence <= 0.49:
                        take_profit = 0.025 # 或0.03
                    elif 0.37 <= entry_v2_confidence <= 0.49:
                        take_profit = 0.015 # 或0.01(0.01比較保險)
                elif entry_v2_target == 8:
                    # if 0.4 <= entry_v2_confidence <= 0.53:
                    #     take_profit = 0.015
                    if 0.49 <= entry_v2_confidence <= 0.55:
                        take_profit = 0.01
                    elif 0.4 <= entry_v2_confidence <= 0.43:
                        take_profit = 0.015
                    elif 0.45 <= entry_v2_confidence <= 0.48:
                        take_profit = 0.03 # 或 0.03
                    elif 0.45 <= entry_v2_confidence <= 0.53:
                        take_profit = 0.025
                elif entry_v2_target == 9:
                    if 0.2 <= entry_v2_confidence <= 0.27:
                        take_profit = 0.02
                    # if 0.23 <= entry_v2_confidence <= 0.26:
                    #     take_profit = 0.02
                elif entry_v2_target == 10:
                    # if 0.23 <= entry_v2_confidence <= 0.255:
                    #     take_profit = 0.01
                    if 0.23 <= entry_v2_confidence <= 0.245:
                        take_profit = 0.01

                if take_profit == 0:
                    if entry_predicted_target == 0 or entry_predicted_target == 1:
                        take_profit = take_profit_0_1
                    elif entry_predicted_target == 2 or entry_predicted_target == 3:
                        take_profit = take_profit_2_3
                    elif entry_predicted_target == 4 or entry_predicted_target == 5:
                        take_profit = take_profit_4_5

            trailing_threshold = take_profit / 1.5
            trailing_return_threshold = trailing_threshold / 2
            if trailing_return_threshold <= 0.004:
                trailing_return_threshold = 0.004

            no_keep_trend_threshold = [0.003, trailing_threshold]

            current_hold_count = i - entry_index

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

                    if current_high_profit <= max_profit:
                        no_keep_update_max_profit_count += 1
                    else:
                        max_profit = max(max_profit, current_high_profit)
                        no_keep_update_max_profit_count = 0

                    if current_close_profit <= 0.0:
                        current_loss_hold_count += 1
                    else:
                        current_loss_hold_count = 0

                    # 結單
                    if max_profit >= take_profit:
                        # 不超過take profit的percentage
                        profit = min(max_profit - fee, take_profit - fee)
                        reason = f'Long take profit {take_profit}' 
                        is_finish_order = True
                    elif (max_profit >= trailing_threshold and current_close_profit <= max_profit - trailing_return_threshold):
                        # 回調的狀況先拿profit
                        profit = max_profit - trailing_return_threshold - fee
                        reason = f'Long trailing stop'
                        is_finish_order = True
                    # elif (current_loss_hold_count >= loss_hold_count_limit):
                    #     profit = current_close_profit - fee
                    #     reason = f'Long loss hold count'
                    #     is_finish_order = True
                    elif current_hold_count >= hold_count_limit:
                        profit = current_close_profit - fee
                        reason = 'Long hold count limit'
                        is_finish_order = True
                    elif no_keep_trend_threshold[0] <= max_profit <= no_keep_trend_threshold[1] and no_keep_update_max_profit_count >= no_keep_update_max_profit_count_limit:
                        profit = current_close_profit - fee
                        reason = 'Long no keep update max profit count'
                        is_finish_order = True

                    
                    if is_finish_order:
                        total_profit += profit
                        
                        if profit <= 0:
                            lose_count += 1
                            total_losing_percentage += profit * 100
                        else:
                            win_count += 1
                            total_winning_percentage += profit * 100
                        
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

                    if current_low_profit <= max_profit:
                        no_keep_update_max_profit_count += 1
                    else:
                        max_profit = max(max_profit, current_low_profit)
                        no_keep_update_max_profit_count = 0

                    if current_close_profit <= 0:
                        current_loss_hold_count += 1
                    else:
                        current_loss_hold_count = 0

                    # 結單
                    if max_profit >= take_profit:
                        # 不超過take profit的percentage
                        profit = min(max_profit - fee, take_profit - fee)
                        reason = f'Short take profit {take_profit}'
                        is_finish_order = True
                    elif (max_profit >= trailing_threshold and current_close_profit <= max_profit - trailing_return_threshold):
                        # 回調的狀況先拿profit
                        profit = max_profit - trailing_return_threshold - fee
                        reason = f'Short trailing stop'
                        is_finish_order = True
                    # elif (current_loss_hold_count >= loss_hold_count_limit):
                    #     profit = current_close_profit - fee
                    #     reason = f'Short loss hold count'
                    #     is_finish_order = True
                    elif current_hold_count >= hold_count_limit:
                        profit = current_close_profit - fee
                        reason = 'Short hold count limit'
                        is_finish_order = True
                    elif no_keep_trend_threshold[0] <= max_profit <= no_keep_trend_threshold[1] and no_keep_update_max_profit_count >= no_keep_update_max_profit_count_limit:
                        profit = current_close_profit - fee
                        reason = 'Short no keep update max profit count'
                        is_finish_order = True

                    if is_finish_order:
                        total_profit += profit

                        if profit <= 0:
                            lose_count += 1
                            total_losing_percentage += profit * 100
                        else:
                            win_count += 1
                            total_winning_percentage += profit * 100

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
                    entry_v2_target = v2_target
                    entry_v2_confidence = v2_confidence
                elif predicted_target == 2 and entry_confidence_2[0] <= confidence <= entry_confidence_2[1] and volume >= 45000:
                    position = 1

                    entry_price = original_close
                    entry_index = i

                    entry_predicted_target = predicted_target
                    entry_v2_target = v2_target
                    entry_v2_confidence = v2_confidence
                elif predicted_target == 4 and entry_confidence_4[0] <= confidence <= entry_confidence_4[1] and volume >= 45000:
                    position = 1
                    entry_price = original_close
                    entry_index = i

                    entry_predicted_target = predicted_target
                    entry_v2_target = v2_target
                    entry_v2_confidence = v2_confidence
            elif supertrend == -1 and kline_color == 1:
                if (predicted_target == 1 and entry_confidence_1[0] <= confidence <= entry_confidence_1[1]) and volume >= 35000:
                    position = -1
                    entry_price = original_close
                    entry_index = i
                    
                    entry_predicted_target = predicted_target
                    entry_v2_target = v2_target
                    entry_v2_confidence = v2_confidence
                elif (predicted_target == 3 and entry_confidence_3[0] <= confidence <= entry_confidence_3[1]) and volume >= 45000:
                    position = -1
                    entry_price = original_close
                    entry_index = i
                    
                    entry_predicted_target = predicted_target
                    entry_v2_target = v2_target
                    entry_v2_confidence = v2_confidence
                elif (predicted_target == 5 and entry_confidence_5[0] <= confidence <= entry_confidence_5[1]) and volume >= 45000:
                    position = -1
                    entry_price = original_close
                    entry_index = i
                    
                    entry_predicted_target = predicted_target
                    entry_v2_target = v2_target
                    entry_v2_confidence = v2_confidence

        max_total_profit = max(max_total_profit, total_profit)
        if max_total_profit > total_profit:
            drawdown = ((1 + max_total_profit) - (1 + total_profit)) / (1 + max_total_profit) * 100 if max_total_profit != 0 else 0
        else:
            drawdown = 0
        max_drawdown = max(max_drawdown, drawdown)

    # print(grid_target, total_profit, start, end, run_up_stop_activation, run_up_constant, trailing_threshold)

    total_trade = win_count + lose_count
    win_rate = win_count / total_trade * 100 if (win_count + lose_count) != 0 else 0

    # 配置输出CSV文件
    with open(f'{datetime_string}_grid_search_results.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        headers = ['No Keep Update Max Profit Count Limit', 'Total Profit', 'Max Total Profit', 'Drawdown', 'Max Drawdown', 'Total Trade', 'Win Rate']

        if file.tell() == 0:
            writer.writerow(headers)

        writer.writerow([no_keep_update_max_profit_count_limit, total_profit, max_total_profit, drawdown, max_drawdown, total_trade, win_rate])

    print(f'no_keep_update_max_profit_count_limit: {no_keep_update_max_profit_count_limit}, total_profit: {total_profit}, max_total_profit: {max_total_profit}, drawdown: {drawdown}, max_drawdown: {max_drawdown}, total_trade: {total_trade}, win_rate: {win_rate}')
    if max_total_profit < total_profit:
        max_total_profit = total_profit

    return max_total_profit

def get_original_data(model_connection_info, model):
    symbol = model_connection_info['Symbol']
    interval = model_connection_info['Interval']
    get_local_file_name = model_connection_info['Get Local File Name']
    total_klines = model_connection_info['Get Original Klines Count']
    end_time_string = model_connection_info['End Time']
    drop_front_data_count = model_connection_info['Drop Front Data Count']
    drop_back_data_count = model_connection_info['Drop Back Data Count']

    original_df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)

    if model == 'v2':
        original_df, customized_cols_infos, new_cols = v2_customized_specific_period_col(original_df)
        original_df['Target'] = signals.super_trend_strategy_v3(original_df, 0.008, 0.008, 0.003, 12)
    else:
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

        scaled, target_y, _ = transform_and_data_info(df, y, cols, robust_features, standard_features, minMax_features, front_drop_count, back_drop_count, look_back, original_X)
        input_model_X_datas.append(scaled)
        input_model_Y_datas.append(target_y)
        
    return input_model_X_datas, input_model_Y_datas



def multiclass_f1_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_f1_score(y_true, y_pred)

def get_predict(total_klines, end_time_string, get_local_file_name, need_v2_predict=False):
    # 載入已經訓練好的模型
    # 獲取當前文件的絕對路徑
    current_path = os.path.abspath(os.path.dirname(__file__))
    parent_path = os.path.join(current_path, '..')
    trained_model_path = os.path.join(parent_path, 'trained_models/1710835373_softmax_loss-1.2277_accuracy-0.4738_f1-0.2171_real_best')
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
    original_X, _ = get_original_data(model_connection_info, 'v1')

    symbol = model_connection_info['Symbol']
    interval = model_connection_info['Interval']
    batch_size = model_connection_info['Batch Size']
    input_model_infos = model_connection_info['Input Model Infos']
    target_types = model_connection_info['Target Types']

    # Step 1: 獲取數據
    df_original = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
    df = df_original.copy()
    df['Target'] = signals.super_trend_strategy_v2(df, 0.008, 0.008, 0, 20)
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

    predicted_v2 = []
    if need_v2_predict == True:
        v2_model_path = os.path.join(parent_path, 'trained_models/1713239614_softmax_loss-1.1766_accuracy-0.5985_f1-0.1456')
        v2_model = tf.keras.models.load_model(v2_model_path, custom_objects={'multiclass_f1_score': multiclass_f1_score})

        v2_model_connection_info = {}
        v2_model_info_path = os.path.join(v2_model_path, 'model_connection_info.json')

        if os.path.exists(v2_model_info_path):
            # 这里可以添加读取或处理文件的代码
            # 例如，读取JSON文件
            import json
            with open(v2_model_info_path, 'r') as file:
                v2_model_connection_info = json.load(file)

        v2_original_X, _ = get_original_data(v2_model_connection_info, 'v2')

        v2_input_model_infos = v2_model_connection_info['Input Model Infos']
        v2_target_types = v2_model_connection_info['Target Types']

        df_v2 = df_original.copy()
        df_v2['Target'] = signals.super_trend_strategy_v3(df_v2, 0.008, 0.008, 0.003, 12)
        df_v2, customized_cols_infos, new_cols = v2_customized_specific_period_col(df_v2)
        # print(df_v2)
        drop_front_data_count = 1000
        drop_back_data_count = 300
        df_v2 = df_v2[drop_front_data_count:]
        df_v2.reset_index(drop=True, inplace=True)

        X_v2 = df_v2
        y_v2 = df_v2['Target']
        y_categorical_v2 = to_categorical(y_v2, num_classes=len(v2_target_types))

        input_model_X_datas_v2, input_model_Y_datas_v2 = set_input_model_layers(X_v2, y_categorical_v2, v2_original_X, v2_input_model_infos)
        target_y_v2 = input_model_Y_datas_v2[0]

        X_v2 = X_v2[-len(target_y_v2):]

        predicted_v2 = v2_model.predict(input_model_X_datas_v2)

    return df, trained_model_path, target_y, X, predicted, predicted_v2

def get_simulation_predict(need_v2_predict=False):
    # 載入已經訓練好的模型
    # 獲取當前文件的絕對路徑
    current_path = os.path.abspath(os.path.dirname(__file__))
    parent_path = os.path.join(current_path, '..')
    local_data_path = os.path.join(parent_path, 'local_data')

    trained_model_path = os.path.join(parent_path, 'trained_models/1710835373_softmax_loss-1.2277_accuracy-0.4738_f1-0.2171_real_best')
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
    original_X, _ = get_original_data(model_connection_info, 'v1')

    symbol = model_connection_info['Symbol']
    interval = model_connection_info['Interval']
    batch_size = model_connection_info['Batch Size']
    input_model_infos = model_connection_info['Input Model Infos']
    target_types = model_connection_info['Target Types']

    # Step 1: 獲取數據
    df_original = pd.read_csv(os.path.join(local_data_path, 'bull_simulation_2.csv'), index_col=0)

    start_date = '2021-01-01 00:00:00'
    date_range = pd.date_range(start=start_date, freq='15T', periods=len(df_original))
    df_original['datetime'] = date_range

    df_original = modules_data.calculate_df_all_data(df_original, '')
    print(f'Original Data: {df_original}')

    df_original.to_csv(os.path.join(local_data_path, 'bull_simulation_2_calculated.csv'))

    df = df_original.copy()
    df['Target'] = signals.super_trend_strategy_v2(df, 0.008, 0.008, 0, 20)
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

    predicted_v2 = []
    if need_v2_predict == True:
        v2_model_path = os.path.join(parent_path, 'trained_models/1713239614_softmax_loss-1.1766_accuracy-0.5985_f1-0.1456')
        v2_model = tf.keras.models.load_model(v2_model_path, custom_objects={'multiclass_f1_score': multiclass_f1_score})

        v2_model_connection_info = {}
        v2_model_info_path = os.path.join(v2_model_path, 'model_connection_info.json')

        if os.path.exists(v2_model_info_path):
            # 这里可以添加读取或处理文件的代码
            # 例如，读取JSON文件
            import json
            with open(v2_model_info_path, 'r') as file:
                v2_model_connection_info = json.load(file)

        v2_original_X, _ = get_original_data(v2_model_connection_info, 'v2')

        v2_input_model_infos = v2_model_connection_info['Input Model Infos']
        v2_target_types = v2_model_connection_info['Target Types']

        df_v2 = df_original.copy()
        df_v2['Target'] = signals.super_trend_strategy_v3(df_v2, 0.008, 0.008, 0.003, 12)
        df_v2, customized_cols_infos, new_cols = v2_customized_specific_period_col(df_v2)
        # print(df_v2)
        drop_front_data_count = 1000
        drop_back_data_count = 300
        df_v2 = df_v2[drop_front_data_count:]
        df_v2.reset_index(drop=True, inplace=True)

        X_v2 = df_v2
        y_v2 = df_v2['Target']
        y_categorical_v2 = to_categorical(y_v2, num_classes=len(v2_target_types))

        input_model_X_datas_v2, input_model_Y_datas_v2 = set_input_model_layers(X_v2, y_categorical_v2, v2_original_X, v2_input_model_infos)
        target_y_v2 = input_model_Y_datas_v2[0]

        X_v2 = X_v2[-len(target_y_v2):]

        predicted_v2 = v2_model.predict(input_model_X_datas_v2)

    return df, trained_model_path, target_y, X, predicted, predicted_v2

def set_hybrid_tp(entry_target, entry_v2_target, entry_v2_confidence):
    take_profit = 0

    take_profit_0_1 = 0.025
    take_profit_2_3 = 0.03
    take_profit_4_5 = 0.03

    if entry_v2_target == 1 and not(0.36 <= entry_v2_confidence <= 0.375):
        if 0.34 <= entry_v2_confidence <= 0.44:
            take_profit = 0.015
        elif 0.23 <= entry_v2_confidence <= 0.26:
            take_profit = 0.03
    elif entry_v2_target == 2 and not(0.35 <= entry_v2_confidence >= 0.36):
        if 0.32 <= entry_v2_confidence <= 0.37:
            take_profit = 0.01
        elif 0.21 <= entry_v2_confidence <= 0.24:
            take_profit = 0.01
    elif entry_v2_target == 3 and not(0.258 <= entry_v2_confidence <= 0.264 or 0.27 <= entry_v2_confidence <= 0.278):
        if 0.252 <= entry_v2_confidence <= 0.28:
            take_profit = 0.015
    elif entry_v2_target == 4 and not(0.255 <= entry_v2_confidence <= 0.263): # 需調整，容易輸
        if 0.245 <= entry_v2_confidence <= 0.25:
            take_profit = 0.015
        elif 0.255 <= entry_v2_confidence <= 0.265:
            take_profit = 0.01
        elif 0.315 <= entry_v2_confidence <= 0.32:
            take_profit = 0.015
    # elif entry_v2_target == 5 and not(0.33 <= entry_v2_confidence <= 0.36 or 0.29 <= entry_v2_confidence <= 0.3): # 需調整，容易輸
    #     if 0.32 <= entry_v2_confidence <= 0.37:
    #         take_profit = 0.01
    #     if 0.27 <= entry_v2_confidence <= 0.32:
    #         take_profit = 0.015
    elif entry_v2_target == 6 and not(0.246 <= entry_v2_confidence <= 0.252 or 0.254 <= entry_v2_confidence <= 0.259):
        if 0.243 <= entry_v2_confidence <= 0.265:
            take_profit = 0.015
    elif entry_v2_target == 7 and not(0.47 <= entry_v2_confidence <= 0.475 or 0.39 <= entry_v2_confidence <= 0.44):
        if 0.46 <= entry_v2_confidence <= 0.49:
            take_profit = 0.025 # 或0.03
        elif 0.37 <= entry_v2_confidence <= 0.49:
            take_profit = 0.015 # 或0.01(0.01比較保險)
    elif entry_v2_target == 8 and not(0.44 <= entry_v2_confidence <= 0.47):
        if 0.4 <= entry_v2_confidence <= 0.53:
            take_profit = 0.02
    elif entry_v2_target == 9:
        if 0.2 <= entry_v2_confidence <= 0.27:
            take_profit = 0.02
    elif entry_v2_target == 10 and not(0.246 <= entry_v2_confidence <= 0.25):
        if 0.243 <= entry_v2_confidence <= 0.255:
            take_profit = 0.01

    if take_profit == 0:
        if entry_target == 0 or entry_target == 1:
            take_profit = take_profit_0_1
        elif entry_target == 2 or entry_target == 3:
            take_profit = take_profit_2_3
        elif entry_target == 4 or entry_target == 5:
            take_profit = take_profit_4_5

    return take_profit

def check_kline_color(enable_kline_color_check, position, kline_color):
    if enable_kline_color_check == True:
        if position == 'Long' and kline_color == 0:
            return True
        elif position == 'Short' and kline_color == 1:
            return True
        else:
            return False
    else:
        return True

def check_entry_volume(volume, volume_threshold, toggle_entry_volume_sma=False, volume_sma=None, entry_volume_sma_excess=None, entry_volume_sma_excess_percentage=None):
    if toggle_entry_volume_sma == False:
        if volume >= volume_threshold:
            return True
        else:
            return False
    elif toggle_entry_volume_sma == True and entry_volume_sma_excess is not None:
        if volume_sma and volume >= volume_sma + entry_volume_sma_excess:
            return True
        elif not volume_sma and volume >= volume_threshold:
            return True
        else:
            return False
    elif toggle_entry_volume_sma == True and entry_volume_sma_excess_percentage is not None:
        if volume_sma and volume >= volume_sma * (1 + entry_volume_sma_excess_percentage):
            return True
        elif not volume_sma and volume >= volume_threshold:
            return True
        else:
            return False

def generate_combinations(conditions, condition_list=[]):
    # Generate the combination of conditions
    possibilities = []

    if condition_list == []:
        condition_list = [i for i in range(1, conditions + 1)]

        for i in range(1, conditions + 1):
            for j in list(itertools.combinations(condition_list, i)):
                possibilities.append(j)
    else:
        for i in range(1, conditions + 1):
            for j in list(itertools.combinations(condition_list, i)):
                combination = (1, 2) + j
                possibilities.append(combination)

    return possibilities

def examine_combination(combination, id):
    return combination == () or id in combination

def get_column_mean_std(df, column):
    mean = df[column].mean()
    std = df[column].std()

    return mean, std

def calculate_volume_sma(df, period=20):
    df['Volume SMA'] = df['Volume'].rolling(window=period).mean()

    return df

def print_column_statistics(df, column):
    mean = df[column].mean()
    median = df[column].median()
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    min_val = df[column].min()
    max_val = df[column].max()
    std = df[column].std()

    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")
    print(f"Median: {median}")
    print(f"Q1: {q1}")
    print(f"Q3: {q3}")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")

    # Plot the box plot
    plt.figure(figsize=(10, 6))
    df.boxplot(column=column)
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)

    # Show plot
    plt.show()

def main():
    total_klines = 30000
    get_local_file_name = ''
    # 當前
    end_time = int(datetime.timestamp(datetime.now())) * 1000

    end_time_seconds = (end_time / 1000) + 1
    end_datetime = datetime.fromtimestamp(end_time_seconds)
    end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
    # 特定
    # end_time_string = "2023-12-31 23:59:59"

    df, trained_model_path, target_y, X, predicted, predicted_v2 = get_predict(total_klines, end_time_string, get_local_file_name, need_v2_predict=True)

    print(f'predicted_length: {len(predicted)}')
    print(f'predicted_v2_length: {len(predicted_v2)}')

    # predicted = target_y

    datetime_string = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    execution_start_time = time.time()

    # df, trained_model_path, target_y, X, predicted, predicted_v2 = get_simulation_predict(need_v2_predict=True)

    # # Exit condition combination grid search
    # # Generate the combination of conditions
    # combination = generate_combinations(8)
    # combination = generate_combinations(6, [3, 4, 5, 6, 7, 8])
    # print(f'combination: {combination}')

    # for c in combination:
    #     enhancement_strategy_v2(df, trained_model_path, target_y, X, predicted, predicted_v2, c, datetime_string)
    # enhancement_strategy(X, predicted)


    # Entry volume grid search
    # for i in range(25000, 92500, 2500):
    #     for j in range(25000, 92500, 2500):
    #         for k in range(25000, 92500, 2500):
    #             enhancement_strategy_v2(df, trained_model_path, target_y, X, predicted, predicted_v2, (1, 2, 3, 6), datetime_string, entry_volume_0_1_arg=i, entry_volume_2_3_arg=j, 
    #                                     entry_volume_4_5_arg=k, save_grid_search_result=True, kline_color_check=True)


    # Entry volume SMA grid search - excess
    # for i in range(2, 645):
    #     X = calculate_volume_sma(X, i)
    #     for j in range(0, 42500, 2500):
    #         enhancement_strategy_v2(df, trained_model_path, target_y, X, predicted, predicted_v2, (1, 2, 3, 6), datetime_string, toggle_entry_volume_sma=True, 
    #                                 volume_sma_window=i, entry_volume_sma_excess=j, save_grid_search_result=True, kline_color_check=False)


    # Entry volume SMA grid search - percentage
    # for i in range(2, 645):
    #     X = calculate_volume_sma(X, i)
    #     percentage_list = [x / 1000 for x in range(100, 1600, 100)]
    #     print(percentage_list)
    #     for j in percentage_list:
    #         enhancement_strategy_v2(df, trained_model_path, target_y, X, predicted, predicted_v2, (1, 2, 3, 6), datetime_string, toggle_entry_volume_sma=True, 
    #                                 volume_sma_window=i, entry_volume_sma_excess_percentage=j, save_grid_search_result=True, kline_color_check=True)


    # Pure backtesting
    X = calculate_volume_sma(X, 50)
    enhancement_strategy_v2(df, trained_model_path, target_y, X, predicted, predicted_v2, (1, 2, 3, 6), datetime_string,
                            toggle_entry_volume_sma=True, volume_sma_window=50, entry_volume_sma_excess=25000, hold_count_limit_arg=47)
    # enhancement_strategy_v2_original(df, trained_model_path, target_y, X, predicted)
    # enhancement_strategy_v2(df, trained_model_path, target_y, X, predicted, predicted_v2, (1, 2, 3, 6), datetime_string, hold_count_limit_arg=18, save_grid_search_result=True)


    # Parameter hold count limit grid search
    # X = calculate_volume_sma(X, 88)
    # for i in range(1, 289):
    #     enhancement_strategy_v2(df, trained_model_path, target_y, X, predicted, predicted_v2, (1, 2, 3, 6), datetime_string,
    #                             toggle_entry_volume_sma=True, volume_sma_window=88, entry_volume_sma_excess=32500, hold_count_limit_arg=i, grid_hold_count_limit_arg=True, save_grid_search_result=True)

    # X = calculate_volume_sma(X, 50)
    # for i in range(1, 289):
    #     enhancement_strategy_v2(df, trained_model_path, target_y, X, predicted, predicted_v2, (1, 2, 3, 6), datetime_string,
    #                             toggle_entry_volume_sma=True, volume_sma_window=50, entry_volume_sma_excess=25000, hold_count_limit_arg=i, grid_hold_count_limit_arg=True, save_grid_search_result=True)


    # # Parameter entry confidence grid search
    # X = calculate_volume_sma(X, 88)

    # confidence_values = [x / 100 for x in range(25, 101, 1)]
    # combinations = [(a, b) for a, b in itertools.combinations(confidence_values, r=2)]

    # grid search
    # for i in combinations:
    #     for j in combinations:
    #         for k in combinations:
    #             for l in combinations:
    #                 for m in combinations:
    #                     for n in combinations:
    #                         enhancement_strategy_v2(df, trained_model_path, target_y, X, predicted, predicted_v2, (1, 2, 3, 6), datetime_string, entry_confidence_0_arg=i, entry_confidence_1_arg=j, entry_confidence_2_arg=k, entry_confidence_3_arg=l, entry_confidence_4_arg=m, entry_confidence_5_arg=n,
    #                                                 toggle_entry_volume_sma=True, volume_sma_window=88, entry_volume_sma_excess=32500, grid_entry_confidence=True, save_grid_search_result=True, hold_count_limit_arg=18)

    # random search
    # for _ in range(40000):
    #     confidence_0 = random.choice(combinations)
    #     confidence_1 = random.choice(combinations)
    #     confidence_2 = random.choice(combinations)
    #     confidence_3 = random.choice(combinations)
    #     confidence_4 = random.choice(combinations)
    #     confidence_5 = random.choice(combinations)
    #     enhancement_strategy_v2(df, trained_model_path, target_y, X, predicted, predicted_v2, (1, 2, 3, 6), datetime_string, entry_confidence_0_arg=confidence_0, entry_confidence_1_arg=confidence_1, entry_confidence_2_arg=confidence_2, entry_confidence_3_arg=confidence_3, entry_confidence_4_arg=confidence_4, entry_confidence_5_arg=confidence_5,
    #                             toggle_entry_volume_sma=True, volume_sma_window=88, entry_volume_sma_excess=32500, grid_entry_confidence=True, save_grid_search_result=True, hold_count_limit_arg=18)


    # # Parameter limit order rate & auto cancel grid search
    # X = calculate_volume_sma(X, 50)
    # for i in range(230, 260, 10):
    #     for j in range(1, 13):
    #         for l in range(12, 50, 2):
    #             for m in range(20, 90, 10):
    #                 if i == 230 and j < 7:
    #                     continue
    #                 limit_order_rate = i / 10000
    #                 auto_cancel = j
    #                 stop_loss = m / 10000
    #                 enhancement_strategy_v2(df, trained_model_path, target_y, X, predicted, predicted_v2, (1, 2, 3, 6), datetime_string,
    #                                         toggle_entry_volume_sma=True, volume_sma_window=50, entry_volume_sma_excess=25000, hold_count_limit_arg=l, stop_loss_arg=stop_loss,
    #                                         toggle_limit_order=True, limit_order_rate=limit_order_rate, auto_cancel=auto_cancel, fee_arg=0.0005, save_grid_search_result=True)


    # # Parameter no keep update max profit count limit grid search
    # datetime_string = end_datetime.strftime('%Y_%m_%d_%H_%M_%S')

    # predicted_targets = np.argmax(predicted, axis=1)
    # confidences = np.max(predicted, axis=1)

    # predicted_targets_v2 = np.argmax(predicted_v2, axis=1)
    # confidences_v2 = np.max(predicted_v2, axis=1)

    # for i in range(1, 101):
    #     max_total_profit = grid_enhancement_strategy(target_y, X, predicted_targets, confidences, datetime_string=datetime_string, 
    #                                                  predicted_targets_v2=predicted_targets_v2, confidences_v2=confidences_v2, no_keep_update_max_profit_count_limit=i)


    execution_end_time = time.time()
    print(f'Execution Time: {execution_end_time - execution_start_time}')

if __name__ == "__main__":
    main()

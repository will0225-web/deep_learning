import sys
import os

os.add_dll_directory('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin')
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
from modules import utilities as utilities

from tensorflow.keras.utils import to_categorical

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


def enhancement_strategy(df, trained_model_path, target_y, X, predicted, combination=(), datetime_string=None, 
                         entry_volume_0_1=None, hold_count_limit=None, entry_confidence_1=None,
                         grid_entry_confidence=False, save_grid_search_result=False, kline_color_check=False):
    output_module_name = f'{datetime_string}_{entry_confidence_1}'

    total_profit = 0
    position = 0  # 目前倉位狀態，0表示無倉位，1表示多頭倉位，-1表示空頭倉位
    entry_price = 0  # 開倉價格
    entry_index = 0  # 開倉索引
    lose_count = 0
    win_count = 0
    max_total_profit = float('-inf')
    drawdown = 0
    max_drawdown = 0
    total_winning_percentage = 0
    total_losing_percentage = 0
    sharpe_ratio = 0

    if entry_volume_0_1 is None:
        entry_volume_0_1 = 0

    take_profit = 0
    stop_loss = 0.04
    max_profit = 0

    fee = 0.0010

    if hold_count_limit is None:
        hold_count_limit = 288

    if entry_confidence_1 is None:
        entry_confidence_1 = 0.5

    is_finish_order = False

    for i in range(1, len(predicted)):
        original_data = X.iloc[i]
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

        p = predicted[i]
        predicted_target = np.argmax(p)
        confidence = np.max(predicted[i])

        # real target
        target = np.argmax(target_y[i])
        reason = ''

        if position != 0:
            current_hold_count = i - entry_index

            if position == 1:
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

                    if current_high_profit <= max_profit:
                        no_keep_update_max_profit_count += 1
                    else:
                        max_profit = max(max_profit, current_high_profit)
                        no_keep_update_max_profit_count = 0

                    # 結單
                    if examine_combination(combination, 2) and max_profit >= take_profit:
                        # 不超過take profit的percentage
                        profit = min(max_profit - fee, take_profit - fee)
                        reason = f'Long take profit {take_profit}' 
                        is_finish_order = True
                    elif examine_combination(combination, 3) and max_profit >= trailing_threshold and current_close_profit <= max_profit - trailing_return_threshold:
                        # 回調的狀況先拿profit
                        profit = max_profit - trailing_return_threshold - fee
                        reason = f'Long trailing stop {round(profit * 100, 2)} %'
                        is_finish_order = True
                    elif examine_combination(combination, 6) and current_hold_count >= hold_count_limit:
                        profit = current_close_profit - fee
                        reason = 'Long hold count limit'
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
                    max_profit = 0
                    entry_price = 0
                    take_profit = 0
                    trailing_threshold = 0
                    trailing_return_threshold = 0
                    no_keep_update_max_profit_count = 0

                    is_finish_order = False

            elif position == -1:
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

                    if current_low_profit <= max_profit:
                        no_keep_update_max_profit_count += 1
                    else:
                        max_profit = max(max_profit, current_low_profit)
                        no_keep_update_max_profit_count = 0

                    # 結單
                    if examine_combination(combination, 2) and max_profit >= take_profit:
                        # 不超過take profit的percentage
                        profit = min(max_profit - fee, take_profit - fee)
                        reason = f'Short take profit {take_profit}'
                        is_finish_order = True
                    elif examine_combination(combination, 3) and max_profit >= trailing_threshold and current_close_profit <= max_profit - trailing_return_threshold:
                        # 回調的狀況先拿profit
                        profit = max_profit - trailing_return_threshold - fee
                        reason = f'Short trailing stop {round(profit * 100, 2)} %'
                        is_finish_order = True
                    elif examine_combination(combination, 6) and current_hold_count >= hold_count_limit:
                        profit = current_close_profit - fee
                        reason = 'Short hold count limit'
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
                    max_profit = 0
                    entry_price = 0
                    take_profit = 0
                    trailing_threshold = 0
                    trailing_return_threshold = 0
                    no_keep_update_max_profit_count = 0

                    is_finish_order = False
        else:
            no_keep_update_max_profit_count = 0
            is_finish_order = False

            # 開倉邏輯
            if predicted_target == 1 and entry_confidence_1 <= confidence and check_entry_volume(volume, entry_volume_0_1):
                position = 1

            # Set take profit and trailing stop accordingly
            if position != 0:
                entry_price = original_close
                entry_index = i

                take_profit = set_tp(predicted_target)

                trailing_threshold = take_profit / 1.5
                trailing_return_threshold = trailing_threshold / 2
                if trailing_return_threshold <= 0.004:
                    trailing_return_threshold = 0.004

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
                headers = ['Original Datetime', 'Position', 'Entry Price', 'Reason', 'Max Profit', 'Original Close', 'Original High', 'Original Low', 'Total Profit', 'Max Total Profit', 'Drawdown', 'Max Drawdown', 'Predicted Target', 'Confidence', 'Target', 'Win Count', 'Lose Count', 'Total trade', 'Win Rate', 'Total Winning Percentage', 'Total Losing Percentage', 'Sharpe Ratio']

                # 检查文件是否为空，如果是空的，写入标题行
                if file.tell() == 0:
                    writer.writerow(headers)


                # 指定原始时间戳的时区为UTC
                original_datetime_utc = original_datetime.tz_localize('UTC')

                # 转换为台湾时间（UTC+8）
                taiwan_datetime = original_datetime_utc.tz_convert('Asia/Taipei')

                # 格式化为字符串
                taiwan_datetime_str = taiwan_datetime.strftime('%Y-%m-%d %H:%M:%S')

                writer.writerow([taiwan_datetime_str, position, entry_price, reason, max_profit, original_close, original_high, original_low, total_profit, max_total_profit, drawdown, max_drawdown, predicted_target, confidence, target, win_count, lose_count, total_trade, win_rate, total_winning_percentage, total_losing_percentage, sharpe_ratio])

    # 配置输出CSV文件
    if save_grid_search_result == True:
        with open(f'{datetime_string}_grid_search_results.csv', 'a', newline='') as file:
            writer = csv.writer(file)

            if file.tell() == 0:
                # Essential performance metrics
                headers = ['Total Profit', 'Max Total Profit', 'Max Drawdown', 'Total Trade', 'Win Rate']

                # Add hold count limit to headers
                index = headers.index('Total Profit')
                headers.insert(index, 'Hold Count Limit')

                # Add entry confidence conditions to headers
                if grid_entry_confidence == True:
                    index = headers.index('Total Profit')
                    headers[index:1] = ['entry_confidence_0', 'entry_confidence_1', 'entry_confidence_2', 'entry_confidence_3', 'entry_confidence_4', 'entry_confidence_5']

                writer.writerow(headers)

            # Add essential performance metrics and entry volume conditions to output string and row
            output_string = f'Total Profit: {total_profit}, Max Total Profit: {max_total_profit}, Max Drawdown: {max_drawdown}, Total Trade: {total_trade}, Win Rate: {win_rate}'
            row = [total_profit, max_total_profit, max_drawdown, total_trade, win_rate]

            # Add hold count limit to output string and row
            parts = output_string.split(', ')
            total_profit_index = next(i for i, part in enumerate(parts) if 'Total Profit' in part)
            parts.insert(total_profit_index, f'Hold Count Limit: {hold_count_limit}')
            output_string = ', '.join(parts)

            row.insert(total_profit_index, hold_count_limit)

            print(output_string)

            writer.writerow(row)

    elif save_grid_search_result == False:
        model_process.export_total_profit_info(df, trained_model_path, datetime_string, f'{trained_model_path}/{output_module_name}_output.csv')

    return total_profit, max_total_profit, drawdown, max_drawdown, total_trade, win_count, win_rate

def spot_strategy(df, trained_model_path, target_y, X, predicted, combination=(), datetime_string=None,
                  entry_volume_0_1=None, entry_confidence_1=None, order_limit_arg=None,
                  take_profit_arg=None, stop_loss_arg=None, hold_count_limit_arg=None,
                  grid_entry_confidence=False, save_grid_search_result=False):
    output_model_name = f'{datetime_string}_{entry_confidence_1}'

    # print(f'df: {df}')
    # print(f'predicted length: {len(predicted)}')

    position = 0  # 目前倉位狀態，0表示無倉位，1表示多頭倉位，-1表示空頭倉位
    position_amount = 0
    entry_price = 0  # 開倉價格
    entry_index = 0  # 開倉索引

    max_runup = 0
    total_profit = 0
    max_total_profit = 0
    drawdown = 0
    max_drawdown = 0
    win_count = 0
    lose_count = 0
    total_winning_percentage = 0
    total_losing_percentage = 0
    sharpe_ratio = 0

    take_profit = 0.03 if take_profit_arg is None else take_profit_arg
    stop_loss = 0.03 if stop_loss_arg is None else stop_loss_arg
    hold_count_limit = 288 if hold_count_limit_arg is None else hold_count_limit_arg

    trailing_threshold = take_profit / 1.5
    trailing_return_threshold = trailing_threshold / 2
    if trailing_return_threshold <= 0.004:
        trailing_return_threshold = 0.004

    long_tp_price = 0

    fee = 0.0010

    if entry_volume_0_1 is None:
        entry_volume_0_1 = 0

    if entry_confidence_1 is None:
        entry_confidence_1 = 0.5

    order_limit = 999 if order_limit_arg is None else order_limit_arg

    is_finish_order = False

    for i in range(1, len(predicted)):
        original_data = X.iloc[i]

        original_close = original_data['Close']
        original_high = original_data['High']
        original_low = original_data['Low']
        original_datetime = pd.to_datetime(original_data['datetime'])
        volume = original_data['Volume']

        p = predicted[i]
        predicted_target = np.argmax(p)
        confidence = np.max(predicted[i])

        # real target
        target = np.argmax(target_y[i])
        reason = ''

        if position in [0, 1] and position_amount < order_limit:
            if predicted_target == 1 and entry_confidence_1 <= confidence and check_entry_volume(volume, entry_volume_0_1):
                if position == 0:
                    position = 1

                if entry_index == 0:
                    entry_index = i

                entry_price = (entry_price * position_amount + original_close) / (position_amount + 1)
                entry_price = round(entry_price, 2)
                position_amount += 1

                long_tp_price = round(entry_price * (1 + take_profit), 2)

        if position != 0:
            current_hold_count = i - entry_index

            if position == 1:
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

                    max_runup = round(max(max_runup, current_high_profit), 4)

                    # 結單
                    if examine_combination(combination, 2) and max_runup >= take_profit:
                        # 不超過take profit的percentage
                        profit = min(max_runup - fee, take_profit - fee)
                        reason = f'Long take profit {take_profit * 100} %' 
                        is_finish_order = True
                    elif examine_combination(combination, 3) and max_runup >= trailing_threshold and current_close_profit <= max_runup - trailing_return_threshold:
                        # 回調的狀況先拿profit
                        profit = max_runup - trailing_return_threshold - fee
                        reason = f'Long trailing stop {round(profit * 100, 2)} %'
                        is_finish_order = True
                    elif examine_combination(combination, 4) and current_hold_count >= hold_count_limit:
                        profit = current_close_profit - fee
                        reason = 'Long hold count limit'
                        is_finish_order = True

                if is_finish_order:
                    profit = round(profit, 4)
                    total_profit += profit * position_amount

                    if profit <= 0:
                        lose_count += 1
                        total_losing_percentage += profit * position_amount * 100
                    else:
                        win_count += 1
                        total_winning_percentage += profit * position_amount * 100

                    position = 0
                    position_amount = 0
                    max_runup = 0
                    entry_price = 0
                    entry_index = 0
                    long_tp_price = 0

                    is_finish_order = False

            elif position == -1:
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

                    max_runup = round(max(max_runup, current_low_profit), 4)

                    # 結單
                    if examine_combination(combination, 2) and max_runup >= take_profit:
                        # 不超過take profit的percentage
                        profit = min(max_runup - fee, take_profit - fee)
                        reason = f'Short take profit {take_profit * 100} %'
                        is_finish_order = True
                    elif examine_combination(combination, 3) and max_runup >= trailing_threshold and current_close_profit <= max_runup - trailing_return_threshold:
                        # 回調的狀況先拿profit
                        profit = max_runup - trailing_return_threshold - fee
                        reason = f'Short trailing stop {round(profit * 100, 2)} %'
                        is_finish_order = True
                    elif examine_combination(combination, 4) and current_hold_count >= hold_count_limit:
                        profit = current_close_profit - fee
                        reason = 'Short hold count limit'
                        is_finish_order = True

                if is_finish_order:
                    profit = round(profit, 4)
                    total_profit += profit * position_amount

                    if profit <= 0:
                        lose_count += 1
                        total_losing_percentage += profit * position_amount * 100
                    else:
                        win_count += 1
                        total_winning_percentage += profit * position_amount * 100

                    position = 0
                    position_amount = 0
                    max_runup = 0
                    entry_price = 0
                    entry_index = 0

                    is_finish_order = False

        max_total_profit = max(max_total_profit, total_profit)

        if max_total_profit > total_profit:
            drawdown = ((1 + max_total_profit) - (1 + total_profit)) / (1 + max_total_profit) * 100 if total_profit != 0 else 0
            drawdown = round(drawdown, 2)
        else:
            drawdown = 0
        max_drawdown = max(max_drawdown, drawdown)

        total_trade = win_count + lose_count
        win_rate = win_count / total_trade * 100 if (total_trade) != 0 else 0
        win_rate = round(win_rate, 2)

        sharpe_ratio = total_winning_percentage / abs(total_losing_percentage) if total_losing_percentage != 0 else 0
        sharpe_ratio = round(sharpe_ratio, 2)

        if save_grid_search_result == False:
            with open(f'{trained_model_path}/{output_model_name}_output.csv', 'a', newline='') as file:  # 'a'表示附加模式，這樣數據將會被添加到文件而不是覆蓋它
                writer = csv.writer(file)

                headers = ['Datetime', 'Position', 'Position Amount', 'Entry Price', 'Reason', 'Max Run-up', 'Long TP Price', 'Original Close', 'Original High', 'Original Low', 'Total Profit', 'Max Total Profit', 'Drawdown', 'Max Drawdown', 'Predicted Target', 'Confidence', 'Target', 'Win Count', 'Lose Count', 'Total trade', 'Win Rate', 'Total Winning Percentage', 'Total Losing Percentage', 'Sharpe Ratio']

                # 假如是空白文件，寫入標題列
                if file.tell() == 0:
                    writer.writerow(headers)

                # 將 Datetime 轉換為台灣時間（UTC+8）
                original_datetime_utc = original_datetime.tz_localize('UTC')
                taiwan_datetime = original_datetime_utc.tz_convert('Asia/Taipei')
                taiwan_datetime_str = taiwan_datetime.strftime('%Y-%m-%d %H:%M:%S')

                writer.writerow([taiwan_datetime_str, position, position_amount, entry_price, reason, max_runup, long_tp_price, original_close, original_high, original_low, total_profit, max_total_profit, drawdown, max_drawdown, predicted_target, confidence, target, win_count, lose_count, total_trade, win_rate, total_winning_percentage, total_losing_percentage, sharpe_ratio])

    # 配置输出CSV文件
    if save_grid_search_result == True:
        with open(f'{datetime_string}_grid_search_results.csv', 'a', newline='') as file:
            writer = csv.writer(file)

            if file.tell() == 0:
                # Essential performance metrics
                headers = ['Total Profit', 'Max Total Profit', 'Max Drawdown', 'Total Trade', 'Win Rate']

                # Add hold count limit to headers
                index = headers.index('Total Profit')
                headers.insert(index, 'Hold Count Limit')

                # Add entry confidence conditions to headers
                if grid_entry_confidence == True:
                    index = headers.index('Total Profit')
                    headers[index:1] = ['entry_confidence_0', 'entry_confidence_1', 'entry_confidence_2', 'entry_confidence_3', 'entry_confidence_4', 'entry_confidence_5']

                writer.writerow(headers)

            # Add essential performance metrics and entry volume conditions to output string and row
            output_string = f'Total Profit: {total_profit}, Max Total Profit: {max_total_profit}, Max Drawdown: {max_drawdown}, Total Trade: {total_trade}, Win Rate: {win_rate}'
            row = [total_profit, max_total_profit, max_drawdown, total_trade, win_rate]

            # Add hold count limit to output string and row
            parts = output_string.split(', ')
            total_profit_index = next(i for i, part in enumerate(parts) if 'Total Profit' in part)
            parts.insert(total_profit_index, f'Hold Count Limit: {hold_count_limit}')
            output_string = ', '.join(parts)

            row.insert(total_profit_index, hold_count_limit)

            print(output_string)

            writer.writerow(row)

    elif save_grid_search_result == False:
        model_process.export_total_profit_info(df, trained_model_path, datetime_string, f'{trained_model_path}/{output_model_name}_output.csv')

    return total_profit, max_total_profit, drawdown, max_drawdown, total_trade, win_count, win_rate

def get_original_data(model_connection_info, get_local_file_name=None):
    symbol = model_connection_info['Symbol']
    interval = model_connection_info['Interval']
    if get_local_file_name is None:
        get_local_file_name = model_connection_info['Get Local File Name']
    total_klines = model_connection_info['Get Original Klines Count']
    end_time_string = model_connection_info['End Time']
    drop_front_data_count = model_connection_info['Drop Front Data Count']
    drop_back_data_count = model_connection_info['Drop Back Data Count']

    original_df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
    print(f'Original df: {original_df}')

    # original_df, customized_cols_infos, new_cols = customized_specific_period_col(original_df)
    # original_df = set_y_label(original_df, lookahead=288, long_percentage=0.03)

    original_df = original_df[drop_front_data_count:-drop_back_data_count]
    original_df.reset_index(drop=True, inplace=True)

    return original_df, original_df['y']


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

def multiclass_f05_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_fbeta_score(y_true, y_pred, beta=0.5)

def get_predict(total_klines, end_time_string, get_local_file_name, model_path):
    # 載入已經訓練好的模型
    # 獲取當前文件的絕對路徑
    current_path = os.path.abspath(os.path.dirname(__file__))
    parent_path = os.path.join(current_path, '..')
    trained_model_path = os.path.join(parent_path, model_path)
    try:
        model = tf.keras.models.load_model(trained_model_path, custom_objects={'multiclass_f05_score': multiclass_f05_score})
    except TypeError as e:
        model = tf.keras.models.load_model(trained_model_path, custom_objects={'multiclass_f05_score': multiclass_f05_score}, compile=False)
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
        raise FileNotFoundError(f'File not found: {model_info_path}')
    # 拿原始資料
    original_X, _ = get_original_data(model_connection_info, get_local_file_name)

    symbol = model_connection_info['Symbol']
    interval = model_connection_info['Interval']
    batch_size = model_connection_info['Batch Size']
    input_model_infos = model_connection_info['Input Model Infos']
    target_types = model_connection_info['Target Types']

    # Step 1: 獲取數據
    df_original = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
    df = df_original.copy()
    # df, _, _ = customized_specific_period_col(df)
    # df = set_y_label(df, lookahead=288, long_percentage=0.03)
    print(f'df: {df}')

    # 拿掉前後無參考性資料
    drop_front_data_count = 1000
    drop_back_data_count = 300
    df = df[drop_front_data_count:]
    df.reset_index(drop=True, inplace=True)

    X = df
    y = df['y']
    y = y.fillna(0)
    y_categorical = to_categorical(y, num_classes=len(target_types))

    input_model_X_datas, input_model_Y_datas = set_input_model_layers(X, y_categorical, original_X, input_model_infos)
    target_y = input_model_Y_datas[0]

    X = X[-len(target_y):]

    # 評估模型
    loss, accuracy, f05 = model.evaluate(input_model_X_datas, target_y, batch_size=batch_size)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test F0.5: {f05:.4f}')

    predicted = model.predict(input_model_X_datas)

    return df, trained_model_path, target_y, X, predicted

def set_tp(entry_target):
    take_profit = 0

    take_profit_0_1 = 0.04

    if take_profit == 0:
        if entry_target == 0 or entry_target == 1:
            take_profit = take_profit_0_1

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

def print_column_statistics(df, column):
    mean = df[column].mean()
    median = df[column].median()
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    min_val = df[column].min()
    max_val = df[column].max()
    std = df[column].std()

    print(f'Mean: {mean}')
    print(f'Standard Deviation: {std}')
    print(f'Median: {median}')
    print(f'Q1: {q1}')
    print(f'Q3: {q3}')
    print(f'Min: {min_val}')
    print(f'Max: {max_val}')

    # Plot the box plot
    plt.figure(figsize=(10, 6))
    df.boxplot(column=column)
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)

    # Show plot
    plt.show()

def main():
    total_klines = 30000
    get_local_file_name = 'ETHUSDT_15m_2024-09-24_17-21-58_300000_spot_full_calculated.csv'

    # 當前
    end_time = int(datetime.timestamp(datetime.now())) * 1000
    end_time_seconds = (end_time / 1000) + 1
    end_datetime = datetime.fromtimestamp(end_time_seconds)
    end_time_string = end_datetime.strftime('%Y-%m-%d %H:%M:%S')
    # # 特定
    # end_time_string = '2022-12-31 23:59:59'

    # # Spot strategy backtesting
    # model_directory = 'trained_models/240927_remove_class_weight'
    # model_name = '1727610623_softmax_loss-0.6868_accuracy-0.6552_f1-0.3654'
    # model_path = os.path.join(model_directory, model_name)

    # df, trained_model_path, target_y, X, predicted = get_predict(total_klines, end_time_string, get_local_file_name, model_path)
    # print(f'Predicted: {predicted}')

    # prob_up = []
    # for prediction in predicted:
    #     prob_up.append(prediction[1])
    # X['prob_up'] = prob_up
    # X_filtered = X[['Close', 'High', 'Low', 'prob_up', 'datetime']]
    # X_filtered.reset_index(drop=True, inplace=True)
    # print(f'X: {X}')
    # print(f'X filtered: {X_filtered}')
    # X_filtered.to_csv(f'{model_path}/{model_name}_predicted.csv', index=False)

    # datetime_string = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # spot_strategy(df, trained_model_path, target_y, X, predicted, (1, 2, 3, 4), datetime_string, 
    #               take_profit_arg=0.04, stop_loss_arg=0.1, hold_count_limit_arg=288, order_limit_arg=10,
    #               entry_confidence_1=0.5, grid_entry_confidence=False, save_grid_search_result=False)

    # Batch backtesting
    model_dir_path = 'trained_models/241001'

    start_time = int(datetime.timestamp(datetime.now()))
    with open(f'{start_time}_backtesting_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model Name', 'Entry Confidence', 'Total Profit', 'Max Total Profit', 'Drawdown', 'Max Drawdown', 'Total Trade', 'Win Count', 'Win Rate'])

    for model in os.listdir(model_dir_path):
        model_path = f'{model_dir_path}/{model}'
        print(f'Model Name: {model}')
        df, trained_model_path, target_y, X, predicted = get_predict(total_klines, end_time_string, get_local_file_name, model_path)

        prob_up = []
        for prediction in predicted:
            prob_up.append(prediction[1])
        X['prob_up'] = prob_up
        X_filtered = X[['Close', 'High', 'Low', 'prob_up', 'datetime']]
        X_filtered.reset_index(drop=True, inplace=True)
        # print(f'X: {X}')
        # print(f'X filtered: {X_filtered}')
        X_filtered.to_csv(f'{model_path}/{model}_predicted.csv', index=False)

        for i in range(5, 10):
            entry_confidence = i / 10
            datetime_string = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            # total_profit, max_total_profit, drawdown, max_drawdown, total_trade, win_count, win_rate = enhancement_strategy(df, trained_model_path, target_y, X, predicted, (1, 2, 3, 6), datetime_string, entry_confidence_1=entry_confidence)
            total_profit, max_total_profit, drawdown, max_drawdown, total_trade, win_count, win_rate = spot_strategy(df, trained_model_path, target_y, X, predicted, (1, 2, 3, 4), datetime_string, 
                                                                                                                     take_profit_arg=0.04, stop_loss_arg=0.1, hold_count_limit_arg=288, order_limit_arg=10,
                                                                                                                     entry_confidence_1=entry_confidence, grid_entry_confidence=False, save_grid_search_result=False)

            if total_trade == 0:
                continue

            total_profit = round(total_profit, 4)
            max_total_profit = round(max_total_profit, 4)
            drawdown = round(drawdown, 2)
            max_drawdown = round(max_drawdown, 2)
            win_rate = round(win_rate, 2)

            with open(f'{start_time}_backtesting_results.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([model, entry_confidence, total_profit, max_total_profit, drawdown, max_drawdown, total_trade, win_count, win_rate])

if __name__ == '__main__':
    main()

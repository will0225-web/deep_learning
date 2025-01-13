import os
import pandas as pd
import time

from datetime import datetime, timedelta

from modules import indicators, signals
from modules.data import get_binance_klines_backward, calculate_df_all_data
from trading.utils.trading import check_server_time

env = os.getenv('ENV', 'development')

# Define model's customized cols
def customized_specific_period_col(df, model_type=None, calculate_target=False):
    customized_cols_infos = []
    
    # 紀錄原先的cols
    original_cols = set(df.columns)

    if model_type == 'supertrend':
        df, supertrend_delta_info = indicators.super_trend_delta_and_risk_v1(df, is_need_return_function_info=True)
        customized_cols_infos.append(supertrend_delta_info)
        
        df, volume_sma_info = indicators.calculate_volume_sma(df, 50, is_need_return_function_info=True)
        customized_cols_infos.append(volume_sma_info)

        if calculate_target == True:
            df['Target'], supertrend_target_info = signals.super_trend_strategy_v1(df, 0.008, 0.008, 0, 20, is_need_return_function_info=True)
            customized_cols_infos.append(supertrend_target_info)

    elif model_type == 'merged':
        indicator_look_back = 282
        price_look_back = 4320  

        df, lower_low_higher_high_info = indicators.add_lower_low_higher_high(df, 0.06, look_back=indicator_look_back, is_need_return_function_info=True)
        customized_cols_infos.append(lower_low_higher_high_info)

        df, macd_hist_continuity_info = indicators.calculate_macd_hist_continuity(df, is_need_return_function_info=True)
        customized_cols_infos.append(macd_hist_continuity_info)

        df, price_indicator_info = indicators.add_price_indicator(df, look_back=price_look_back, is_need_return_function_info=True)
        customized_cols_infos.append(price_indicator_info)

        df['ATR_35'], atr_info = indicators.calculate_atr(df['TR'], 35, is_need_return_function_info=True)
        customized_cols_infos.append(atr_info)

        if calculate_target == True:
            df['Target'], macd_target_info = signals.set_rsi_macd_target(df, 0.005, 0.005, 0.001, 0.006, 10, 36, is_need_return_function_info=True)
            customized_cols_infos.append(macd_target_info)

    elif model_type == 'bb':
        if calculate_target == True:
            df['Target'], bb_target_info = signals.set_bb_specific_profit_strategy(df, 0.008, 0.008, 0.012, 0.005, 16, is_need_return_function_info=True)
            customized_cols_infos.append(bb_target_info)

    elif model_type == 'supertrend_v2':
        df, lower_low_higher_high_info = indicators.add_lower_low_higher_high(df, 0.04, look_back=288, is_need_return_function_info=True)
        customized_cols_infos.append(lower_low_higher_high_info)

        df, price_indicator_info = indicators.add_price_indicator(df, look_back=480, is_need_return_function_info=True)
        customized_cols_infos.append(price_indicator_info)

        df, min_max_info = indicators.add_24h_min_max_price(df, look_back=96, is_need_return_function_info=True)
        customized_cols_infos.append(min_max_info)

        df['EMA_9'], ema_9_info = indicators.calculate_ema(df['Close'], 9, is_need_return_function_info=True)
        customized_cols_infos.append(ema_9_info)

        df['SMA_15'], sma_15_info = indicators.calculate_sma(df['Close'], 15, is_need_return_function_info=True)
        customized_cols_infos.append(sma_15_info)

        df['SMA_30'], sma_30_info = indicators.calculate_sma(df['Close'], 30, is_need_return_function_info=True)
        customized_cols_infos.append(sma_30_info)

        if calculate_target == True:
            df['Target'], supertrend_target_info = signals.super_trend_strategy_v3(df, 0.008, 0.008, 0.003, 12, is_need_return_function_info=True)
            customized_cols_infos.append(supertrend_target_info)

    # 紀錄新的cols
    modified_cols = set(df.columns)
    # 篩選多出來的cols
    new_cols = list(modified_cols - original_cols)
    return df, customized_cols_infos, new_cols

# Fetch latest data
def fetch_data(symbol, interval, data_file_name, fetch_data_length=180000, model_type=None):
    open_time_string = ''
    timezone_delta = datetime.now() - datetime.utcnow()

    # 計算時間，資料抓到上一根已經 close 的 bar
    # 假如目前為 16 分，上一根 close 的 bar 是 00 分，紀錄時段為 00-15 分
    # 資料紀錄 datetime 都以 UTC 為標準，時間轉換上須特別留意
    current_time = datetime.now()
    time_elapsed_last_bar = current_time.timestamp() % 900
    if current_time.timestamp() % 900 == 899:
        # 若執行快了 0.0xxxx 秒，會因時間差沒有更新到最新資料
        last_bar_open_time = current_time - timezone_delta - timedelta(seconds=time_elapsed_last_bar)
    else:
        last_bar_open_time = current_time - timezone_delta - timedelta(minutes=15, seconds=time_elapsed_last_bar)
    open_time_string = last_bar_open_time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'Last bar open time: {open_time_string}')

    # 讀取 local 端在 open_time_string 以前的資料
    df = get_binance_klines_backward(symbol, interval, open_time_string, fetch_data_length, get_local_file_name=data_file_name, 
                                     is_need_calculated=True)

    # 讀取最近一次更新交易資料的時間
    last_update_time_string = df.iloc[-1]['datetime']
    last_update_time = datetime.strptime(last_update_time_string, '%Y-%m-%d %H:%M:%S')
    print(f'Last update kline open time: {last_update_time}')

    # 將上根 close 的 K 線與最近一次更新的時間相減，算出有幾根 K 線要更新
    time_difference = last_bar_open_time - last_update_time
    print(f'Time difference: {time_difference} (equals {time_difference.total_seconds()} seconds)')
    klines_to_update = int(time_difference.total_seconds() // (60 * 15))

    if klines_to_update > 0:
        print(f'Updating {klines_to_update} klines...')

        # 確認 Binance server time 是否已經過了下一根 K 線的開盤時間，若否則等待
        server_time = check_server_time()

        if server_time:
            # Response server time includes timezone offset and is in milliseconds
            next_bar_open_time = last_bar_open_time + timezone_delta + timedelta(minutes=15)

            wait_time = next_bar_open_time.timestamp() - server_time / 1000

            # Add 4.5 second buffer
            wait_time = wait_time + 4.5
            print(f'Fetch kline data wait time: {wait_time} seconds')
            if wait_time > 0:
                time.sleep(wait_time)
        else:
            if env == 'development':
                print('Fetch kline data wait time: 7.2 seconds')
                time.sleep(7.2)
            elif env == 'ec2':
                print('Fetch kline data wait time: 4.5 seconds')
                time.sleep(4.5)

        # 從 binance 抓資料用原時區即可，binance 會自動將 datetime 轉換到 UTC
        last_bar_open_time_tz = last_bar_open_time + timezone_delta
        last_bar_open_time_string = last_bar_open_time_tz.strftime('%Y-%m-%d %H:%M:%S')

        # 抓取 K 線數量預留 200 根作為計算技術指標的緩衝
        # 抓完資料後，先將 Open time 欄位換算成 datetime，以方便後續做資料合併
        buffer_klines = 200
        fetch_kline_amount = (((klines_to_update + buffer_klines) // 500) + 1) * 500 if ((klines_to_update + buffer_klines) % 500) != 0 else ((klines_to_update + buffer_klines) // 500) * 500
        print(f'Fetch kline amount: {fetch_kline_amount}')

        new_data = get_binance_klines_backward(symbol, interval, last_bar_open_time_string, fetch_kline_amount, is_need_calculated=False)
        new_data['datetime'] = pd.to_datetime(new_data['Open time'], unit='ms')
        new_data.drop(columns=['Open time'], inplace=True)

        # 依據計算指標所需的資料長度，將抓取的新資料與擷取的原資料片段合併再行計算
        # 擷取的原資料片段要去除掉部分重疊的 K 線數據 - overlapped_klines
        # 1600 為部分指標計算所需暖機長度（暖機後指標數值才正確）
        overlapped_klines = fetch_kline_amount - klines_to_update
        if model_type == 'merged':
            price_look_back = 4320
            df_for_calculation = pd.concat([df[-(overlapped_klines + price_look_back + 1600): -overlapped_klines][list(new_data.columns.values)], new_data])
        else:
            df_for_calculation = pd.concat([df[-(overlapped_klines + 1600): -overlapped_klines][list(new_data.columns.values)], new_data])
        df_for_calculation.reset_index(drop=True, inplace=True)

        df_calculated = calculate_df_all_data(df_for_calculation, '', False)
        df_calculated, _, _ = customized_specific_period_col(df_calculated, model_type, calculate_target=True)

        # 確保抓資料不會不小心抓到新一根還沒 close 的 K 線
        df_calculated = df_calculated[df_calculated.datetime <= open_time_string]
        # print(f'original df:\n{df}\n')
        # print(f'df_calculated:\n{df_calculated}\n')

        # 為新資料計算完指標並整理後，與原資料進行合併並儲存
        # 多更新一根 K 線，避免抓取上根 K 線時有誤差（上根 K 線尚未 close）
        combined_data = pd.concat([df[:-1], df_calculated[-(klines_to_update + 1):]])
        combined_data.reset_index(drop=True, inplace=True)

        combined_data.to_csv(f'local_data/{data_file_name}')
        print(f'combined_data:\n{combined_data}\n')
        return combined_data
    else:
        print('No data needs to be updated')
        df = get_binance_klines_backward(symbol, interval, open_time_string, fetch_data_length, get_local_file_name=data_file_name, 
                                         is_need_calculated=True)
        print(f'\ndf:\n{df}\n')
        return df

# Fetch latest data
def fetch_data_for_two(symbol, interval, data_file_name_1, data_file_name_2, fetch_data_length=180000, 
                       model_type_1=None, model_type_2=None):
    open_time_string = ''
    timezone_delta = datetime.now() - datetime.utcnow()

    # 計算時間，資料抓到上一根已經 close 的 bar
    # 假如目前為 16 分，上一根 close 的 bar 是 00 分，紀錄時段為 00-15 分
    # 資料紀錄 datetime 都以 UTC 為標準，時間轉換上須特別留意
    current_time = datetime.now()
    time_elapsed_last_bar = current_time.timestamp() % 900
    if current_time.timestamp() % 900 == 899:
        # 若執行快了 0.0xxxx 秒，會因時間差沒有更新到最新資料
        last_bar_open_time = current_time - timezone_delta - timedelta(seconds=time_elapsed_last_bar)
    else:
        last_bar_open_time = current_time - timezone_delta - timedelta(minutes=15, seconds=time_elapsed_last_bar)
    open_time_string = last_bar_open_time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'Last bar open time: {open_time_string}')

    # 讀取 local 端在 open_time_string 以前的資料
    df_1 = get_binance_klines_backward(symbol, interval, open_time_string, fetch_data_length, get_local_file_name=data_file_name_1, 
                                       is_need_calculated=True)
    df_2 = get_binance_klines_backward(symbol, interval, open_time_string, fetch_data_length, get_local_file_name=data_file_name_2, 
                                       is_need_calculated=True)

    # 讀取最近一次更新交易資料的時間
    last_update_time_string_1 = df_1.iloc[-1]['datetime']
    last_update_time_1 = datetime.strptime(last_update_time_string_1, '%Y-%m-%d %H:%M:%S')

    last_update_time_string_2 = df_2.iloc[-1]['datetime']
    last_update_time_2 = datetime.strptime(last_update_time_string_2, '%Y-%m-%d %H:%M:%S')

    print(f'Model 1 last update kline open time: {last_update_time_1}')
    print(f'Model 2 last update kline open time: {last_update_time_2}')

    # 將上根 close 的 K 線與最近一次更新的時間相減，算出有幾根 K 線要更新
    time_difference_1 = last_bar_open_time - last_update_time_1
    time_difference_2 = last_bar_open_time - last_update_time_2

    print(f'Model 1 time difference: {time_difference_1} (equals {time_difference_1.total_seconds()} seconds)')
    print(f'Model 2 time difference: {time_difference_2} (equals {time_difference_2.total_seconds()} seconds)')

    klines_to_update_1 = int(time_difference_1.total_seconds() // (60 * 15))
    klines_to_update_2 = int(time_difference_2.total_seconds() // (60 * 15))

    if klines_to_update_1 > 0 or klines_to_update_2 > 0:
        print(f'Model 1 updating {klines_to_update_1} klines...')
        print(f'Model 2 updating {klines_to_update_2} klines...')

        # 確認 Binance server time 是否已經過了下一根 K 線的開盤時間，若否則等待
        server_time = check_server_time()

        if server_time:
            # Response server time includes timezone offset and is in milliseconds
            next_bar_open_time = last_bar_open_time + timezone_delta + timedelta(minutes=15)

            wait_time = next_bar_open_time.timestamp() - server_time / 1000

            # Add 4.5 second buffer
            wait_time = wait_time + 4.5
            print(f'Fetch kline data wait time: {wait_time} seconds')
            if wait_time > 0:
                time.sleep(wait_time)
        else:
            if env == 'development':
                print('Fetch kline data wait time: 7.2 seconds')
                time.sleep(7.2)
            elif env == 'ec2':
                print('Fetch kline data wait time: 4.5 seconds')
                time.sleep(4.5)

        # 從 binance 抓資料用原時區即可，binance 會自動將 datetime 轉換到 UTC
        last_bar_open_time_tz = last_bar_open_time + timezone_delta
        last_bar_open_time_string = last_bar_open_time_tz.strftime('%Y-%m-%d %H:%M:%S')

        # 抓取 K 線數量預留 200 根作為計算技術指標的緩衝
        # 抓完資料後，先將 Open time 欄位換算成 datetime，以方便後續做資料合併
        buffer_klines = 200
        klines_to_update = max(klines_to_update_1, klines_to_update_2)
        fetch_kline_amount = (((klines_to_update + buffer_klines) // 500) + 1) * 500 if ((klines_to_update + buffer_klines) % 500) != 0 else ((klines_to_update + buffer_klines) // 500) * 500
        print(f'Fetch kline amount: {fetch_kline_amount}')

        new_data = get_binance_klines_backward(symbol, interval, last_bar_open_time_string, fetch_kline_amount, is_need_calculated=False)
        new_data['datetime'] = pd.to_datetime(new_data['Open time'], unit='ms')
        new_data.drop(columns=['Open time'], inplace=True)

        # 依據計算指標所需的資料長度，將抓取的新資料與擷取的原資料片段合併再行計算
        # 擷取的原資料片段要去除掉部分重疊的 K 線數據 - overlapped_klines
        # 1600 為部分指標計算所需暖機長度（暖機後指標數值才正確）
        overlapped_klines_1 = fetch_kline_amount - klines_to_update_1
        if model_type_1 == 'merged':
            price_look_back = 4320
            df_for_calculation_1 = pd.concat([df_1[-(overlapped_klines_1 + price_look_back + 1600): -overlapped_klines_1][list(new_data.columns.values)], new_data])
        else:
            df_for_calculation_1 = pd.concat([df_1[-(overlapped_klines_1 + 1600): -overlapped_klines_1][list(new_data.columns.values)], new_data])
        df_for_calculation_1.reset_index(drop=True, inplace=True)

        df_calculated_1 = calculate_df_all_data(df_for_calculation_1, '', False)
        df_calculated_1, _, _ = customized_specific_period_col(df_calculated_1, model_type_1, calculate_target=True)

        overlapped_klines_2 = fetch_kline_amount - klines_to_update_2
        if model_type_2 == 'merged':
            price_look_back = 4320
            df_for_calculation_2 = pd.concat([df_2[-(overlapped_klines_2 + price_look_back + 1600): -overlapped_klines_2][list(new_data.columns.values)], new_data])
        else:
            df_for_calculation_2 = pd.concat([df_2[-(overlapped_klines_2 + 1600): -overlapped_klines_2][list(new_data.columns.values)], new_data])
        df_for_calculation_2.reset_index(drop=True, inplace=True)

        df_calculated_2 = calculate_df_all_data(df_for_calculation_2, '', False)
        df_calculated_2, _, _ = customized_specific_period_col(df_calculated_2, model_type_2, calculate_target=True)

        # 確保抓資料不會不小心抓到新一根還沒 close 的 K 線
        df_calculated_1 = df_calculated_1[df_calculated_1.datetime <= open_time_string]
        df_calculated_2 = df_calculated_2[df_calculated_2.datetime <= open_time_string]
        # print(f'original df:\n{df}\n')
        # print(f'df_calculated:\n{df_calculated}\n')

        # 為新資料計算完指標並整理後，與原資料進行合併並儲存
        # 多更新一根 K 線，避免抓取上根 K 線時有誤差（上根 K 線尚未 close）
        combined_data_1 = pd.concat([df_1[:-1], df_calculated_1[-(klines_to_update + 1):]])
        combined_data_1.reset_index(drop=True, inplace=True)

        combined_data_2 = pd.concat([df_2[:-1], df_calculated_2[-(klines_to_update + 1):]])
        combined_data_2.reset_index(drop=True, inplace=True)

        combined_data_1.to_csv(f'local_data/{data_file_name_1}')
        combined_data_2.to_csv(f'local_data/{data_file_name_2}')

        print(f'Model 1 combined_data:\n{combined_data_1}\n')
        print(f'Model 2 combined_data:\n{combined_data_2}\n')

        return combined_data_1, combined_data_2
    else:
        print('No data needs to be updated')
        df_1 = get_binance_klines_backward(symbol, interval, open_time_string, fetch_data_length, get_local_file_name=data_file_name_1, 
                                           is_need_calculated=True)
        df_2 = get_binance_klines_backward(symbol, interval, open_time_string, fetch_data_length, get_local_file_name=data_file_name_2,
                                           is_need_calculated=True)
        print(f'\nModel 1 df:\n{df_1}\n')
        print(f'\nModel 2 df:\n{df_2}\n')

        return df_1, df_2

# Fetch latest data
def fetch_data_hybrid(symbol, interval, data_file_name, fetch_data_length=180000, 
                      model_type=None, new_klines=pd.DataFrame(), return_calculation_data=False):
    open_time_string = ''
    timezone_delta = datetime.now() - datetime.utcnow()
    df_calculated = None

    # 計算時間，資料抓到上一根已經 close 的 bar
    # 假如目前為 16 分，上一根 close 的 bar 是 00 分，紀錄時段為 00-15 分
    # 資料紀錄 datetime 都以 UTC 為標準，時間轉換上須特別留意
    current_time = datetime.now()
    time_elapsed_last_bar = current_time.timestamp() % 900
    if current_time.timestamp() % 900 == 899:
        # 若執行快了 0.0xxxx 秒，會因時間差沒有更新到最新資料
        last_bar_open_time = current_time - timezone_delta - timedelta(seconds=time_elapsed_last_bar)
    else:
        last_bar_open_time = current_time - timezone_delta - timedelta(minutes=15, seconds=time_elapsed_last_bar)
    open_time_string = last_bar_open_time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'Last bar open time: {open_time_string}')

    # 讀取 local 端在 open_time_string 以前的資料
    df = get_binance_klines_backward(symbol, interval, open_time_string, fetch_data_length, get_local_file_name=data_file_name, 
                                     is_need_calculated=True)

    # 讀取最近一次更新交易資料的時間
    last_update_time_string = df.iloc[-1]['datetime']
    last_update_time = datetime.strptime(last_update_time_string, '%Y-%m-%d %H:%M:%S')

    print(f'Model {model_type} last update kline open time: {last_update_time}')

    # 將上根 close 的 K 線與最近一次更新的時間相減，算出有幾根 K 線要更新
    time_difference = last_bar_open_time - last_update_time

    print(f'Model {model_type} time difference: {time_difference} (equals {time_difference.total_seconds()} seconds)')

    klines_to_update = int(time_difference.total_seconds() // (60 * 15))

    if klines_to_update > 0:
        print(f'Model {model_type} updating {klines_to_update} klines...')

        if new_klines.empty:
            # 確認 Binance server time 是否已經過了下一根 K 線的開盤時間，若否則等待
            server_time = check_server_time()

            if server_time:
                # Response server time includes timezone offset and is in milliseconds
                next_bar_open_time = last_bar_open_time + timezone_delta + timedelta(minutes=15)

                wait_time = next_bar_open_time.timestamp() - server_time / 1000

                # Add 4.5 second buffer
                wait_time = wait_time + 4.5
                print(f'Fetch kline data wait time: {wait_time} seconds')
                if wait_time > 0:
                    time.sleep(wait_time)
            else:
                if env == 'development':
                    print('Fetch kline data wait time: 7.2 seconds')
                    time.sleep(7.2)
                elif env == 'ec2':
                    print('Fetch kline data wait time: 4.5 seconds')
                    time.sleep(4.5)

            # 從 binance 抓資料用原時區即可，binance 會自動將 datetime 轉換到 UTC
            last_bar_open_time_tz = last_bar_open_time + timezone_delta
            last_bar_open_time_string = last_bar_open_time_tz.strftime('%Y-%m-%d %H:%M:%S')

            # 抓取 K 線數量預留 200 根作為計算技術指標的緩衝
            # 抓完資料後，先將 Open time 欄位換算成 datetime，以方便後續做資料合併
            buffer_klines = 200
            fetch_kline_amount = (((klines_to_update + buffer_klines) // 500) + 1) * 500 if ((klines_to_update + buffer_klines) % 500) != 0 else ((klines_to_update + buffer_klines) // 500) * 500
            print(f'Fetch kline amount: {fetch_kline_amount}')

            new_data = get_binance_klines_backward(symbol, interval, last_bar_open_time_string, fetch_kline_amount, is_need_calculated=False)
            new_data['datetime'] = pd.to_datetime(new_data['Open time'], unit='ms')
            new_data.drop(columns=['Open time'], inplace=True)

            # 依據計算指標所需的資料長度，將抓取的新資料與擷取的原資料片段合併再行計算
            # 擷取的原資料片段要去除掉部分重疊的 K 線數據 - overlapped_klines
            # 1600 為部分指標計算所需暖機長度（暖機後指標數值才正確）
            overlapped_klines = fetch_kline_amount - klines_to_update
            df_for_calculation = pd.concat([df[-(overlapped_klines + 1600): -overlapped_klines][list(new_data.columns.values)], new_data])
            df_for_calculation.reset_index(drop=True, inplace=True)

            df_calculated = calculate_df_all_data(df_for_calculation, '', False)
            df_return_calculated = df_calculated.copy()
            df_customized, _, _ = customized_specific_period_col(df_calculated, model_type, calculate_target=True)
        else:
            df_customized, _, _ = customized_specific_period_col(new_klines, model_type, calculate_target=True)

        # 確保抓資料不會不小心抓到新一根還沒 close 的 K 線
        df_customized = df_customized[df_customized.datetime <= open_time_string]
        # print(f'original df:\n{df}\n')
        # print(f'df_calculated:\n{df_calculated}\n')

        # 為新資料計算完指標並整理後，與原資料進行合併並儲存
        # 多更新一根 K 線，避免抓取上根 K 線時有誤差（上根 K 線尚未 close）
        combined_data = pd.concat([df[:-1], df_customized[-(klines_to_update + 1):]])
        combined_data.reset_index(drop=True, inplace=True)

        combined_data.to_csv(f'local_data/{data_file_name}')
        print(f'Model {model_type} combined_data:\n{combined_data}\n')

        if return_calculation_data == True:
            return combined_data, df_return_calculated
        else:
            return combined_data
    else:
        print('No data needs to be updated')
        df = get_binance_klines_backward(symbol, interval, open_time_string, fetch_data_length, get_local_file_name=data_file_name, 
                                         is_need_calculated=True)
        print(f'\nModel {model_type} df:\n{df}\n')

        if return_calculation_data == True:
            return df, df_calculated
        else:
            return df

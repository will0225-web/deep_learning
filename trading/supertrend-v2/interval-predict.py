import os, sys
sys.path.append(os.getcwd())

import time
import ast

from dotenv import load_dotenv

load_dotenv()

from trading.utils.data import fetch_data
from trading.utils.trading import send_order, send_trading_record, query_user_order, login_server, get_strategy_user, query_user_trade
from trading.utils.predict import predict_result
from trading.utils.record import save_predicted_result, save_position_history, check_current_position

# 定義 model 名稱與檔案位置
model_name = 'supertrend_v2'

model_path = 'trained_models/1713239614_softmax_loss-1.1766_accuracy-0.5985_f1-0.1456'
data_file_name = 'ETHUSDT_15m_realtime_caculated_614.csv'

fetch_data_length = 300000

# 交易標的與週期
symbol = 'ETHUSDT'
interval = '15m'

# 策略參數設定
# 模型預測結果之條件參數
predicted_long_results = [1, 3, 5, 7, 9]
predicted_short_results = [2, 4, 6, 8, 10]

volume_threshold = 20000

# 模型交易策略參數設定
model_sl = 0.008

hold_count_limit = 12
loss_hold_count_limit = 5
no_keep_update_max_profit_count_limit = 4

# Determine trade actions
def execute_trade(predicted_result, confidence, data, model_type=None, stop_loss=None):
    if predicted_result in predicted_long_results or predicted_result in predicted_short_results:
        # Determine entry confidence and take profit based on predicted result and confidence
        meet_entry_confidence, take_profit = check_entry_confidence_and_set_TP(predicted_result, confidence)

        # Check whether entry conditions are met
        if meet_entry_confidence == True:
            volume = data.iloc[-1]['Volume']
            supertrend = data.iloc[-1]['Super Trend']
            entry_position = check_entry_conditions(predicted_result, volume, supertrend)
        else:
            entry_position = ''
            print('Entry confidence is not met, so trade will not be executed.')

        if entry_position != '':
            print(f'{entry_position} entry')
            sl_price = None
            current_price = data.iloc[-1]['Close']

            if entry_position == 'Long':
                sl_price = current_price * (1 - stop_loss)
            elif entry_position == 'Short':
                sl_price = current_price * (1 + stop_loss)

            trailing_threshold, callback_rate = set_trailing_stop(take_profit)
            no_keep_trend_threshold = [0.003, trailing_threshold / 100]

            # Send order
            send_order(symbol, f'{entry_position} with TP', model_type, sl_value=stop_loss * 100, sl_price=sl_price, tp_value=take_profit * 100, 
                       trailing_activation=trailing_threshold, trailing_offset=callback_rate, current_price=current_price)

            # Save current position info to csv
            save_position_history(symbol, entry_position, entry_price=current_price, take_profit=take_profit, stop_loss=stop_loss,
                                  trailing_stop_threshold=trailing_threshold, callback_rate=callback_rate, entry_prediction=predicted_result,
                                  no_keep_trend_threshold=no_keep_trend_threshold)
        else:
            print('Entry conditions are not met, so trade will not be executed.')
    else:
        print('No action')

def check_entry_confidence_and_set_TP(predicted_result, confidence):
    meet_entry_confidence = False
    take_profit = 0

    if predicted_result == 1 and not(0.36 <= confidence <= 0.375):
        if 0.34 <= confidence <= 0.44:
            meet_entry_confidence = True
            take_profit = 0.015
        elif 0.23 <= confidence <= 0.26:
            meet_entry_confidence = True
            take_profit = 0.03
    elif predicted_result == 2 and not(0.35 <= confidence <= 0.36):
        if 0.32 <= confidence <= 0.37:
            meet_entry_confidence = True
            take_profit = 0.01
        elif 0.21 <= confidence <= 0.24:
            meet_entry_confidence = True
            take_profit = 0.01
    elif predicted_result == 3 and not(0.258 <= confidence <= 0.264 or 0.27 <= confidence <= 0.278):
        if 0.252 <= confidence <= 0.28:
            meet_entry_confidence = True
            take_profit = 0.015
    elif predicted_result == 4 and not(0.255 <= confidence <= 0.263): # 需調整，容易輸
        if 0.245 <= confidence <= 0.25:
            meet_entry_confidence = True
            take_profit = 0.015
        elif 0.255 <= confidence <= 0.265:
            meet_entry_confidence = True
            take_profit = 0.01
        elif 0.315 <= confidence <= 0.32:
            meet_entry_confidence = True
            take_profit = 0.015
    elif predicted_result == 6 and not(0.246 <= confidence <= 0.252 or 0.254 <= confidence <= 0.259):
        if 0.243 <= confidence <= 0.265:
            meet_entry_confidence = True
            take_profit = 0.015
    elif predicted_result == 7 and not(0.47 <= confidence <= 0.475 or 0.39 <= confidence <= 0.44):
        if 0.46 <= confidence <= 0.49:
            meet_entry_confidence = True
            take_profit = 0.025 # 或0.03
        elif 0.37 <= confidence <= 0.49:
            meet_entry_confidence = True
            take_profit = 0.015 # 或0.01(0.01比較保險)
    elif predicted_result == 8 and not(0.44 <= confidence <= 0.47):
        if 0.4 <= confidence <= 0.53:
            meet_entry_confidence = True
            take_profit = 0.02
    elif predicted_result == 9:
        if 0.2 <= confidence <= 0.27:
            meet_entry_confidence = True
            take_profit = 0.02
    elif predicted_result == 10 and not(0.246 <= confidence <= 0.25):
        if 0.243 <= confidence <= 0.255:
            meet_entry_confidence = True
            take_profit = 0.01

    return meet_entry_confidence, take_profit

def check_entry_conditions(predicted_result, volume=None, supertrend=None):
    entry_position = ''
    
    if predicted_result in predicted_long_results and supertrend == 1 and volume > volume_threshold:
        entry_position = 'Long'
    elif predicted_result in predicted_short_results and supertrend == -1 and volume > volume_threshold:
        entry_position = 'Short'
    else:
        pass

    return entry_position

def set_trailing_stop(take_profit):
    trailing_threshold = take_profit / 1.5 * 100
    trailing_return_threshold = round(trailing_threshold / 2, 3)
    callback_rate = max(trailing_return_threshold, 0.4)

    return trailing_threshold, callback_rate

def close_position(data, position, position_info={}, model_type=None):
    exit_condition = None
    current_position_csv = os.path.join(os.path.abspath(os.path.dirname('__file__')), 'records/current_position.csv')

    # Read current market data and position info to determine exit conditions
    current_rsi = round(data.iloc[-1]['RSI'], 2)
    current_price = data.iloc[-1]['Close']
    current_high = data.iloc[-1]['High']
    current_low = data.iloc[-1]['Low']
    print(f'Current RSI: {current_rsi}')
    print(f'Current Price: {current_price}')

    entry_price = position_info.iloc[-1]['entry_price']
    max_profit = position_info.iloc[-1]['max_profit']
    stop_loss = position_info.iloc[-1]['stop_loss']
    take_profit = position_info.iloc[-1]['take_profit']
    trailing_threshold = position_info.iloc[-1]['trailing_stop_threshold']
    callback_rate = position_info.iloc[-1]['callback_rate']
    loss_hold_count = position_info.iloc[-1]['loss_hold_count']
    no_keep_update_max_profit_count = position_info.iloc[-1]['no_keep_update_max_profit_count']
    # Convert no_keep_trend_threshold from string to list
    no_keep_trend_threshold = position_info['no_keep_trend_threshold'].apply(ast.literal_eval)[0]
    entry_prediction = position_info.iloc[-1]['entry_prediction']
    print(f'Entry Prediction: {entry_prediction}, Take Profit: {take_profit * 100} %')

    hold_count = position_info.loc[0, 'hold_count'] + 1
    position_info.loc[0, 'hold_count'] = hold_count
    position_info.to_csv(current_position_csv, index=False)

    # Print current position info
    print(f"Position found: {'Long' if position == 1 else 'Short'}. Current hold count: {hold_count}. Entry price: {entry_price}")

    # Long position exit conditions
    if position == 1:
        # calculate max profit
        current_profit = round((current_high - entry_price) / entry_price * 100, 2)
        if current_profit > max_profit:
            max_profit = max(max_profit, current_profit)
            position_info.loc[0, 'max_profit'] = max_profit

            if no_keep_update_max_profit_count != 0:
                no_keep_update_max_profit_count = 0
                position_info.loc[0, 'no_keep_update_max_profit_count'] = no_keep_update_max_profit_count

            position_info.to_csv(current_position_csv, index=False)
            print(f'Max Profit Increased: {max_profit}')
        else:
            no_keep_update_max_profit_count += 1
            position_info.loc[0, 'no_keep_update_max_profit_count'] = no_keep_update_max_profit_count
            position_info.to_csv(current_position_csv, index=False)
            print(f'No Keep Update Max Profit Count: {no_keep_update_max_profit_count}')
            print(f'Max Profit: {max_profit}')

        # calculate loss hold count
        if current_price <= entry_price:
            loss_hold_count += 1
            position_info.loc[0, 'loss_hold_count'] = loss_hold_count
            position_info.to_csv(current_position_csv, index=False)
            print(f'Loss Hold Count: {loss_hold_count}')
        else:
            if loss_hold_count != 0:
                loss_hold_count = 0
                position_info.loc[0, 'loss_hold_count'] = loss_hold_count
                position_info.to_csv(current_position_csv, index=False)
                print(f'Loss Hold Count: {loss_hold_count}')

        if current_low <= ((1 - stop_loss) * entry_price):
            exit_condition = 'Stop Loss Long'
            execute_exit_trade(symbol, 'Exit Long', take_profit=take_profit, stop_loss=stop_loss, close_price=current_low, exit_condition=exit_condition, model_type=model_type)
        elif hold_count_limit > 0 and hold_count >= hold_count_limit:
            exit_condition = 'Long Hold Count Limit Reached'
            execute_exit_trade(symbol, 'Exit Long', take_profit=take_profit, stop_loss=stop_loss, close_price=current_price, exit_condition=exit_condition, model_type=model_type)
        elif take_profit > 0 and current_high >= ((1 + take_profit) * entry_price):
            exit_condition = 'Take Profit Long'
            execute_exit_trade(symbol, 'Exit Long', take_profit=take_profit, stop_loss=stop_loss, close_price=current_high, exit_condition=exit_condition, model_type=model_type)
        elif (trailing_threshold > 0 and callback_rate > 0) and max_profit >= trailing_threshold and ((current_low - entry_price) / entry_price * 100) <= (max_profit - callback_rate):
            server_token = login_server()
            strategy_user = get_strategy_user(model=model_name, token=server_token)
            response = query_user_order(strategy_user=strategy_user, symbol=symbol, order_id='trailing_stop_order', token=server_token)
            if response is not None and 'orderId' in response:
                print('Trailing stop order on Binance has not been triggered yet.')
            else:
                exit_condition = 'Long Trailing Stop'
                response = query_user_trade(strategy_user=strategy_user, symbol=symbol, token=server_token)
                if response and float(response[-1]['realizedPnl']) != 0:
                    trailing_close_price = float(response[-1]['price'])
                else:
                    trailing_close_price = entry_price * (1 + (max_profit - callback_rate) / 100)
                execute_exit_trade(symbol, 'Exit Long', take_profit=take_profit, stop_loss=stop_loss, close_price=trailing_close_price, exit_condition=exit_condition, model_type=model_type, token=server_token)
        elif loss_hold_count_limit > 0 and loss_hold_count >= loss_hold_count_limit:
            exit_condition = 'Long Loss Hold Count Limit Reached'
            execute_exit_trade(symbol, 'Exit Long', take_profit=take_profit, stop_loss=stop_loss, close_price=current_price, exit_condition=exit_condition, model_type=model_type)
        elif (no_keep_update_max_profit_count_limit > 0 and no_keep_trend_threshold != []) and no_keep_trend_threshold[0] <= (max_profit / 100) <= no_keep_trend_threshold[1] and no_keep_update_max_profit_count >= no_keep_update_max_profit_count_limit:
            exit_condition = 'Long No Keep Update Max Profit Count Limit Reached'
            execute_exit_trade(symbol, 'Exit Long', take_profit=take_profit, stop_loss=stop_loss, close_price=current_price, exit_condition=exit_condition, model_type=model_type)

        if exit_condition is not None:
            print(f'Exit Condition Met: {exit_condition}')

    # Short position exit conditions
    elif position == 2:
        # calculate max profit
        current_profit = round((entry_price - current_low) / entry_price * 100, 2)
        if current_profit > max_profit:
            max_profit = max(max_profit, current_profit)
            position_info.loc[0, 'max_profit'] = max_profit

            if no_keep_update_max_profit_count != 0:
                no_keep_update_max_profit_count = 0
                position_info.loc[0, 'no_keep_update_max_profit_count'] = no_keep_update_max_profit_count

            position_info.to_csv(current_position_csv, index=False)
            print(f'Max Profit Increased: {max_profit}')
        else:
            no_keep_update_max_profit_count += 1
            position_info.loc[0, 'no_keep_update_max_profit_count'] = no_keep_update_max_profit_count
            position_info.to_csv(current_position_csv, index=False)
            print(f'No Keep Update Max Profit Count: {no_keep_update_max_profit_count}')
            print(f'Max Profit: {max_profit}')

        # calculate loss hold count
        if current_price >= entry_price:
            loss_hold_count += 1
            position_info.loc[0, 'loss_hold_count'] = loss_hold_count
            position_info.to_csv(current_position_csv, index=False)
            print(f'Loss Hold Count: {loss_hold_count}')
        else:
            if loss_hold_count != 0:
                loss_hold_count = 0
                position_info.loc[0, 'loss_hold_count'] = loss_hold_count
                position_info.to_csv(current_position_csv, index=False)
                print(f'Loss Hold Count: {loss_hold_count}')

        if current_high >= ((1 + stop_loss) * entry_price):
            exit_condition = 'Stop Loss Short'
            execute_exit_trade(symbol, 'Exit Short', take_profit=take_profit, stop_loss=stop_loss, close_price=current_high, exit_condition=exit_condition, model_type=model_type)
        elif hold_count_limit > 0 and hold_count >= hold_count_limit:
            exit_condition = 'Short Hold Count Limit Reached'
            execute_exit_trade(symbol, 'Exit Short', take_profit=take_profit, stop_loss=stop_loss, close_price=current_price, exit_condition=exit_condition, model_type=model_type)
        elif take_profit > 0 and current_low <= ((1 - take_profit) * entry_price):
            exit_condition = 'Take Profit Short'
            execute_exit_trade(symbol, 'Exit Short', take_profit=take_profit, stop_loss=stop_loss, close_price=current_low, exit_condition=exit_condition, model_type=model_type)
        elif (trailing_threshold > 0 and callback_rate > 0) and max_profit >= trailing_threshold and ((entry_price - current_high) / entry_price * 100) <= (max_profit - callback_rate):
            server_token = login_server()
            strategy_user = get_strategy_user(model=model_name, token=server_token)
            response = query_user_order(strategy_user=strategy_user, symbol=symbol, order_id='trailing_stop_order', token=server_token)
            if response is not None and 'orderId' in response:
                print('Trailing stop order on Binance has not been triggered yet.')
            else:
                exit_condition = 'Short Trailing Stop'
                response = query_user_trade(strategy_user=strategy_user, symbol=symbol, token=server_token)
                if response and float(response[-1]['realizedPnl']) != 0:
                    trailing_close_price = float(response[-1]['price'])
                else:
                    trailing_close_price = entry_price * (1 - (max_profit - callback_rate) / 100)
                execute_exit_trade(symbol, 'Exit Short', take_profit=take_profit, stop_loss=stop_loss, close_price=trailing_close_price, exit_condition=exit_condition, model_type=model_type, token=server_token)
        elif loss_hold_count_limit > 0 and loss_hold_count >= loss_hold_count_limit:
            exit_condition = 'Short Loss Hold Count Limit Reached'
            execute_exit_trade(symbol, 'Exit Short', take_profit=take_profit, stop_loss=stop_loss, close_price=current_price, exit_condition=exit_condition, model_type=model_type)
        elif (no_keep_update_max_profit_count_limit > 0 and no_keep_trend_threshold != []) and no_keep_trend_threshold[0] <= (max_profit / 100) <= no_keep_trend_threshold[1] and no_keep_update_max_profit_count >= no_keep_update_max_profit_count_limit:
            exit_condition = 'Short No Keep Update Max Profit Count Limit Reached'
            execute_exit_trade(symbol, 'Exit Short', take_profit=take_profit, stop_loss=stop_loss, close_price=current_price, exit_condition=exit_condition, model_type=model_type)

        if exit_condition is not None:
            print(f'Exit Condition Met: {exit_condition}')

def execute_exit_trade(symbol, action, take_profit=None, stop_loss=None, close_price=None, exit_condition=None, model_type=None, token=None):
    send_order(symbol, action, model_type)
    data = save_position_history(symbol, action, take_profit=take_profit, stop_loss=stop_loss, close_price=close_price, exit_condition=exit_condition)
    send_trading_record(data, model_type, token)


# 抓取最新資料並更新目前倉位
fetch_data_start = time.time()
data = fetch_data(symbol, interval, data_file_name, fetch_data_length, model_type=model_name)
fetch_data_end = time.time()
print(f'Time elapsed during data fetch: {fetch_data_end - fetch_data_start}')
position, position_info = check_current_position()

# 只有在每剛經過 15 分時才執行交易相關動作
# 假如目前未持倉，則進行模型預測；已持倉的話，則判斷是否達到出場條件
if time.time() % 900 < 50:
    if position == 0:
        print('No position found. Start predicting...')
        model_predict_start = time.time()
        predicted_result, confidence = predict_result(model_path, data, model_type=model_name)
        model_predict_end = time.time()
        print(f'Time elapsed during model prediction: {model_predict_end - model_predict_start}')
        print(f'Predicted result: {predicted_result} with confidence {confidence}')
        save_predicted_result(predicted_result, confidence, model_type=model_name)
        execute_trade(predicted_result, confidence, data, model_type=model_name, stop_loss=model_sl)
    else:
        close_position(data, position, position_info, model_type=model_name)
else:
    print('Trade will only execute at 15, 30, 45, 00 in every hour')

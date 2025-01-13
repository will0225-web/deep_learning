import os, sys
sys.path.append(os.getcwd())

import time
from dotenv import load_dotenv

load_dotenv()

from trading.utils.data import fetch_data
from trading.utils.trading import send_order, send_trading_record, query_user_order, login_server, get_strategy_user, query_user_trade
from trading.utils.predict import predict_result
from trading.utils.record import save_predicted_result, save_position_history, check_current_position

# 定義 model 名稱與檔案位置
model_name = 'supertrend'

model_path = 'trained_models/1710835373_softmax_loss-1.2277_accuracy-0.4738_f1-0.2171_real_best'
data_file_name = 'ETHUSDT_15m_realtime_caculated_373.csv'

fetch_data_length = 300000

# 交易標的與週期
symbol = 'ETHUSDT'
interval = '15m'

# 策略參數設定
# 模型預測結果之條件參數
predicted_long_results = [0, 2, 4]
predicted_short_results = [1, 3, 5]

entry_confidence_0 = (0.65, 0.85)
entry_confidence_2 = (0.5, 0.6)
entry_confidence_4 = (0.6, 0.7)

entry_confidence_1 = (0.55, 0.75)
entry_confidence_3 = (0.4, 0.5)
entry_confidence_5 = (0.45, 0.55)

# 模型交易策略參數設定
model_tp_0_1 = 0.02
model_tp_2_3 = 0.03
model_tp_4_5 = 0.03
model_sl = 0.008

trailing_threshold = 1
trailing_return_threshold = 0.4

detect_is_need_exit_threshold = 0.8
if_win_at_least_profit = 0.3

hold_count_limit = 20
loss_hold_count_limit = 6

run_up_stop_activation = 0.8
run_up_constant = 35

# Determine trade actions
def execute_trade(predicted_result, confidence, data, model_type=None, stop_loss=None, take_profit=None, trailing_threshold=None, trailing_return_threshold=None):
    entry_position = ''

    if predicted_result in predicted_long_results or predicted_result in predicted_short_results:
        # Check whether entry conditions are met
        if model_type == 'supertrend':
            volume = data.iloc[-1]['Volume']
            supertrend = data.iloc[-1]['Super Trend']
            meet_entry_conditions = check_entry_conditions(predicted_result, confidence, volume, supertrend)
        else:
            meet_entry_conditions = check_entry_conditions(predicted_result, confidence)

        if predicted_result in predicted_long_results and meet_entry_conditions:
            entry_position = 'Long'
        elif predicted_result in predicted_short_results and meet_entry_conditions:
            entry_position = 'Short'
        else:
            print('Entry conditions were not met, so trade will not be executed.')

        if entry_position != '':
            print(f'{entry_position} entry')
            sl_price = None
            current_price, rsi_peak = data.iloc[-1]['Close'], data.iloc[-1]['RSI']

            if entry_position == 'Long':
                sl_price = current_price * (1 - stop_loss)
            elif entry_position == 'Short':
                sl_price = current_price * (1 + stop_loss)

            # Send order depending on different models
            if model_type in ['merged', 'supertrend']:
                take_profit = set_position_tp(predicted_result, model_type)

                if model_type == 'supertrend':
                    send_order(symbol, f'{entry_position} with TP', model_type, sl_value=stop_loss * 100, sl_price=sl_price, tp_value=take_profit * 100, 
                               trailing_activation=trailing_threshold, trailing_offset=trailing_return_threshold, current_price=current_price)
                else:
                    send_order(symbol, f'{entry_position} with TP', model_type, sl_value=stop_loss * 100, sl_price=sl_price, tp_value=take_profit * 100,
                               current_price=current_price)
            else:
                send_order(symbol, entry_position, model_type, sl_value=stop_loss * 100, sl_price=sl_price, current_price=current_price)

            # Save current position info to csv
            save_position_history(symbol, entry_position, entry_price=current_price, take_profit=take_profit, stop_loss=stop_loss,
                                  rsi_peak=rsi_peak, entry_prediction=predicted_result)
    else:
        print('No action')

def set_position_tp(entry_prediction, model_type=None):
    global model_tp
    take_profit = 0
    if model_type == 'supertrend':
        if entry_prediction in [0, 1]:
            take_profit = model_tp_0_1
        elif entry_prediction in [2, 3]:
            take_profit = model_tp_2_3
        elif entry_prediction in [4, 5]:
            take_profit = model_tp_4_5
        print(f'Take Profit set to {take_profit * 100} %')
    else:
        take_profit = model_tp
        print(f'Take Profit set to {take_profit * 100} %')
    return take_profit 

def check_entry_conditions(predicted_result, confidence, volume=None, supertrend=None, kline_color=None):
    if supertrend == 1:
        if predicted_result == 0 and entry_confidence_0[0] <= confidence <= entry_confidence_0[1] and volume >= 35000:
            return True
        elif predicted_result == 2 and entry_confidence_2[0] <= confidence <= entry_confidence_2[1] and volume >= 45000:
            return True
        elif predicted_result == 4 and entry_confidence_4[0] <= confidence <= entry_confidence_4[1] and volume >= 45000:
            return True
        else:
            return False
    elif supertrend == -1:
        if predicted_result == 1 and entry_confidence_1[0] <= confidence <= entry_confidence_1[1] and volume >= 35000:
            return True
        elif predicted_result == 3 and entry_confidence_3[0] <= confidence <= entry_confidence_3[1] and volume >= 45000:
            return True
        elif predicted_result == 5 and entry_confidence_5[0] <= confidence <= entry_confidence_5[1] and volume >= 45000:
            return True
        else:
            return False
    else:
        return False

def close_position(data, position, position_info={}, model_type=None):
    exit_condition = None
    runup_stop_price = 0
    current_position_csv = os.path.join(os.path.abspath(os.path.dirname('__file__')), 'records/current_position.csv')

    # Read current market data and position info to determine exit conditions
    current_rsi = round(data.iloc[-1]['RSI'], 2)
    current_price = data.iloc[-1]['Close']
    current_high = data.iloc[-1]['High']
    current_low = data.iloc[-1]['Low']
    current_open = data.iloc[-1]['Open']
    print(f'Current RSI: {current_rsi}')
    print(f'Current Price: {current_price}')

    entry_price = position_info.iloc[-1]['entry_price']
    max_profit = position_info.iloc[-1]['max_profit']
    stop_loss = position_info.iloc[-1]['stop_loss']
    take_profit = position_info.iloc[-1]['take_profit']
    loss_hold_count = position_info.iloc[-1]['loss_hold_count']
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
            position_info.to_csv(current_position_csv, index=False)
            print(f'Max Profit Increased: {max_profit}')
        else:
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

        # calculate run-up stop price
        if max_profit >= run_up_stop_activation:
            runup_stop_price = entry_price * (1 - stop_loss) * (1 + ((max_profit / 100) / run_up_constant * hold_count))

        if current_low <= ((1 - stop_loss) * entry_price):
            exit_condition = 'Stop Loss Long'
            execute_exit_trade(symbol, 'Exit Long', take_profit=take_profit, stop_loss=stop_loss, close_price=current_low, exit_condition=exit_condition, model_type=model_type)
        elif hold_count >= hold_count_limit:
            exit_condition = 'Long Hold Count Limit Reached'
            execute_exit_trade(symbol, 'Exit Long', take_profit=take_profit, stop_loss=stop_loss, close_price=current_price, exit_condition=exit_condition, model_type=model_type)
        elif take_profit > 0 and current_high >= ((1 + take_profit) * entry_price):
            exit_condition = 'Take Profit Long'
            execute_exit_trade(symbol, 'Exit Long', take_profit=take_profit, stop_loss=stop_loss, close_price=current_high, exit_condition=exit_condition, model_type=model_type)
        elif max_profit >= trailing_threshold and ((current_low - entry_price) / entry_price * 100) <= (max_profit - trailing_return_threshold):
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
                    trailing_close_price = entry_price * (1 + (max_profit - trailing_return_threshold) / 100)
                execute_exit_trade(symbol, 'Exit Long', take_profit=take_profit, stop_loss=stop_loss, close_price=trailing_close_price, exit_condition=exit_condition, model_type=model_type, token=server_token)
        elif max_profit >= detect_is_need_exit_threshold and (((current_price - entry_price) / entry_price * 100) <= if_win_at_least_profit):
            exit_condition = 'Long take at least profit'
            execute_exit_trade(symbol, 'Exit Long', take_profit=take_profit, stop_loss=stop_loss, close_price=current_price, exit_condition=exit_condition, model_type=model_type)
        elif max_profit >= run_up_stop_activation and (current_low <= runup_stop_price or current_open <= runup_stop_price):
            exit_condition = 'Long Run-up Stop'
            execute_exit_trade(symbol, 'Exit Long', take_profit=take_profit, stop_loss=stop_loss, close_price=runup_stop_price, exit_condition=exit_condition, model_type=model_type)
        elif loss_hold_count >= loss_hold_count_limit:
            exit_condition = 'Long Loss Hold Count Limit Reached'
            execute_exit_trade(symbol, 'Exit Long', take_profit=take_profit, stop_loss=stop_loss, close_price=current_price, exit_condition=exit_condition, model_type=model_type)

        if exit_condition is not None:
            print(f'Exit Condition Met: {exit_condition}')
        elif exit_condition is None and runup_stop_price != 0:
            send_order(symbol, 'Long Set Run-up Stop', model_type, runup_stop_price=runup_stop_price, current_price=current_price)

    # Short position exit conditions
    elif position == 2:
        # calculate max profit
        current_profit = round((entry_price - current_low) / entry_price * 100, 2)
        if current_profit > max_profit:
            max_profit = max(max_profit, current_profit)
            position_info.loc[0, 'max_profit'] = max_profit
            position_info.to_csv(current_position_csv, index=False)
            print(f'Max Profit Increased: {max_profit}')
        else:
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

        # calculate run-up stop price
        if max_profit >= run_up_stop_activation:
            runup_stop_price = entry_price * (1 + stop_loss) * (1 - ((max_profit / 100) / run_up_constant * hold_count))

        if current_high >= ((1 + stop_loss) * entry_price):
            exit_condition = 'Stop Loss Short'
            execute_exit_trade(symbol, 'Exit Short', take_profit=take_profit, stop_loss=stop_loss, close_price=current_high, exit_condition=exit_condition, model_type=model_type)
        elif hold_count >= hold_count_limit:
            exit_condition = 'Short Hold Count Limit Reached'
            execute_exit_trade(symbol, 'Exit Short', take_profit=take_profit, stop_loss=stop_loss, close_price=current_price, exit_condition=exit_condition, model_type=model_type)
        elif take_profit > 0 and current_low <= ((1 - take_profit) * entry_price):
            exit_condition = 'Take Profit Short'
            execute_exit_trade(symbol, 'Exit Short', take_profit=take_profit, stop_loss=stop_loss, close_price=current_low, exit_condition=exit_condition, model_type=model_type)
        elif max_profit >= trailing_threshold and ((entry_price - current_high) / entry_price * 100) <= (max_profit - trailing_return_threshold):
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
                    trailing_close_price = entry_price * (1 - (max_profit - trailing_return_threshold) / 100)
                execute_exit_trade(symbol, 'Exit Short', take_profit=take_profit, stop_loss=stop_loss, close_price=trailing_close_price, exit_condition=exit_condition, model_type=model_type, token=server_token)
        elif max_profit >= detect_is_need_exit_threshold and (((entry_price - current_price) / entry_price * 100) <= if_win_at_least_profit):
            exit_condition = 'Short take at least profit'
            execute_exit_trade(symbol, 'Exit Short', take_profit=take_profit, stop_loss=stop_loss, close_price=current_price, exit_condition=exit_condition, model_type=model_type)
        elif max_profit >= run_up_stop_activation and (current_high >= runup_stop_price or current_open >= runup_stop_price):
            exit_condition = 'Short Run-up Stop'
            execute_exit_trade(symbol, 'Exit Short', take_profit=take_profit, stop_loss=stop_loss, close_price=runup_stop_price, exit_condition=exit_condition, model_type=model_type)
        elif loss_hold_count >= loss_hold_count_limit:
            exit_condition = 'Short Loss Hold Count Limit Reached'
            execute_exit_trade(symbol, 'Exit Short', take_profit=take_profit, stop_loss=stop_loss, close_price=current_price, exit_condition=exit_condition, model_type=model_type)

        if exit_condition is not None:
            print(f'Exit Condition Met: {exit_condition}')
        elif exit_condition is None and runup_stop_price != 0:
            send_order(symbol, 'Short Set Run-up Stop', model_type, runup_stop_price=runup_stop_price, current_price=current_price)

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
        predicted_result, confidence = predict_result(model_path, data, model_type=model_name)
        print(f'Predicted result: {predicted_result} with confidence {confidence}')
        save_predicted_result(predicted_result, confidence, model_type=model_name)
        execute_trade(predicted_result, confidence, data, model_type=model_name, stop_loss=model_sl,
                        trailing_threshold=trailing_threshold, trailing_return_threshold=trailing_return_threshold)
    else:
        close_position(data, position, position_info, model_type=model_name)
else:
    print('Trade will only execute at 15, 30, 45, 00 in every hour')

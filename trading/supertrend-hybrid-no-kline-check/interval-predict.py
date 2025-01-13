import os, sys
sys.path.append(os.getcwd())

import time
from dotenv import load_dotenv

load_dotenv()

from trading.utils.data import fetch_data_hybrid
from trading.utils.trading import send_order, send_trading_record, query_user_order, login_server, get_strategy_user, query_user_trade
from trading.utils.predict import predict_result
from trading.utils.record import save_predicted_result, save_position_history, check_current_position

# 定義 model 名稱與檔案位置
v1_model_name = 'supertrend_hybrid_without_kline_check'
v2_model_name = 'supertrend_v2'

v1_data_file_name = 'ETHUSDT_15m_supertrend_hybrid_v1.csv'
v2_data_file_name = 'ETHUSDT_15m_supertrend_hybrid_v2.csv'

v1_model_path = 'trained_models/1710835373_softmax_loss-1.2277_accuracy-0.4738_f1-0.2171_real_best'
v2_model_path = 'trained_models/1713239614_softmax_loss-1.1766_accuracy-0.5985_f1-0.1456'

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
model_tp_0_1 = 0.025
model_tp_2_3 = 0.03
model_tp_4_5 = 0.03
model_sl = 0.008

v1_trailing_threshold = 1
v1_trailing_return_threshold = 0.4

hold_count_limit = 49

# Determine trade actions
def execute_trade(predicted_result, confidence, data, model_type=None, stop_loss=None, take_profit=None, 
                  trailing_threshold=None, trailing_return_threshold=None):
    entry_position = ''

    if predicted_result in predicted_long_results or predicted_result in predicted_short_results:
        # Check whether entry conditions are met
        volume = data.iloc[-1]['Volume']
        supertrend = data.iloc[-1]['Super Trend']
        kline_color = data.iloc[-1]['kline_color']
        meet_entry_conditions = check_entry_conditions(predicted_result, confidence, volume, supertrend, kline_color)

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

            # Set tp and trailing stop values
            take_profit = set_position_tp(predicted_result, model_type)

            trailing_threshold = v1_trailing_threshold
            trailing_return_threshold = v1_trailing_return_threshold           

            # Send order to Trading Server
            send_order(symbol, f'{entry_position} with TP', model_type, sl_value=stop_loss * 100, sl_price=sl_price, tp_value=take_profit * 100, 
                       trailing_activation=trailing_threshold, trailing_offset=trailing_return_threshold, current_price=current_price)

            # Save current position info to csv
            save_position_history(symbol, entry_position, entry_price=current_price, take_profit=take_profit, stop_loss=stop_loss,
                                  entry_prediction=predicted_result, trailing_stop_threshold=trailing_threshold, callback_rate=trailing_return_threshold)
    else:
        print('No action')

    return entry_position

def set_position_tp(entry_prediction, model_type=None):
    take_profit = 0

    if entry_prediction in [0, 1]:
        take_profit = model_tp_0_1
    elif entry_prediction in [2, 3]:
        take_profit = model_tp_2_3
    elif entry_prediction in [4, 5]:
        take_profit = model_tp_4_5
    print(f'Take Profit set by model 1 as {take_profit * 100} %')

    return take_profit

def combine_v2_exit_conditions(model_type=None, data=None, entry_position=None, predicted_result_2=None, confidence_2=None):
    take_profit = set_v2_tp(predicted_result_2, confidence_2)

    # Return if not meeting v2 tp conditions
    if take_profit == 0:
        print('v2 TP conditions not met. No action will be taken.')
        return

    print(f'Take Profit set by model 2 as {take_profit * 100} %')
    trailing_threshold = round(take_profit / 1.5 * 100, 2)
    trailing_return_threshold = round(trailing_threshold / 2, 2)
    if trailing_return_threshold <= 0.4:
        trailing_return_threshold = 0.4

    current_price = data.iloc[-1]['Close']

    send_order(symbol, f'Set v2 TP - {entry_position}', model_type, tp_value=take_profit * 100, 
               trailing_activation=trailing_threshold, trailing_offset=trailing_return_threshold, current_price=current_price)

    # Update current position info to csv
    save_position_history(symbol, 'Set v2 TP', take_profit=take_profit, trailing_stop_threshold=trailing_threshold, callback_rate=trailing_return_threshold)

def set_v2_tp(v2_prediction, v2_confidence):
    take_profit = 0

    if v2_prediction == 1:
        if 0.39 <= v2_confidence <= 0.44:
            take_profit = 0.015 # 或者0.01
        elif 0.23 <= v2_confidence <= 0.26:
            take_profit = 0.03
    elif v2_prediction == 2: # 需調整
        if 0.28 <= v2_confidence <= 0.29:
            take_profit = 0.015 # 或者0.01
        elif 0.32 <= v2_confidence <= 0.33:
            take_profit = 0.02
        elif 0.2 <= v2_confidence <= 0.23:
            take_profit = 0.025
    elif v2_prediction == 3:
        if 0.256 <= v2_confidence <= 0.26:
            take_profit = 0.015 # 或者0.01
        elif 0.274 <= v2_confidence <= 0.278:
            take_profit = 0.02
        elif 0.262 <= v2_confidence <= 0.27:
            take_profit = 0.025
    elif v2_prediction == 4 and not(0.255 <= v2_confidence <= 0.26):
        if 0.245 <= v2_confidence <= 0.25:
            take_profit = 0.015
        elif 0.255 <= v2_confidence <= 0.265:
            take_profit = 0.01
        elif 0.315 <= v2_confidence <= 0.32:
            take_profit = 0.015
    elif v2_prediction == 5: # 需調整
        if 0.36 <= v2_confidence <= 0.37:
            take_profit = 0.015
        elif 0.32 <= v2_confidence <= 0.33:
            take_profit = 0.02
        elif 0.34 <= v2_confidence <= 0.35:
            take_profit = 0.025
    elif v2_prediction == 6:
        if 0.243 <= v2_confidence <= 0.265:
            take_profit = 0.015
        elif 0.251 <= v2_confidence <= 0.253:
            take_profit = 0.02
    elif v2_prediction == 7 and not(0.47 <= v2_confidence <= 0.475 or 0.39 <= v2_confidence <= 0.44): # 7有大回撤點
        if 0.46 <= v2_confidence <= 0.49:
            take_profit = 0.025 # 或0.03
        elif 0.37 <= v2_confidence <= 0.49:
            take_profit = 0.015 # 或0.01(0.01比較保險)
    elif v2_prediction == 8:
        if 0.49 <= v2_confidence <= 0.55:
            take_profit = 0.01
        elif 0.4 <= v2_confidence <= 0.43:
            take_profit = 0.015
        elif 0.45 <= v2_confidence <= 0.48:
            take_profit = 0.03 # 或 0.03
        elif 0.45 <= v2_confidence <= 0.53:
            take_profit = 0.025
    elif v2_prediction == 9:
        if 0.2 <= v2_confidence <= 0.27:
            take_profit = 0.02
    elif v2_prediction == 10:
        if 0.23 <= v2_confidence <= 0.245:
            take_profit = 0.01

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
    current_position_csv = os.path.join(os.path.abspath(os.path.dirname('__file__')), 'records/current_position.csv')

    # Read current market data and position info to determine exit conditions
    current_price = data.iloc[-1]['Close']
    current_high = data.iloc[-1]['High']
    current_low = data.iloc[-1]['Low']
    print(f'Current Price: {current_price}')

    entry_price = position_info.iloc[-1]['entry_price']
    max_profit = position_info.iloc[-1]['max_profit']
    stop_loss = position_info.iloc[-1]['stop_loss']
    take_profit = position_info.iloc[-1]['take_profit']
    loss_hold_count = position_info.iloc[-1]['loss_hold_count']
    entry_prediction = position_info.iloc[-1]['entry_prediction']
    trailing_threshold = position_info.iloc[-1]['trailing_stop_threshold']
    trailing_return_threshold = position_info.iloc[-1]['callback_rate']
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
            strategy_user = get_strategy_user(model=v1_model_name, token=server_token)
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

        if exit_condition is not None:
            print(f'Exit Condition Met: {exit_condition}')

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
            strategy_user = get_strategy_user(model=v1_model_name, token=server_token)
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

        if exit_condition is not None:
            print(f'Exit Condition Met: {exit_condition}')

def execute_exit_trade(symbol, action, take_profit=None, stop_loss=None, close_price=None, exit_condition=None, model_type=None, token=None):
    send_order(symbol, action, model_type)
    data = save_position_history(symbol, action, take_profit=take_profit, stop_loss=stop_loss, close_price=close_price, exit_condition=exit_condition)
    send_trading_record(data, model_type, token)


# 抓取最新資料並更新目前倉位
fetch_data_start = time.time()
data_1, calculated_data = fetch_data_hybrid(symbol, interval, v1_data_file_name, fetch_data_length, 'supertrend', return_calculation_data=True)
fetch_data_end = time.time()
print(f'Time elapsed during data fetch: {fetch_data_end - fetch_data_start}')
position, position_info = check_current_position()

# 只有在每剛經過 15 分時才執行交易相關動作
# 假如目前未持倉，則進行模型預測；已持倉的話，則判斷是否達到出場條件
if time.time() % 900 < 50:
    if position == 0:
        print('No position found. Start predicting...')
        predicted_result_1, confidence_1 = predict_result(v1_model_path, data_1, model_type=v1_model_name)
        print(f'Model 1 predicted result: {predicted_result_1} with confidence {confidence_1}')
        save_predicted_result(predicted_result_1, confidence_1, model_type=v1_model_name)

        entry_position = execute_trade(predicted_result_1, confidence_1, data_1, model_type=v1_model_name, stop_loss=model_sl)

        # Update model 2 data after trade execution
        print(calculated_data)
        data_2 = fetch_data_hybrid(symbol, interval, v2_data_file_name, fetch_data_length, v2_model_name, calculated_data)

        # If entry position is long or short, then predict with model 2 to set exit conditions
        if entry_position != '':
            predicted_result_2, confidence_2 = predict_result(v2_model_path, data_2, model_type=v2_model_name)
            print(f'Model 2 predicted result: {predicted_result_2} with confidence {confidence_2}')

            save_predicted_result(predicted_result_2, confidence_2, model_type=v2_model_name)
            combine_v2_exit_conditions(model_type=v1_model_name, data=data_2, entry_position=entry_position, predicted_result_2=predicted_result_2, confidence_2=confidence_2)
    else:
        close_position(data_1, position, position_info, model_type=v1_model_name)
else:
    print('Trade will only execute at 15, 30, 45, 00 in every hour')

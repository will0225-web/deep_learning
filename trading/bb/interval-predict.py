import os, sys
sys.path.append(os.getcwd())

import time
from dotenv import load_dotenv

load_dotenv()

from trading.utils.data import fetch_data
from trading.utils.trading import send_order, send_trading_record
from trading.utils.predict import predict_result
from trading.utils.record import save_predicted_result, save_position_history, check_current_position

# 定義 model 名稱與檔案位置
model_name = 'bb'

model_path = 'trained_models/1709987084_softmax_loss-0.7022_accuracy-0.7377_f1-0.3519_BB_strategy'
data_file_name = 'ETHUSDT_15m_realtime_caculated_084.csv'

fetch_data_length = 300000

# 交易標的與週期
symbol = 'ETHUSDT'
interval = '15m'

# 策略參數設定
# 模型預測結果之條件參數
predicted_long_results = [1, 3]
predicted_short_results = [2, 4]

entry_confidence = 0.4
entry_confidence_3_4 = 0.7

# 模型交易策略參數設定
model_sl = 0.008

rsi_exit_delta = 15
hold_count_limit = 36
stable_hold_count_limit = 10
stable_price_percentage = 0.006

# Determine trade actions
def execute_trade(predicted_result, confidence, data, model_type=None, stop_loss=None, take_profit=None, trailing_threshold=None, trailing_return_threshold=None):
    entry_position = ''

    if predicted_result in predicted_long_results or predicted_result in predicted_short_results:
        # Check whether entry conditions are met
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
            send_order(symbol, entry_position, model_type, sl_value=stop_loss * 100, sl_price=sl_price, current_price=current_price)

            # Save current position info to csv
            save_position_history(symbol, entry_position, entry_price=current_price, take_profit=take_profit, stop_loss=stop_loss,
                                  rsi_peak=rsi_peak, entry_prediction=predicted_result)
    else:
        print('No action')

def check_entry_conditions(predicted_result, confidence, volume=None, supertrend=None, kline_color=None):
    if predicted_result in [1, 2] and confidence >= entry_confidence:
        return True
    elif predicted_result in [3, 4] and confidence >= entry_confidence_3_4:
        return True
    else:
        return False

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
    rsi_peak = position_info.iloc[-1]['rsi_peak']
    max_profit = position_info.iloc[-1]['max_profit']
    stop_loss = position_info.iloc[-1]['stop_loss']

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

        # calculate rsi peak
        if current_rsi > rsi_peak:
            rsi_peak = max(rsi_peak, current_rsi)
            position_info.loc[0, 'rsi_peak'] = rsi_peak
            position_info.to_csv(current_position_csv, index=False)
            print(f'RSI Peak Rised: {rsi_peak}')
        else:
            print(f'RSI Peak: {rsi_peak}')                  

        if current_low <= ((1 - stop_loss) * entry_price):
            exit_condition = 'Stop Loss Long'
            execute_exit_trade(symbol, 'Exit Long', stop_loss=stop_loss, close_price=current_low, exit_condition=exit_condition, model_type=model_type)
        elif hold_count >= hold_count_limit:
            exit_condition = 'Long Hold Count Limit Reached'
            execute_exit_trade(symbol, 'Exit Long', stop_loss=stop_loss, close_price=current_price, exit_condition=exit_condition, model_type=model_type)
        elif rsi_peak - rsi_exit_delta >= current_rsi:
            exit_condition = 'Long RSI Weaken'
            execute_exit_trade(symbol, 'Exit Long', stop_loss=stop_loss, close_price=current_price, exit_condition=exit_condition, model_type=model_type)
        elif hold_count >= stable_hold_count_limit and max_profit <= stable_price_percentage:
            exit_condition = 'Long Stable Hold Count Limit Reached'
            execute_exit_trade(symbol, 'Exit Long', stop_loss=stop_loss, close_price=current_price, exit_condition=exit_condition, model_type=model_type)

        print(f'Exit Condition Meet: {exit_condition}')

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

        # calculate rsi peak
        if current_rsi < rsi_peak:
            rsi_peak = min(rsi_peak, current_rsi)
            position_info.loc[0, 'rsi_peak'] = rsi_peak
            position_info.to_csv(current_position_csv, index=False)
            print(f'RSI Peak Dropped: {rsi_peak}')
        else:
            print(f'RSI Peak: {rsi_peak}')

        if current_high >= ((1 + stop_loss) * entry_price):
            exit_condition = 'Stop Loss Short'
            execute_exit_trade(symbol, 'Exit Short', stop_loss=stop_loss, close_price=current_high, exit_condition=exit_condition, model_type=model_type)
        elif hold_count >= hold_count_limit:
            exit_condition = 'Short Hold Count Limit Reached'
            execute_exit_trade(symbol, 'Exit Short', stop_loss=stop_loss, close_price=current_price, exit_condition=exit_condition, model_type=model_type)
        elif rsi_peak + rsi_exit_delta <= current_rsi:
            exit_condition = 'Short RSI Weaken'
            execute_exit_trade(symbol, 'Exit Short', stop_loss=stop_loss, close_price=current_price, exit_condition=exit_condition, model_type=model_type)
        elif hold_count >= stable_hold_count_limit and max_profit <= stable_price_percentage:
            exit_condition = 'Short Stable Hold Count Limit Reached'
            execute_exit_trade(symbol, 'Exit Short', stop_loss=stop_loss, close_price=current_price, exit_condition=exit_condition, model_type=model_type)

        if exit_condition is not None:
            print(f'Exit Condition Met: {exit_condition}')

def execute_exit_trade(symbol, action, take_profit=None, stop_loss=None, close_price=None, exit_condition=None, model_type=None):
    send_order(symbol, action, model_type)
    data = save_position_history(symbol, action, take_profit=take_profit, stop_loss=stop_loss, close_price=close_price, exit_condition=exit_condition)
    send_trading_record(data, model_type)


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
        execute_trade(predicted_result, confidence, data, model_type=model_name, stop_loss=model_sl)
    else:
        close_position(data, position, position_info, model_type=model_name)
else:
    print('Trade will only execute at 15, 30, 45, 00 in every hour')

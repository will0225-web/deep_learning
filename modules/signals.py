import sys
import os
import pandas as pd

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from modules import indicators as indicators
from modules import utilities as utilities



# MACD Cross
macd_long_hist = 0
macd_short_hist = 0
def check_macd_hist_cross(data):
    macd_hist_cross = []

    for i in range(len(data)):
        if i == 0:
            macd_hist_cross.append(0)
        elif data[i - 1] < 0 and data[i] > macd_long_hist:
            macd_hist_cross.append(2)
        elif data[i - 1] > 0 and data[i] < -macd_short_hist:
            macd_hist_cross.append(1)
        else:
            macd_hist_cross.append(0)
    
    return macd_hist_cross

def is_macd_short_long(macd_hist_series):
    conditions = [
        (macd_hist_series == 0),  # neutral
        (macd_hist_series < 0),  # short
        (macd_hist_series > 0),  # long
    ]
    
    choices = [0, 1, 2]
    return np.select(conditions, choices, default=0)

def is_up_down_SMA(close_series, sma_series):
    return np.where(close_series < sma_series, 0, 1)

def is_green_red_kline(close_series, open_series):
    # 0是紅色, 1是綠色
    return np.where(close_series < open_series, 0, 1)

def is_overbought_oversold(rsi_series):
    overbought_range = 74
    oversold_range = 26
    conditions = [
        (rsi_series <= overbought_range) & (rsi_series >= oversold_range),  # neutral
        (rsi_series > overbought_range),  # overbought
        (rsi_series < oversold_range),    # oversold
    ]
    
    choices = [0, 1, 2]
    return np.select(conditions, choices, default=0)

# MACD Consecutive Reverse
reverse_count = 3
def check_macd_consecutive_reverse(data):
    macd_hist_reverse = []

    for i in range(len(data)):
        if i < 2:
            macd_hist_reverse.append(0)
        else:
            reverse_times = reverse_count * 2 - 1
            past_3_bars = data[i - 2: i + 1]
            past_3_bars = past_3_bars.to_list()

            if data[i] > 0 and past_3_bars.index(min(past_3_bars)) == 1:
                uprising = True
                for j in range(2, len(data)):
                    examine_bar_index = i - j
                    if examine_bar_index - 1 < 0 or data[examine_bar_index - 1] < 0:
                        macd_hist_reverse.append(0)
                        break                      

                    if data[examine_bar_index] > data[examine_bar_index - 1] and uprising == True:
                        uprising = False
                        reverse_times -= 1
                    elif data[examine_bar_index] < data[examine_bar_index - 1] and uprising == False:
                        uprising = True
                        reverse_times -= 1
                    
                    if reverse_times == 0:
                        macd_hist_reverse.append(1)
                        break
            elif data[i] < 0 and past_3_bars.index(max(past_3_bars)) == 1:
                downfalling = True
                for j in range(2, len(data)):
                    examine_bar_index = i - j
                    if examine_bar_index - 1 < 0 or data[examine_bar_index - 1] > 0:
                        macd_hist_reverse.append(0)
                        break

                    if data[examine_bar_index] < data[examine_bar_index - 1] and downfalling == True:
                        downfalling == False
                        reverse_times -= 1
                    elif data[examine_bar_index] > data[examine_bar_index - 1] and downfalling == False:
                        downfalling == True
                        reverse_times -= 1
                    
                    if reverse_times == 0:
                        macd_hist_reverse.append(-1)
                        break
            else:
                macd_hist_reverse.append(0)

    return macd_hist_reverse

def set_target(df, threshold, at_least_profit, hold_count_limit):
    # 0為橫盤
    # 1為碰上
    # 2為碰下

    close_price_series = df['Close']
    low_series = df['Low']
    high_series = df['High']
    targets = []

    for idx in range(len(close_price_series)):
        current_price = close_price_series[idx]
        upper_bound = current_price * (1 + threshold)
        lower_bound = current_price * (1 - threshold)
        
        target = 0
        check_end_index = min(len(close_price_series), idx + hold_count_limit + 1)
        for future_idx in range(idx+1, check_end_index):
            future_price = close_price_series[future_idx]
            high_price = high_series[future_idx]
            low_price = low_series[future_idx]
            
            current_profit = indicators.calculate_percentage_change(future_price, current_price)

            if high_price >= upper_bound:
                target = 1
                break
            elif low_price <= lower_bound:
                target = 2
                break
                
            if (future_idx == check_end_index - 1) or future_idx + hold_count_limit == idx:
                if current_profit > at_least_profit:
                    target = 1
                elif -current_profit > at_least_profit:
                    target = 2
                else:
                    target = 0
                break
        
        targets.append(target)
        
    return targets

def set_win_loss_target(df, win_threshold, lose_threshold, at_least_profit, hold_count_limit):
    win_threshold = win_threshold
    lose_threshold = lose_threshold
    
    actions = []
    price_series = df['Close']
    low_series = df['Low']
    high_series = df['High']
    for idx in range(len(price_series)):
        current_price = price_series[idx]
        upper_bound_win = current_price * (1 + win_threshold)
        lower_bound_win = current_price * (1 - win_threshold)
        upper_bound_lose = current_price * (1 + lose_threshold)
        lower_bound_lose = current_price * (1 - lose_threshold)
        
        touched_down_or_up_first = 0
        touched_down = False
        touched_up = False

        action = 0  # 預設為無效
        check_end_index = min(len(price_series), idx + hold_count_limit + 1)

        for future_idx in range(idx + 1, check_end_index):
            future_price = price_series[future_idx]
            future_high_price = high_series[future_idx]
            future_low_price = low_series[future_idx]

            current_profit = indicators.calculate_percentage_change(current_price, future_price)

            # 检查止损条件
            if future_low_price <= lower_bound_lose:
                # 多輸
                touched_down = True
                if touched_down_or_up_first == 0:
                    touched_down_or_up_first = 1

            if future_high_price >= upper_bound_lose:
                # 空輸
                touched_up = True
                if touched_down_or_up_first == 0:
                    touched_down_or_up_first = 2

            if touched_down and touched_up:
                # 3 多輸
                # 4 空輸
                action = 4 if touched_down_or_up_first == 1 else 3
                break

            # 判斷贏的情況
            if future_high_price >= upper_bound_win and not(touched_down):
                action = 1  # 多贏
                break

            if future_low_price <= lower_bound_win and not(touched_up):
                action = 2  # 空贏
                break
            

            if (future_idx == check_end_index - 1):
                if touched_up:
                    action = 3
                elif touched_down:
                    action = 4
                elif current_profit >= at_least_profit:
                    action = 1
                elif -current_profit >= at_least_profit:
                    action = 2
                break
        
        actions.append(action)
        
    return actions

# TODO：加入 hold_count_limit
def set_target_macd_cross(df, long_exit_threshold, short_exit_threshold, lose_threshold):
    close_series, low_series, high_series, open_series = df['Close'], df['Low'], df['High'], df['Open']
    rsi_series = df['RSI']
    macd_cross_series = df['MACD_Cross']
    results = []

    for idx in range(len(close_series)):
        # 未發生 MACD 交叉，以及未達到出場與停損條件時，result 為 -1
        result = -1
        macd_signal = macd_cross_series[idx]  # 1 為 Hist 轉負，2 為 Hist 轉正

        # 當有發生 MACD 交叉時
        if macd_signal in [1, 2]:
            entry_price = close_series[idx]

            for future_idx in range(idx + 1, len(close_series)):
                future_rsi = rsi_series[future_idx]
                future_price = close_series[future_idx]
                future_low_price = low_series[future_idx]
                future_high_price = high_series[future_idx]
                previous_rsi = rsi_series[future_idx - 1]

                # 做多訊號且觸發多單出場條件
                if macd_signal == 2 and future_rsi < long_exit_threshold and previous_rsi > long_exit_threshold:
                    if future_price > entry_price:
                        result = 1
                        break
                    elif future_price < entry_price:
                        result = 0
                        break       
                # 做多訊號且觸發停損條件
                elif macd_signal == 2 and future_low_price < entry_price * (1 - lose_threshold / 100):
                    result = 0
                    break
                # 做空訊號且觸發空單出場條件
                elif macd_signal == 1 and future_rsi > short_exit_threshold and previous_rsi < short_exit_threshold:
                    if future_price < entry_price:
                        result = 1
                        break
                    elif future_price > entry_price:
                        result = 0
                        break
                # 做空訊號且觸發停損條件
                elif macd_signal == 1 and future_high_price > entry_price * (1 + lose_threshold / 100):
                    result = 0
                    break

        results.append(result)

    return results

def is_green_red_kline(close_series, open_series):
    # 0是紅色, 1是綠色
    return np.where(close_series < open_series, 0, 1)

def is_overbought_oversold(rsi_series):
    overbought_range = 74
    oversold_range = 26
    conditions = [
        (rsi_series <= overbought_range) & (rsi_series >= oversold_range)  # neutral
        (rsi_series > overbought_range),  # overbought
        (rsi_series < oversold_range),    # oversold
    ]
    
    choices = [0, 1, 2]
    return np.select(conditions, choices, default=0)

# 是否轉強
def is_macd_stronger(hist_series):
    targets = [0] * len(hist_series) # 初始化目標列表為0

    for i in range(1, len(hist_series)):
        if hist_series[i] < 0 and hist_series[i] < hist_series[i-1]:
            targets[i] = 1
        elif hist_series[i] > 0 and hist_series[i] > hist_series[i-1]:
            targets[i] = 2
            
    return targets

def set_rsi_target(price_series, rsi_series, low_series, high_series, loss_percent_long = 0.015, loss_percent_short = 0.015, at_least_profit = 0.5, hold_count_limit = 48):
    # 0 = 沒有作為

    # 1 = short rsi出場
    # 2 = long rsi出場
    
    # 3 = short loss_percent_short%
    # 4 = long loss_percent_long%
    
    buy_threshold = 26
    sell_threshold = 65

    actions = []

    for idx in range(len(price_series) - 1):
        current_price = price_series[idx]
        touched_down = False
        touched_up = False
        touched_down_or_up_first = 0 # 0是還沒碰到, 1是touched down first, 2是touched up first
        action = 0  # 預設為0: 未達到目標

        check_end_index = min(len(price_series), idx + hold_count_limit + 1)
        for future_idx in range(idx + 1, check_end_index):
            future_price = price_series[future_idx]
            future_low = low_series[future_idx]
            future_high = high_series[future_idx]

            # 判斷是否先碰到下loss_percent
            if future_low < current_price * (1 - loss_percent_long):
                if touched_down_or_up_first == 0:
                    touched_down_or_up_first = 1
                touched_down = True

            # 判斷是否先碰到上loss_percent
            elif future_high > current_price * (1 + loss_percent_short):
                if touched_down_or_up_first == 0:
                    touched_down_or_up_first = 2
                touched_up = True

            # 當先碰到下loss_percent後再碰到上loss_percent 或 當先碰到上loss_percent後再碰到下loss_percent
            if touched_down and touched_up:
                if touched_down_or_up_first == 1:
                    action = 3
                elif touched_down_or_up_first == 2:
                    action = 4
                break

            current_percentage = indicators.calculate_percentage_change(current_price, future_price)
            # 判斷RSI下穿sell_threshold，且touched_down為False
            if not touched_down and rsi_series[future_idx] < sell_threshold and rsi_series[future_idx - 1] >= sell_threshold:
                if future_price > current_price and current_percentage > at_least_profit:
                    action = 2
                    break
                else:
                    action = 0
                    break

            # 判斷RSI上穿buy_threshold，且touched_up為False
            if not touched_up and rsi_series[future_idx] > buy_threshold and rsi_series[future_idx - 1] <= buy_threshold:
                if future_price < current_price and -current_percentage > at_least_profit:
                    action = 1
                    break
                else:
                    action = 0
                    break

            if future_idx == check_end_index - 1:
                if touched_down_or_up_first == 1:
                    action = 3
                elif touched_down_or_up_first == 2:
                    action = 4
                break


        actions.append(action)

    actions.append(0)  # 最後一個數據沒有未來的數據可以比較，所以直接設定為0
    return actions


def set_macd_thesame_trend_target(df, loss_percent_long=0.015, loss_percent_short=0.015, at_least_profit=0.5):
    # -1非macd交叉時的數據

    # 0為long直接停損
    # 1為short直接停損
    # 2為long有at_leatst_profit 但停損
    # 3為short有at_leatst_profit 但停損
    # 4為long有at_leatst_profit 贏
    # 5為short有at_leatst_profit 贏
    # 6為long macd轉換出場贏
    # 7為short macd轉換出場贏
    ### hold_count X
    # 8為long hold到底 贏
    # 9為short hold到底 贏

    price_series = df['Close']
    rsi_series = df['RSI']
    low_series = df['Low']
    high_series = df['High']
    macd_hist_series = df['Hist']

    # 初始化目标序列
    target = np.full_like(price_series, -1, dtype=np.int64)

    macd_zero_line = 0
    rsi_exit_delta = 12  # RSI退出阈值的偏移量


    for i in range(len(price_series)):
        long_position = False
        short_position = False
        entry_price = None

        max_rsi = float('-inf')  # 初始化为负无穷大
        min_rsi = float('inf')   # 初始化为正无穷大
        max_profit = 0
        

        # 确定当前点是黄金交叉还是死亡交叉
        if i > 0 and macd_hist_series[i-1] < macd_zero_line and macd_hist_series[i] >= macd_zero_line :
            long_position = True
            entry_price = price_series[i]
            max_rsi = rsi_series[i]
            long_stop_loss_price = entry_price * (1 - loss_percent_long)

        elif i > 0 and macd_hist_series[i-1] > macd_zero_line and macd_hist_series[i] <= macd_zero_line:
            short_position = True
            entry_price = price_series[i]
            min_rsi = rsi_series[i]
            short_stop_loss_price = entry_price * (1 + loss_percent_short)

        if long_position or short_position:
            # 往后查看直到条件满足或MACD状态改变
            for j in range(i + 1, len(price_series)):
                current_profit = (price_series[j] - entry_price) / entry_price if long_position else (entry_price - price_series[j]) / entry_price
                
                max_profit = max(max_profit, current_profit)

                # 更新最大或最小RSI
                if long_position:
                    max_rsi = max(max_rsi, rsi_series[j])
                elif short_position:
                    min_rsi = min(min_rsi, rsi_series[j])

                # 检查止损
                if long_position and low_series[j] <= long_stop_loss_price:
                    if max_profit < at_least_profit:
                        target[i] = 0
                    else:
                        target[i] = 2
                    break
                elif short_position and high_series[j] >= short_stop_loss_price:
                    if max_profit < at_least_profit:
                        target[i] = 1
                    else:
                        target[i] = 3
                    break

                # 检查RSI退出条件和利润
                if long_position and rsi_series[j] <= max_rsi - rsi_exit_delta:
                    if current_profit >= (at_least_profit * 2):
                        target[i] = 4
                        break
                elif short_position and rsi_series[j] >= min_rsi + rsi_exit_delta:
                    if current_profit >= (at_least_profit * 2):
                        target[i] = 5
                        break
                    
                # 如果MACD状态改变，则检查是否达到最少利润，并停止往后查看
                if long_position and macd_hist_series[j-1] > macd_zero_line and macd_hist_series[j] <= macd_zero_line:
                    if current_profit >= at_least_profit:
                        target[i] = 6
                        break

                if short_position and macd_hist_series[j-1] < macd_zero_line and macd_hist_series[j] >= macd_zero_line:
                    if current_profit >= at_least_profit:
                        target[i] = 7
                        break
                    

                if i + 60 == j:
                    if current_profit >= 0:
                        if long_position:
                            target[i] = 8
                        elif short_position:
                            target[i] = 9
                    else:
                        if long_position:
                            if max_profit < at_least_profit:
                                target[i] = 0
                            else:
                                target[i] = 2
                        elif short_position:
                            if max_profit < at_least_profit:
                                target[i] = 1
                            else:
                                target[i] = 3
                    break
    return target


def set_macd_entry_trend_target(df, loss_percent_long=0.015, loss_percent_short=0.015, at_least_profit=0.5):
    # -1非macd交叉時的數據

    ### 出場定義
    # RSI出場
    # MACD轉向
    # ATR出場
    # BB出場
    # 價格反轉?幅度
    
    # 看誰的profit大就用誰的當target

    # 0為恰似橫盤, 最好的出場獲利不到最低接受的獲利
    # 1為long停損
    # 2為short停損
    # 3贏

    price_series = df['Close']
    rsi_series = df['RSI']
    low_series = df['Low']
    high_series = df['High']
    macd_hist_series = df['Hist']
    upper_band = df['Upper Band']
    lower_band = df['Lower Band']
    middle_band = df['Middle Band']

    # 初始化目标序列
    target = np.full_like(price_series, -1, dtype=np.int64)

    macd_zero_line = 0
    rsi_exit_delta = 12  # RSI退出阈值的偏移量


    for i in range(len(price_series)):
        long_position = False
        short_position = False
        entry_price = None

        max_rsi = float('-inf')  # 初始化为负无穷大
        min_rsi = float('inf')   # 初始化为正无穷大

        best_exit_point_profit = 0
        best_exit_point_target = -1
        action = -1

        is_rsi_exit = False
        is_bb_exit = False
        is_macd_cross = False
        is_stop_loss = False

        # 确定当前点是黄金交叉还是死亡交叉
        if i > 0 and macd_hist_series[i-1] < macd_zero_line and macd_hist_series[i] >= macd_zero_line :
            long_position = True
            entry_price = price_series[i]
            max_rsi = rsi_series[i]
            long_stop_loss_price = entry_price * (1 - loss_percent_long)

        elif i > 0 and macd_hist_series[i-1] > macd_zero_line and macd_hist_series[i] <= macd_zero_line:
            short_position = True
            entry_price = price_series[i]
            min_rsi = rsi_series[i]
            short_stop_loss_price = entry_price * (1 + loss_percent_short)

        if long_position or short_position:
            # 往后查看直到条件满足或MACD状态改变
            for j in range(i + 1, len(price_series)):
                feature_close_price = price_series[j]
                feature_upper_band = upper_band[j]
                feature_lower_band = lower_band[j]
                feature_middle_band = middle_band[j]
                current_profit = (feature_close_price - entry_price) / entry_price if long_position else (entry_price - feature_close_price) / entry_price
                
                # 更新最大或最小RSI
                if long_position:
                    max_rsi = max(max_rsi, rsi_series[j])
                elif short_position:
                    min_rsi = min(min_rsi, rsi_series[j])

                # 检查止损
                if long_position and low_series[j] <= long_stop_loss_price and not(is_stop_loss):
                    is_stop_loss = True
                    action = 1
                elif short_position and high_series[j] >= short_stop_loss_price and not(is_stop_loss):
                    is_stop_loss = True
                    action = 2

                # 如果MACD状态改变，则检查是否达到最少利润，并停止往后查看
                if long_position and macd_hist_series[j-1] > macd_zero_line and macd_hist_series[j] <= macd_zero_line and not(is_macd_cross):
                    is_macd_cross = True
                    if current_profit >= 0:
                        if current_profit >= best_exit_point_profit:
                            best_exit_point_profit = current_profit
                            best_exit_point_target = 3
                    else:
                        action = 1
                        
                if short_position and macd_hist_series[j-1] < macd_zero_line and macd_hist_series[j] >= macd_zero_line and not(is_macd_cross):
                    is_macd_cross = True
                    if current_profit >= 0:
                        if current_profit >= best_exit_point_profit:
                            best_exit_point_profit = current_profit
                            best_exit_point_target = 3
                    else:
                        action = 2

                # 检查RSI退出条件和利润
                if long_position and rsi_series[j] <= max_rsi - rsi_exit_delta and not(is_rsi_exit):
                    if current_profit >= best_exit_point_profit:
                        best_exit_point_profit = current_profit
                        best_exit_point_target = 3
                        is_rsi_exit = True
                        
                elif short_position and rsi_series[j] >= min_rsi + rsi_exit_delta and not(is_rsi_exit):
                    if current_profit >= best_exit_point_profit:
                        best_exit_point_profit = current_profit
                        best_exit_point_target = 3
                        is_rsi_exit = True

                # BB
                if long_position and (feature_close_price >= feature_upper_band or feature_close_price <= feature_middle_band) and not(is_bb_exit):
                    if current_profit >= best_exit_point_profit:
                        best_exit_point_profit = current_profit
                        best_exit_point_target = 3
                        is_bb_exit = True

                if short_position and (feature_close_price <= feature_lower_band or feature_close_price >= feature_middle_band) and not(is_bb_exit):
                    if current_profit >= best_exit_point_profit:
                        best_exit_point_profit = current_profit
                        best_exit_point_target = 3
                        is_bb_exit = True

                # ATR

                if is_macd_cross or is_stop_loss:
                    # 結算誰比較好
                    if best_exit_point_profit >= at_least_profit:
                        action = best_exit_point_target
                    else:
                        if action == -1:
                            action = 0
                    break

                
            target[i] = action

    return target

@utilities.capture_args
def set_rsi_macd_target(df, loss_percent_long=0.015, loss_percent_short=0.015, at_least_profit=0.005, horizontal_price_percentage=0.01, horizontal_price_hold_count_limit=20, hold_count_limit=60):
    # horizontal_price_hold_count_limit是來判定這個步數內, 有沒有一定的漲幅
    # 假設horizontal_price_hold_count_limit有漲幅, 就會進入hold_count_limit判定, 這期間的target會是什麼
    # 先測試出場機制用rsi_exit_delta的方式
    # 另外還會再測試TradingView的出場方式

    # 0就是在hold_count_limit到達時, 沒有超過一定的漲幅
    # 1就是rsi long收場有賺超過at_least_profit
    # 2就是rsi short收場有賺超過at_least_profit
    # 3就是下long 1.5% or 偏向於long
    # 4就是下short 1.5% or 偏向於short



    price_series = df['Close']
    rsi_series = df['RSI']
    low_series = df['Low']
    high_series = df['High']
    rsi_exit_delta = 12  # RSI退出阈值的偏移量


    actions = []

    for idx in range(len(price_series) - 1):
        current_price = price_series[idx]
        touched_down = False
        touched_up = False

        # touched_down_or_up_first = 0 # 0是還沒碰到, 1是touched down first, 2是touched up first
        action = 0  # 預設為0: 未達到目標, 意指橫盤
        has_result = False
        first_touch_upper = None
        first_touch_lower = None

        max_price_in_period = max(price_series[idx:min(idx + horizontal_price_hold_count_limit, len(price_series))])
        min_price_in_period = min(price_series[idx:min(idx + horizontal_price_hold_count_limit, len(price_series))])
        # 检查是否有horizontal_price_percentage%以上的涨跌幅
        if not (max_price_in_period >= current_price * (1 + horizontal_price_percentage) or (min_price_in_period <= current_price * (1 - horizontal_price_percentage))):
            action = 0
        else:
            if (max_price_in_period >= current_price * (1 + horizontal_price_percentage)) and (min_price_in_period <= current_price * (1 - horizontal_price_percentage)):
                # 期間內上下各有horizontal_price_percentage%, 就判定有沒有碰到停損點, 同一根碰到停損點就當作橫盤.
                # 沒有停損就判定先碰上還是碰下, 再決定要不要判斷long或short
                upper_limit = current_price * (1 + horizontal_price_percentage)
                lower_limit = current_price * (1 - horizontal_price_percentage)

                check_end_index = min(len(price_series), idx + hold_count_limit + 1)
                for future_idx in range(idx + 1, check_end_index):
                    future_price = price_series[future_idx]
                    future_low = low_series[future_idx]
                    future_high = high_series[future_idx]
                    future_rsi = rsi_series[future_idx]

                    # 同一根止損當橫盤
                    if future_low <= current_price * (1 - loss_percent_long) and future_high >= current_price * (1 + loss_percent_short):
                        action = 0
                        has_result = True
                        break
                    
                    # 看先碰上close price還是下
                    if future_price >= upper_limit and first_touch_upper is None:
                        first_touch_upper = True
                        break
                    elif future_price <= lower_limit and first_touch_lower is None:
                        first_touch_lower = True
                        break
                    elif future_idx == check_end_index - 1:
                        current_percentage = indicators.calculate_percentage_change(future_price, current_price)
                        if current_percentage >= at_least_profit:
                            action = 1
                            has_result = True
                            break
                        elif -current_percentage >= at_least_profit:
                            action = 2
                            has_result = True
                            break
                    
            
            if ((max_price_in_period >= current_price * (1 + horizontal_price_percentage) and first_touch_lower == None and first_touch_upper == None) or first_touch_upper) and not(has_result):
                # 這個就是下Long, 判定是否有達標
                check_end_index = min(len(price_series), idx + hold_count_limit + 1)
                max_rsi = rsi_series[idx]
                for future_idx in range(idx + 1, check_end_index):
                    future_price = price_series[future_idx]
                    future_low = low_series[future_idx]
                    future_high = high_series[future_idx]
                    future_rsi = rsi_series[future_idx]
                    max_rsi = max(max_rsi, future_rsi)
                    current_percentage = indicators.calculate_percentage_change(future_price, current_price)

                    if future_high >= current_price * (1 + loss_percent_short):
                        touched_up = True

                    if future_low <= current_price * (1 - loss_percent_long):
                        if touched_up:
                            action = 3
                        else:
                            action = 4
                        break
                        
                    if future_rsi < max_rsi - rsi_exit_delta or future_idx == check_end_index - 1:
                        if current_percentage >= at_least_profit:
                            action = 1
                        else:
                            if touched_up:
                                action = 3
                            else:
                                action = 0
                        break

            elif ((min_price_in_period <= current_price * (1 - horizontal_price_percentage) and first_touch_lower == None and first_touch_upper == None) or first_touch_lower) and not(has_result):
                # 這個就是下Short, 判定是否有達標
                check_end_index = min(len(price_series), idx + hold_count_limit + 1)
                min_rsi = rsi_series[idx]
                for future_idx in range(idx + 1, check_end_index):
                    future_price = price_series[future_idx]
                    future_low = low_series[future_idx]
                    future_high = high_series[future_idx]
                    future_rsi = rsi_series[future_idx]
                    min_rsi = min(min_rsi, future_rsi)
                    current_percentage = -indicators.calculate_percentage_change(future_price, current_price)

                    if future_low <= current_price * (1 - loss_percent_long):
                        touched_down = True

                    if future_high >= current_price * (1 + loss_percent_short):
                        if touched_down:
                            action = 4
                        else:
                            action = 3
                        break
                    
                    if future_rsi > min_rsi + rsi_exit_delta or future_idx == check_end_index - 1:
                        if current_percentage >= at_least_profit:
                            action = 2
                        else:
                            if touched_down:
                                action = 4
                            else:
                                action = 0
                        break
        actions.append(action)

    actions.append(0)  # 最後一個數據沒有未來的數據可以比較，所以直接設定為0
    return actions

def set_future_long_short_target(df, at_least_profit=0.005, hold_count_limit=8):
    # 看未來hold_count_limit根數是偏向漲還是跌
    # 一定要大於at_least_profit才會算是漲或者跌, 否則一律橫盤

    price_series = df['Close']
    actions = []

    for idx in range(len(price_series) - 1):
        current_price = price_series[idx]
        
        check_end_index = min(len(price_series), idx + hold_count_limit + 1)
        action = 0
        for future_idx in range(idx + 1, check_end_index):
            future_price = price_series[future_idx]
            current_percentage = indicators.calculate_percentage_change(future_price, current_price)
            
            if current_percentage > at_least_profit:
                action = 1
            elif -current_percentage > at_least_profit:
                action = 2
                
        actions.append(action)

    actions.append(0)  # 最後一個數據沒有未來的數據可以比較，所以直接設定為0
    return actions

@utilities.capture_args
def set_bb_specific_profit_strategy(df, loss_percent_long=0.015, loss_percent_short=0.015, profit=0.005, hold_count_at_least_profit=0.015, hold_count_limit=8):
    price_series = df['Close']
    low_series = df['Low']
    high_series = df['High']

    upper_band_series = df['Upper Band']
    lower_band_series = df['Lower Band']

    actions = []

    for idx in range(len(price_series) - 1):
        current_price = price_series[idx]
        upper_band = upper_band_series[idx]
        lower_band = lower_band_series[idx]

        action = 0
        position = 0 #1為long, 2為short
        max_profit = 0

        if current_price > upper_band:
            position = 1
        if current_price < lower_band:
            position = 2

        if position != 0:
            check_end_index = min(len(price_series), idx + hold_count_limit + 1)
            for future_idx in range(idx + 1, check_end_index):
                future_price = price_series[future_idx]
                future_low = low_series[future_idx]
                future_high = high_series[future_idx]
            
                # 计算价格和成交量的变化百分比
                price_change_pct = indicators.calculate_percentage_change(future_price, current_price)
                high_change_pct = indicators.calculate_percentage_change(future_high, current_price)
                low_change_pct = indicators.calculate_percentage_change(future_low, current_price)
                if position == 1:
                    # 假設回彈有低於剛剛的max_profit的一半, 就出場
                    max_profit = max(max_profit, high_change_pct)

                    if -low_change_pct >= loss_percent_long:
                        # long止損
                        action = 3
                        break
                    elif high_change_pct >= profit:
                        action = 1
                        break
                    elif (future_idx == check_end_index - 1):
                        if price_change_pct >= hold_count_at_least_profit:
                            action = 1
                        else:
                            action = 3
                        break 

                elif position == 2:
                    
                    max_profit = max(max_profit, -low_change_pct)

                    if high_change_pct >= loss_percent_short:
                        action = 4
                        break
                    elif -low_change_pct >= profit:
                        action = 2
                        break
                    elif (future_idx == check_end_index - 1):
                        if -price_change_pct >= hold_count_at_least_profit:
                            action = 2
                        else:
                            action = 4
                        break

        actions.append(action)

    actions.append(0)  # 最後一個數據沒有未來的數據可以比較，所以直接設定為0
    return actions

@utilities.capture_args
def set_bb_trailing_strategy(df, loss_percent_long=0.015, loss_percent_short=0.015, profit=0.005, hold_count_at_least_profit=0.015, hold_count_limit=8):
    price_series = df['Close']
    low_series = df['Low']
    high_series = df['High']

    upper_band_series = df['Upper Band']
    lower_band_series = df['Lower Band']

    actions = []

    for idx in range(len(price_series) - 1):
        current_price = price_series[idx]
        upper_band = upper_band_series[idx]
        lower_band = lower_band_series[idx]

        action = 0
        position = 0 #1為long, 2為short
        max_profit = 0

        if current_price > upper_band:
            position = 1
        if current_price < lower_band:
            position = 2

        if position != 0:
            check_end_index = min(len(price_series), idx + hold_count_limit + 1)
            for future_idx in range(idx + 1, check_end_index):
                future_price = price_series[future_idx]
                future_low = low_series[future_idx]
                future_high = high_series[future_idx]
            
                # 计算价格和成交量的变化百分比
                price_change_pct = indicators.calculate_percentage_change(future_price, current_price)
                high_change_pct = indicators.calculate_percentage_change(future_high, current_price)
                low_change_pct = indicators.calculate_percentage_change(future_low, current_price)
                if position == 1:
                    # 假設回彈有低於剛剛的max_profit的一半, 就出場
                    max_profit = max(max_profit, high_change_pct)

                    if -low_change_pct >= loss_percent_long:
                        # long止損
                        action = 3
                        break
                    elif max_profit >= 0.01:
                        action = 1
                        break
                    elif future_price < upper_band:
                        if price_change_pct >= 0 or max_profit >= 0.005:
                            action = 1
                        else:
                            action = 5
                        break
                    elif (future_idx == check_end_index - 1):
                        if price_change_pct >= hold_count_at_least_profit:
                            action = 1
                        elif price_change_pct > 0:
                            action = 7
                        else:
                            action = 5
                        break 

                elif position == 2:
                    
                    max_profit = max(max_profit, -low_change_pct)

                    if high_change_pct >= loss_percent_short:
                        action = 4
                        break
                    elif max_profit >= 0.01:
                        action = 2
                        break
                    elif future_price > lower_band:
                        if -price_change_pct >= 0 or max_profit >= 0.005:
                            action = 2
                        else:
                            action = 6
                        break
                    elif (future_idx == check_end_index - 1):
                        if -price_change_pct >= hold_count_at_least_profit:
                            action = 2
                        elif -price_change_pct > 0:
                            action = 8
                        else:
                            action = 6
                        break

        actions.append(action)

    actions.append(0)  # 最後一個數據沒有未來的數據可以比較，所以直接設定為0
    return actions

# 此為目前的supertrend最好的target, 所以要研究其他target, 要另外加一個function去研究, 這個就不要動了
@utilities.capture_args
def super_trend_strategy_v1(df, loss_percent_long=0.015, loss_percent_short=0.015, hold_count_at_least_profit=0.015, hold_count_limit=8):
    # 0, 1 2%
    # 2, 3 3%
    # 4, 5 4%
    # 6, 7 輸0.8%

    price_series = df['Close']
    open_series = df['Open']
    low_series = df['Low']
    high_series = df['High']

    super_trend_series = df['Super Trend']
    up_trend_series = df['Up Trend']
    down_trend_series = df['Down Trend']

    actions = []

    for idx in range(len(price_series) - 1):
        current_price = price_series[idx]
        current_high = high_series[idx]
        current_low = low_series[idx]

        super_trend = super_trend_series[idx]
        up_trend = up_trend_series[idx]
        down_trend = down_trend_series[idx]

        action = 0
        position = 0 #1為long, 2為short
        max_profit = 0

        if super_trend == 1:
            position = 1
        if super_trend == -1:
            position = 2

        if position != 0:
            check_end_index = min(len(price_series), idx + hold_count_limit + 1)
            for future_idx in range(idx + 1, check_end_index):
                future_price = price_series[future_idx]
                future_low = low_series[future_idx]
                future_high = high_series[future_idx]
            
                # 计算价格和成交量的变化百分比
                price_change_pct = indicators.calculate_percentage_change(future_price, current_price)
                high_change_pct = indicators.calculate_percentage_change(future_high, current_price)
                low_change_pct = indicators.calculate_percentage_change(future_low, current_price)
                if position == 1:
                    max_profit = max(max_profit, high_change_pct)
                    if -low_change_pct >= loss_percent_long:
                        # long止損
                        action = 7
                        break
                    elif (future_idx == check_end_index - 1):
                        if price_change_pct >= hold_count_at_least_profit:
                            action = 1
                        else:
                            action = 7
                        break 
                elif position == 2:
                    max_profit = max(max_profit, -low_change_pct)
                    if high_change_pct >= loss_percent_short:
                        action = 8
                        break
                    elif (future_idx == check_end_index - 1):
                        if -price_change_pct >= hold_count_at_least_profit:
                            action = 2
                        else:
                            action = 8
                        break

            if position == 1:
                if max_profit >= 0.04:
                    action = 5
                elif max_profit >= 0.03:
                    action = 3
                elif max_profit >= 0.02:
                    action = 1
            elif position == 2:
                if max_profit >= 0.04:
                    action = 6
                elif max_profit >= 0.03:
                    action = 4
                elif max_profit >= 0.02:
                    action = 2
        
        action -= 1
        actions.append(action)

    actions.append(-1)  # 最後一個數據沒有未來的數據可以比較，所以直接設定為0
    return actions

@utilities.capture_args
def super_trend_strategy_v2(df, loss_percent_long=0.015, loss_percent_short=0.015, hold_count_at_least_profit=0.015, hold_count_limit=8):
    # 0 不下單
    # 1, 2 2%
    # 3, 4 3%
    # 5, 6 輸0.8%

    price_series = df['Close']
    open_series = df['Open']
    low_series = df['Low']
    high_series = df['High']
    volume_series = df['Volume']

    super_trend_series = df['Super Trend']
    up_trend_series = df['Up Trend']
    down_trend_series = df['Down Trend']

    rsi_series = df['RSI']

    actions = []

    rsi_delta = 15

    for idx in range(len(price_series) - 1):
        current_price = price_series[idx]
        current_volume = volume_series[idx]
        current_high = high_series[idx]
        current_low = low_series[idx]

        super_trend = super_trend_series[idx]
        up_trend = up_trend_series[idx]
        down_trend = down_trend_series[idx]

        rsi = rsi_series[idx]

        action = 0
        position = 0 #1為long, 2為short
        max_profit = 0
        up_trend_pct = indicators.calculate_percentage_change(current_price, up_trend)
        down_trend_pct = indicators.calculate_percentage_change(current_price, down_trend)
        if super_trend == 1:
            position = 1
            max_rsi = rsi
        if super_trend == -1:
            position = 2
            min_rsi = rsi

        if position != 0:
            check_end_index = min(len(price_series), idx + hold_count_limit + 1)
            for future_idx in range(idx + 1, check_end_index):
                future_price = price_series[future_idx]
                future_low = low_series[future_idx]
                future_high = high_series[future_idx]
            
                # 计算价格和成交量的变化百分比
                price_change_pct = indicators.calculate_percentage_change(future_price, current_price)
                high_change_pct = indicators.calculate_percentage_change(future_high, current_price)
                low_change_pct = indicators.calculate_percentage_change(future_low, current_price)
                if position == 1:
                    # 假設回彈有低於剛剛的max_profit的一半, 就出場
                    max_profit = max(max_profit, high_change_pct)
                    max_rsi = max(max_rsi, rsi)
                    if -low_change_pct >= loss_percent_long:
                        # long止損
                        action = 3
                        break
                    elif max_rsi - rsi_delta <= rsi:
                        if max_profit <= 0.003:
                            action = 0
                        elif price_change_pct >= hold_count_at_least_profit:
                            action = 1
                        break
                    elif (future_idx == check_end_index - 1):
                        if max_profit <= 0.005:
                            action = 0
                        elif price_change_pct >= hold_count_at_least_profit:
                            action = 1
                        else:
                            action = 3
                        break 
                elif position == 2:
                    max_profit = max(max_profit, -low_change_pct)
                    min_rsi = max(min_rsi, rsi)
                    if high_change_pct >= loss_percent_short:
                        action = 4
                        break
                    elif min_rsi + rsi_delta >= rsi:
                        if max_profit <= 0.003:
                            action = 0
                        elif price_change_pct >= hold_count_at_least_profit:
                            action = 2
                        break
                    elif (future_idx == check_end_index - 1):
                        if max_profit <= 0.005:
                            action = 0
                        elif -price_change_pct >= hold_count_at_least_profit:
                            action = 2
                        else:
                            action = 4
                        break

            if position == 1:
                if max_profit >= 0.03:
                    action = 1
                elif max_profit >= 0.02:
                    action = 1
            elif position == 2:
                if max_profit >= 0.03:
                    action = 2
                elif max_profit >= 0.02:
                    action = 2
        
        actions.append(action)

    actions.append(0)  # 最後一個數據沒有未來的數據可以比較，所以直接設定為0
    return actions


@utilities.capture_args
def max_high_and_max_low_target(df, hold_count_limit=8):
    price_series = df['Close']
    low_series = df['Low']
    high_series = df['High']

    max_highs = []
    max_lows = []

    for idx in range(len(price_series) - 1):
        current_price = price_series[idx]

        # 初始化最大涨跌幅度为0
        max_high_pct = 0
        max_low_pct = 0


        check_end_index = min(len(price_series), idx + hold_count_limit + 1)
        for future_idx in range(idx + 1, check_end_index):
            future_low = low_series[future_idx]
            future_high = high_series[future_idx]
        
            # 计算价格和成交量的变化百分比
            high_change_pct = indicators.calculate_percentage_change(future_high, current_price)
            low_change_pct = indicators.calculate_percentage_change(future_low, current_price)

            # 更新最大涨跌幅度
            if high_change_pct > max_high_pct:
                max_high_pct = high_change_pct
            if low_change_pct < max_low_pct:  # 注意：下跌幅度应该取更小的值（更大的负数）
                max_low_pct = low_change_pct

        max_highs.append(max_high_pct)
        max_lows.append(max_low_pct)

    # 补齐最后hold_count_limit个数据点的输出为0，因为它们没有足够的未来数据进行比较
    max_highs.append(0)
    max_lows.append(0)

    return max_highs, max_lows

@utilities.capture_args
def super_trend_strategy_v3(df, loss_percent_long=0.015, loss_percent_short=0.015, hold_count_at_least_profit=0.015, hold_count_limit=8):
    # 0 hold_count_at_least_profit%以下
    # 1, 2 hold_count_at_least_profit% - 1%
    # 3, 4 1% - 2.0%
    # 5, 6 2% - 3%
    # 7, 8 3%up
    # 9, 10 輸loss_percent%
    PROFIT_THRESHOLD_1 = hold_count_at_least_profit
    PROFIT_THRESHOLD_2 = 0.01
    PROFIT_THRESHOLD_3 = 0.02
    PROFIT_THRESHOLD_4 = 0.03

    
    price_series = df['Close']
    low_series = df['Low']
    high_series = df['High']
    hist_series = df['Hist']
    rsi_series = df['RSI']
    volume_series = df['Volume']

    super_trend_series = df['Super Trend']
    up_trend_series = df['Up Trend']
    down_trend_series = df['Down Trend']

    actions = []

    for idx in range(len(price_series) - 1):
        current_price = price_series[idx]
        current_high = high_series[idx]
        current_low = low_series[idx]

        super_trend = super_trend_series[idx]
        up_trend = up_trend_series[idx]
        down_trend = down_trend_series[idx]

        hist = hist_series[idx]
        rsi = rsi_series[idx]
        volume = volume_series[idx]

        action = 0
        position = 0 #1為long, 2為short
        max_profit = 0

        if super_trend == 1 and hist > 0 and volume > 10000:
            position = 1
        if super_trend == -1 and hist <= 0 and volume > 10000:
            position = 2

        if position != 0:
            check_end_index = min(len(price_series), idx + hold_count_limit + 1)
            for future_idx in range(idx + 1, check_end_index):
                future_price = price_series[future_idx]
                future_low = low_series[future_idx]
                future_high = high_series[future_idx]
            
                # 计算价格和成交量的变化百分比
                price_change_pct = indicators.calculate_percentage_change(future_price, current_price)
                high_change_pct = indicators.calculate_percentage_change(future_high, current_price)
                low_change_pct = indicators.calculate_percentage_change(future_low, current_price)
                if position == 1:
                    max_profit = max(max_profit, high_change_pct)
                    if -low_change_pct >= loss_percent_long:
                        action = 9
                        break
                    elif (future_idx == check_end_index - 1):
                        if max_profit <= hold_count_at_least_profit:
                            action = 0
                        break
                elif position == 2:
                    max_profit = max(max_profit, -low_change_pct)
                    if high_change_pct >= loss_percent_short:
                        action = 10
                        break
                    elif (future_idx == check_end_index - 1):
                        if max_profit <= hold_count_at_least_profit:
                            action = 0
                        break

            if position == 1:
                if max_profit >= PROFIT_THRESHOLD_4:
                    action = 7
                if max_profit >= PROFIT_THRESHOLD_3:
                    action = 5
                elif max_profit >= PROFIT_THRESHOLD_2:
                    action = 3
                elif max_profit > PROFIT_THRESHOLD_1:
                    action = 1
            elif position == 2:
                if max_profit >= PROFIT_THRESHOLD_4:
                    action = 8
                if max_profit >= PROFIT_THRESHOLD_3:
                    action = 6
                elif max_profit >= PROFIT_THRESHOLD_2:
                    action = 4
                elif max_profit > PROFIT_THRESHOLD_1:
                    action = 2
        
        # action -= 1
        actions.append(action)

    actions.append(-1)  # 最後一個數據沒有未來的數據可以比較，所以直接設定為0
    return actions


@utilities.capture_args
def super_trend_strategy_v4(df, loss_percent_long=0.015, loss_percent_short=0.015, hold_count_at_least_profit=0.015, hold_count_limit=8):
    # 0 hold_count_at_least_profit%以下
    # 1, 2 hold_count_at_least_profit% - 1%
    # 3, 4 1% - 2.0%
    # 5, 6 2% - 3%
    # 7, 8 3%up
    # 9, 10 輸loss_percent%
    # PROFIT_THRESHOLD_1 = hold_count_at_least_profit
    # PROFIT_THRESHOLD_2 = 0.02
    # PROFIT_THRESHOLD_3 = 0.03
    # PROFIT_THRESHOLD_4 = 0.03

    PROFIT_THRESHOLD_1 = hold_count_at_least_profit
    PROFIT_THRESHOLD_2 = 0.01
    PROFIT_THRESHOLD_3 = 0.015
    PROFIT_THRESHOLD_4 = 0.02
    PROFIT_THRESHOLD_5 = 0.025
    PROFIT_THRESHOLD_6 = 0.03
    
    price_series = df['Close']
    low_series = df['Low']
    high_series = df['High']
    hist_series = df['Hist']
    rsi_series = df['RSI']
    volume_series = df['Volume']

    super_trend_series = df['Super Trend']
    up_trend_series = df['Up Trend']
    down_trend_series = df['Down Trend']

    actions = []

    for idx in range(len(price_series) - 1):
        current_price = price_series[idx]
        current_high = high_series[idx]
        current_low = low_series[idx]

        super_trend = super_trend_series[idx]
        up_trend = up_trend_series[idx]
        down_trend = down_trend_series[idx]

        hist = hist_series[idx]
        rsi = rsi_series[idx]
        volume = volume_series[idx]

        action = 0
        position = 0 #1為long, 2為short
        max_profit = 0

        if super_trend == 1:
            position = 1
        if super_trend == -1:
            position = 2

        if position != 0:
            check_end_index = min(len(price_series), idx + hold_count_limit + 1)
            for future_idx in range(idx + 1, check_end_index):
                future_price = price_series[future_idx]
                future_low = low_series[future_idx]
                future_high = high_series[future_idx]
            
                # 计算价格和成交量的变化百分比
                price_change_pct = indicators.calculate_percentage_change(future_price, current_price)
                high_change_pct = indicators.calculate_percentage_change(future_high, current_price)
                low_change_pct = indicators.calculate_percentage_change(future_low, current_price)
                if position == 1:
                    max_profit = max(max_profit, high_change_pct)
                    if -low_change_pct >= loss_percent_long:
                        action = 13
                        break
                    elif (future_idx == check_end_index - 1):
                        if price_change_pct >= hold_count_at_least_profit:
                            action = 1
                        break
                elif position == 2:
                    max_profit = max(max_profit, -low_change_pct)
                    if high_change_pct >= loss_percent_short:
                        action = 14
                        break
                    elif (future_idx == check_end_index - 1):
                        if -price_change_pct >= hold_count_at_least_profit:
                            action = 2
                        break

            if position == 1:
                if max_profit >= PROFIT_THRESHOLD_6:
                    action = 11
                elif max_profit >= PROFIT_THRESHOLD_5:
                    action = 9
                elif max_profit >= PROFIT_THRESHOLD_4:
                    action = 7
                elif max_profit >= PROFIT_THRESHOLD_3:
                    action = 5
                elif max_profit >= PROFIT_THRESHOLD_2:
                    action = 3
                elif max_profit >= PROFIT_THRESHOLD_1:
                    action = 1
            elif position == 2:
                if max_profit >= PROFIT_THRESHOLD_6:
                    action = 12
                elif max_profit >= PROFIT_THRESHOLD_5:
                    action = 10
                elif max_profit >= PROFIT_THRESHOLD_4:
                    action = 8
                elif max_profit >= PROFIT_THRESHOLD_3:
                    action = 6
                elif max_profit >= PROFIT_THRESHOLD_2:
                    action = 4
                elif max_profit >= PROFIT_THRESHOLD_1:
                    action = 2
        
        actions.append(action)

    actions.append(0)  # 最後一個數據沒有未來的數據可以比較，所以直接設定為0
    return actions

def prepare_features_and_targets(df, max_lookahead, cols, min_increase=0.001, max_increase=0.02, step=0.001):
    features = []
    targets_up = []
    targets_down = []

    for i in range(len(df) - max_lookahead):
        for lookahead in range(1, max_lookahead + 1):
            high_window = df['High'].iloc[i+1:i+lookahead+1].max()
            low_window = df['Low'].iloc[i+1:i+lookahead+1].min()
            current_close = df['Close'].iloc[i]
            
            for increase in np.arange(min_increase, max_increase + step, step):
                future_high = df['Close'].iloc[i] * (1 + increase)
                future_low = df['Close'].iloc[i] * (1 - increase)
                features.append([
                    df['Open'].iloc[i],
                    df['High'].iloc[i],
                    df['Low'].iloc[i],
                    df['Close'].iloc[i],
                    future_high,
                    future_low,
                    df['Volume'].iloc[i],
                    df['fibonacci_0.382'].iloc[i],
                    df['fibonacci_0.5'].iloc[i],
                    df['fibonacci_0.618'].iloc[i],
                    df['Open_Close_pct'].iloc[i],
                    df['High_Low_pct'].iloc[i],
                    df['Up_Shadow_pct'].iloc[i],
                    df['Down_Shadow_pct'].iloc[i],
                    df['EMA_7'].iloc[i],
                    df['EMA_25'].iloc[i],
                    df['EMA_99'].iloc[i],
                    df['RSI'].iloc[i],
                    df['Middle Band'].iloc[i],
                    df['Upper Band'].iloc[i],
                    df['Lower Band'].iloc[i],
                    df['ATR'].iloc[i],
                    lookahead,
                    increase
                ])
                targets_up.append(int((high_window - current_close) / current_close >= increase))
                targets_down.append(int((current_close - low_window) / current_close >= increase))

    return pd.DataFrame(features, columns=cols), np.array(targets_up), np.array(targets_down)

@utilities.capture_args
def pure_tp_sl_strategy(df, long_tp=0.015, short_tp=0.015, long_sl=0.0075, short_sl=0.0075, hold_count_limit=12):
    # 0 做多贏
    # 1 做空贏
    # 2 多空皆輸 / 盤整無波動

    price_series = df['Close']
    low_series = df['Low']
    high_series = df['High']

    cases = []

    for idx in range(len(price_series) - 1):
        current_price = price_series[idx]

        hit_long_tp = False
        hit_short_tp = False
        hit_long_sl = False
        hit_short_sl = False

        case = 0

        check_end_index = min(len(price_series), idx + hold_count_limit + 1)
        for future_idx in range(idx + 1, check_end_index):
            future_price = price_series[future_idx]
            future_low = low_series[future_idx]
            future_high = high_series[future_idx]

            high_change_pct = indicators.calculate_percentage_change(future_high, current_price)
            low_change_pct = indicators.calculate_percentage_change(future_low, current_price)

            if -low_change_pct >= long_sl:
                hit_long_sl = True

            if high_change_pct >= short_sl:
                hit_short_sl = True

            if hit_long_sl and hit_short_sl:
                case = 3
                break

            if high_change_pct >= long_tp:
                if not hit_long_sl:
                    case = 1
                    hit_long_tp = True
                    break
                else:
                    case = 3
                    break

            if -low_change_pct >= short_tp:
                if not hit_short_sl:
                    case = 2
                    hit_short_tp = True
                    break
                else:
                    case = 3
                    break

        if not hit_long_tp and not hit_short_tp:
            case = 3

        case -= 1
        cases.append(case)

    cases.append(-1)  # 最後一個數據沒有未來的數據可以比較，所以直接設定為0
    return cases
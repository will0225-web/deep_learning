import sys
import os
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gym
from stable_baselines3 import PPO  # 使用 Proximal Policy Optimization (PPO) 代理
from gym import spaces

import numpy as np
import pandas as pd

from modules import data as modules_data

# 定義交易環境
class TradingEnv(gym.Env):
    def __init__(self, input_data, lookahead, prob_threshold, tp=0.02, sl_range=(0.006, 0.006), fee=0.0012):
        super(TradingEnv, self).__init__()
        
        self.input_data = input_data
        self.lookahead = lookahead
        self.tp = tp  # 固定的止盈
        self.sl_min, self.sl_max = sl_range  # 動態止損範圍 (0.4% - 0.8%)
        self.fee = fee
        self.prob_threshold = prob_threshold
        self.current_step = 0
        
        # 記錄交易結果
        self.total_trades = 0  # 累計交易次數
        self.must_loss_trades = 0  # 必定虧損交易次數
        self.entry_price_not_good_trades = 0  # 進場價格不好交易次數
        self.correct_trades = 0  # 正確交易次數
        self.missed_trades = 0  # 錯過交易次數
        
        # 行動空間：模型將選擇具體的進場價格 (以當前價格上下波動為範圍)
        self.action_space = spaces.Box(low=-0.02, high=0, shape=(1,), dtype=np.float32)  # 行動空間為價格的變動範圍
        
        # 觀測空間
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.input_data.columns),), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.total_trades = 0  # 累計交易次數
        self.must_loss_trades = 0  # 必定虧損交易次數
        self.entry_price_not_good_trades = 0  # 進場價格不好交易次數
        self.correct_trades = 0  # 正確交易次數
        self.missed_trades = 0  # 錯過交易次數
        return self.input_data.iloc[self.current_step].values

    def step(self, action):
        if self.current_step >= len(self.input_data) - self.lookahead:
            done = True
            return self.input_data.iloc[self.current_step].values, 0, done, {"profit": 0, "entry_price": 0, "correct_trades": self.correct_trades, "missed_trades": self.missed_trades}
            
        # prob = self.input_data['prob_class_1'].iloc[self.current_step]
        prob = 0.5
        current_price = self.input_data['Close'].iloc[self.current_step]
        # volume = self.input_data['Volume'].iloc[self.current_step]

        
        # 選擇的進場價格是當前價格的變動
        entry_price = current_price * (1 + action[0])

        # 設置動態止損
        sl = np.random.uniform(self.sl_min, self.sl_max)

        lookahead_end = min(self.current_step + self.lookahead, len(self.input_data) - 1)
        future_highs = self.input_data['High'].iloc[self.current_step+1:lookahead_end + 1]
        future_lows = self.input_data['Low'].iloc[self.current_step+1:lookahead_end + 1]

        max_future_high = future_highs.max()
        min_future_low = future_lows.min()
        # 找出未來高點達到 TP 價格的位置
        tp_price = current_price * (1 + self.tp)
        tp_indices  = np.where(future_highs >= tp_price)[0]  # 找到高於 TP 價格的 index
        
        actual_profit = 0
        done = False
        reward = 0

        entry_index = -1
        tp_index = -1

        if prob >= self.prob_threshold:
            is_long_position = False  # 初始化
            future_lows_until_tp = future_lows
            future_highs_until_tp = future_highs
            entry_index = None
            # 確保有一個符合條件的 TP index
            if len(tp_indices) > 0:
                tp_index = tp_indices[0]  # 最早触发 TP 的位置

                # 获取从当前时刻到 tp_index 的 future_lows
                future_lows_until_tp = future_lows.iloc[:tp_index+1]
                future_highs_until_tp = future_highs.iloc[:tp_index+1]

                max_future_high = future_highs_until_tp.max()
                min_future_low = future_lows_until_tp.min()

                entry_condition = (future_lows_until_tp <= entry_price)
                # 检查在此期间是否有机会以 entry_price 进场
                if entry_condition.any():
                    entry_index = entry_condition.idxmax()
                    if entry_index < tp_index:
                        future_lows_until_tp = future_lows.loc[entry_index:tp_index+1]
                        future_highs_until_tp = future_highs.loc[entry_index:tp_index+1]
                        max_future_high = future_highs_until_tp.max()
                        min_future_low = future_lows_until_tp.min()
                        is_long_position = True
                else:
                    is_long_position = False
            else:
                pass
                # 先不判定沒有TP的進場
                # 如果没有触发 TP，我们依然需要检查是否有机会进场
                # if (future_lows <= entry_price).any():
                #     is_long_position = True
                # else:
                #     is_long_position = False

            sl_price = entry_price * (1 - sl)
            tp_price = current_price * (1 + self.tp)
            final_price = self.input_data['Close'].iloc[self.current_step + self.lookahead]
            entry_price_profit = (final_price - entry_price) / entry_price

            if is_long_position:
                # 如果觸發止損
                if min_future_low <= sl_price:
                    if action[0] <= -0.02:
                        reward = 0
                        self.must_loss_trades += 1
                    else:
                        actual_profit = -sl - self.fee
                        
                        # 檢查未來是否觸發 TP
                        if max_future_high >= tp_price:
                            distance_from_min_future_low_to_tp = (tp_price - min_future_low) / min_future_low
                            # 根據距離進行懲罰，距離越大，懲罰越大
                            distance_from_min_to_entry = ((min_future_low - entry_price) / entry_price)

                            reward = actual_profit - abs(distance_from_min_future_low_to_tp) - abs(distance_from_min_to_entry)
                            self.entry_price_not_good_trades += 1
                        elif entry_price_profit >= 0:
                            # 正常情況下，根據距離止損價格的遠近進行懲罰
                            reward = 0
                            # reward = entry_price_profit
                            # self.missed_trades += 1
                            # self.wrong_trades += 1
                        else:
                            reward = 0
                # 如果觸發止盈
                elif max_future_high >= tp_price:
                    # 使用 entry_price 來計算實際的盈利
                    actual_profit = (tp_price - entry_price) / entry_price - self.fee
                    
                    if current_price * (1 - sl) >= min_future_low:
                        reward = actual_profit * 2
                    else:
                        # 根據盈利給予獎勵
                        reward = actual_profit  # 獎勵

                    self.correct_trades += 1
                else:
                    # 計算未觸發止損或止盈的情況下的結果
                    actual_profit = ((final_price - entry_price) / entry_price) - self.fee
                    # reward = actual_profit
                    reward = 0
            else:
                # 沒有進場的狀態
                # 檢查是否觸及止盈 (TP)
                if max_future_high >= tp_price:
                    # 如果有進場少賺的距離(最大化)
                    entry_price_to_tp_price_distance = ((tp_price - min_future_low) / min_future_low)

                    # entry price跟這段期間的最低點相差的距離，意旨錯了多遠
                    distance_from_min_to_entry = ((min_future_low - entry_price) / entry_price)
                    
                    # 懲罰未進場但有盈利的情況，懲罰程度根據最低價格與進場價格的距離
                    reward = (-abs(entry_price_to_tp_price_distance) - abs(distance_from_min_to_entry) - sl) * 2 # 可以調整懲罰比例

                    self.missed_trades += 1
                # elif potential_profit >= 0.1:
                #     # 如果未觸發TP，但有潛在盈利，進行輕微懲罰
                #     reward = -abs(potential_profit)

                #     # self.missed_trades += 1
                # else:
                #     # 無盈利或虧損時，正常給予輕微獎勵或懲罰
                #     reward = potential_profit * 0.2  # 給予虧損或0的獎勵
                    # reward = 0

        self.current_step += 1
        
        
        return self.input_data.iloc[self.current_step].values, reward, done, {"profit": actual_profit, "entry_price": entry_price, "correct_trades": self.correct_trades, "missed_trades": self.missed_trades}
    
    def render(self):
        pass
    
    def print_summary(self):
        # 打印總結結果
        print(f"Correct Trades: {self.correct_trades}")
        print(f"Must Loss Trades: {self.must_loss_trades}")
        print(f"Entry Price Not Good Trades: {self.entry_price_not_good_trades}")
        print(f"Missed Trades: {self.missed_trades}")
        print("-----------------------------------")


# # 定義交易環境
# class TradingEnv(gym.Env):
#     def __init__(self, input_data, lookahead, prob_threshold, tp=0.02, sl_range=(0.006, 0.006), fee=0.0012):
#         super(TradingEnv, self).__init__()
        
#         self.input_data = input_data
#         self.lookahead = lookahead
#         self.tp = tp  # 固定的止盈
#         self.sl_min, self.sl_max = sl_range  # 動態止損範圍 (0.4% - 0.8%)
#         self.fee = fee
#         self.prob_threshold = prob_threshold
#         self.current_step = 0
        
#         # 記錄交易結果
#         self.total_trades = 0  # 累計交易次數
#         self.must_loss_trades = 0  # 必定虧損交易次數
#         self.entry_price_not_good_trades = 0  # 進場價格不好交易次數
#         self.correct_trades = 0  # 正確交易次數
#         self.missed_trades = 0  # 錯過交易次數
        
#         # 行動空間：模型將選擇具體的進場價格 (以當前價格上下波動為範圍)
#         self.action_space = spaces.Box(low=-0.02, high=0, shape=(1,), dtype=np.float32)  # 行動空間為價格的變動範圍
        
#         # 觀測空間
#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.input_data.columns),), dtype=np.float32)

#     def reset(self):
#         self.current_step = 0
#         self.total_trades = 0  # 累計交易次數
#         self.must_loss_trades = 0  # 必定虧損交易次數
#         self.entry_price_not_good_trades = 0  # 進場價格不好交易次數
#         self.correct_trades = 0  # 正確交易次數
#         self.missed_trades = 0  # 錯過交易次數
#         return self.input_data.iloc[self.current_step].values

#     def step(self, action):
#         if self.current_step >= len(self.input_data) - self.lookahead:
#             done = True
#             return self.input_data.iloc[self.current_step].values, 0, done, {"profit": 0, "entry_price": 0, "correct_trades": self.correct_trades, "missed_trades": self.missed_trades}
            
#         # prob = self.input_data['prob_class_1'].iloc[self.current_step]
#         prob = 0.5
#         current_price = self.input_data['Close'].iloc[self.current_step]
#         # volume = self.input_data['Volume'].iloc[self.current_step]

        
#         # 選擇的進場價格是當前價格的變動
#         entry_price = current_price * (1 + action[0])

#         # 設置動態止損
#         sl = np.random.uniform(self.sl_min, self.sl_max)

#         lookahead_end = min(self.current_step + self.lookahead, len(self.input_data) - 1)
#         future_highs = self.input_data['High'].iloc[self.current_step+1:lookahead_end + 1]
#         future_lows = self.input_data['Low'].iloc[self.current_step+1:lookahead_end + 1]

#         max_future_high = future_highs.max()
#         min_future_low = future_lows.min()
#         # 找出未來高點達到 TP 價格的位置
#         tp_price = current_price * (1 + self.tp)
#         tp_indices  = np.where(future_highs >= tp_price)[0]  # 找到高於 TP 價格的 index
        
#         actual_profit = 0
#         done = False
#         reward = 0

#         entry_index = -1
#         tp_index = -1

#         if prob >= self.prob_threshold:
#             is_long_position = False  # 初始化
#             future_lows_until_tp = future_lows
#             future_highs_until_tp = future_highs
#             entry_index = None
#             # 確保有一個符合條件的 TP index
#             if len(tp_indices) > 0:
#                 tp_index = tp_indices[0]  # 最早触发 TP 的位置

#                 # 获取从当前时刻到 tp_index 的 future_lows
#                 future_lows_until_tp = future_lows.iloc[:tp_index+1]
#                 future_highs_until_tp = future_highs.iloc[:tp_index+1]

#                 max_future_high = future_highs_until_tp.max()
#                 min_future_low = future_lows_until_tp.min()

#                 entry_condition = (future_lows_until_tp <= entry_price)
#                 # 检查在此期间是否有机会以 entry_price 进场
#                 if entry_condition.any():
#                     entry_index = entry_condition.idxmax()
#                     if entry_index < tp_index:
#                         future_lows_until_tp = future_lows.loc[entry_index:tp_index+1]
#                         future_highs_until_tp = future_highs.loc[entry_index:tp_index+1]
#                         max_future_high = future_highs_until_tp.max()
#                         min_future_low = future_lows_until_tp.min()
#                         is_long_position = True
#                 else:
#                     is_long_position = False
#             else:
#                 pass
#                 # 先不判定沒有TP的進場
#                 # 如果没有触发 TP，我们依然需要检查是否有机会进场
#                 # if (future_lows <= entry_price).any():
#                 #     is_long_position = True
#                 # else:
#                 #     is_long_position = False

#             sl_price = entry_price * (1 - sl)
#             tp_price = current_price * (1 + self.tp)
#             final_price = self.input_data['Close'].iloc[self.current_step + self.lookahead]
#             entry_price_profit = (final_price - entry_price) / entry_price

#             if is_long_position:
#                 # 如果觸發止損
#                 if min_future_low <= sl_price:
#                     if action[0] <= -0.02:
#                         reward = 0
#                         self.must_loss_trades += 1
#                     else:
#                         actual_profit = -sl - self.fee
                        
#                         # 檢查未來是否觸發 TP
#                         if max_future_high >= tp_price:
#                             distance_from_min_future_low_to_tp = (tp_price - min_future_low) / min_future_low
#                             # 根據距離進行懲罰，距離越大，懲罰越大
#                             distance_from_min_to_entry = ((min_future_low - entry_price) / entry_price)

#                             reward = actual_profit - abs(distance_from_min_future_low_to_tp) - abs(distance_from_min_to_entry)
#                             self.entry_price_not_good_trades += 1
#                         elif entry_price_profit >= 0:
#                             # 正常情況下，根據距離止損價格的遠近進行懲罰
#                             reward = 0
#                             # reward = entry_price_profit
#                             # self.missed_trades += 1
#                             # self.wrong_trades += 1
#                         else:
#                             reward = 0
#                 # 如果觸發止盈
#                 elif max_future_high >= tp_price:
#                     # 使用 entry_price 來計算實際的盈利
#                     actual_profit = (tp_price - entry_price) / entry_price - self.fee
                    
#                     if current_price * (1 - sl) >= min_future_low:
#                         reward = actual_profit * 2
#                     else:
#                         # 根據盈利給予獎勵
#                         reward = actual_profit  # 獎勵

#                     self.correct_trades += 1
#                 else:
#                     # 計算未觸發止損或止盈的情況下的結果
#                     actual_profit = ((final_price - entry_price) / entry_price) - self.fee
#                     # reward = actual_profit
#                     reward = 0
#             else:
#                 # 沒有進場的狀態
#                 # 檢查是否觸及止盈 (TP)
#                 if max_future_high >= tp_price:
#                     # 如果有進場少賺的距離(最大化)
#                     entry_price_to_tp_price_distance = ((tp_price - min_future_low) / min_future_low)

#                     # entry price跟這段期間的最低點相差的距離，意旨錯了多遠
#                     distance_from_min_to_entry = ((min_future_low - entry_price) / entry_price)
                    
#                     # 懲罰未進場但有盈利的情況，懲罰程度根據最低價格與進場價格的距離
#                     reward = (-abs(entry_price_to_tp_price_distance) - abs(distance_from_min_to_entry) - sl) * 2 # 可以調整懲罰比例

#                     self.missed_trades += 1
#                 # elif potential_profit >= 0.1:
#                 #     # 如果未觸發TP，但有潛在盈利，進行輕微懲罰
#                 #     reward = -abs(potential_profit)

#                 #     # self.missed_trades += 1
#                 # else:
#                 #     # 無盈利或虧損時，正常給予輕微獎勵或懲罰
#                 #     reward = potential_profit * 0.2  # 給予虧損或0的獎勵
#                     # reward = 0

#         self.current_step += 1
        
        
#         return self.input_data.iloc[self.current_step].values, reward, done, {"profit": actual_profit, "entry_price": entry_price, "correct_trades": self.correct_trades, "missed_trades": self.missed_trades}
    
#     def render(self):
#         pass
    
#     def print_summary(self):
#         # 打印總結結果
#         print(f"Correct Trades: {self.correct_trades}")
#         print(f"Must Loss Trades: {self.must_loss_trades}")
#         print(f"Entry Price Not Good Trades: {self.entry_price_not_good_trades}")
#         print(f"Missed Trades: {self.missed_trades}")
#         print("-----------------------------------")

def check_logic(df, entry_price, tp, lookahead):
    # prob = self.input_data['prob_class_1'].iloc[self.current_step]
    fee = 0.0012
    prob = 0.5
    current_price = df['Close'].iloc[0]
    # volume = self.input_data['Volume'].iloc[self.current_step]

    
    # 選擇的進場價格是當前價格的變動
    # entry_price = current_price * (1 + tp)

    # 設置動態止損
    sl = 0.006

    lookahead_end = min(lookahead, len(df) - 1)
    future_highs = df['High'].iloc[1:lookahead_end + 1]
    future_lows = df['Low'].iloc[1:lookahead_end + 1]
    max_future_high = future_highs.max()
    min_future_low = future_lows.min()
    # 找出未來高點達到 TP 價格的位置
    tp_price = current_price * (1 + tp)
    tp_indices  = np.where(future_highs >= tp_price)[0]  # 找到高於 TP 價格的 index
    actual_profit = 0
    done = False
    reward = 0

    entry_index = -1
    tp_index = -1
    if prob >= 0:
        is_long_position = False  # 初始化
        future_lows_until_tp = future_lows
        future_highs_until_tp = future_highs
        # 確保有一個符合條件的 TP index
        
        if len(tp_indices) > 0:
            tp_index = tp_indices[0]  # 最早触发 TP 的位置
            
            # 获取从当前时刻到 tp_index 的 future_lows
            future_lows_until_tp = future_lows.iloc[:tp_index+1]
            future_highs_until_tp = future_highs.iloc[:tp_index+1]
            
            max_future_high = future_highs_until_tp.max()
            min_future_low = future_lows_until_tp.min()


            entry_condition = (future_lows_until_tp <= entry_price)
            # 检查在此期间是否有机会以 entry_price 进场
            if entry_condition.any():
                entry_index = entry_condition.idxmax()
                print(entry_index, tp_index)
                if entry_index < tp_index:
                    future_lows_until_tp = future_lows.loc[entry_index:tp_index+1]
                    future_highs_until_tp = future_highs.loc[entry_index:tp_index+1]
                    max_future_high = future_highs_until_tp.max()
                    min_future_low = future_lows_until_tp.min()
                    is_long_position = True
            else:
                is_long_position = False
        else:
            pass
            # 先不判定沒有TP的進場
            # 如果没有触发 TP，我们依然需要检查是否有机会进场
            # if (future_lows <= entry_price).any():
            #     is_long_position = True
            # else:
            #     is_long_position = False

        sl_price = entry_price * (1 - sl)
        tp_price = current_price * (1 + tp)
        final_price = df['Close'].iloc[lookahead]
        entry_price_profit = (final_price - entry_price) / entry_price

        if is_long_position:
            # 如果觸發止損
            if min_future_low <= sl_price:
                if 0.03 <= -0.02:
                    reward = 0
                    # self.must_loss_trades += 1
                else:
                    actual_profit = -sl - fee
                    
                    # 檢查未來是否觸發 TP
                    if max_future_high >= tp_price:
                        distance_from_min_future_low_to_tp = (tp_price - min_future_low) / min_future_low
                        # 根據距離進行懲罰，距離越大，懲罰越大
                        distance_from_min_to_entry = ((min_future_low - entry_price) / entry_price)

                        reward = actual_profit - abs(distance_from_min_future_low_to_tp) - abs(distance_from_min_to_entry)
                        # self.entry_price_not_good_trades += 1
                    elif entry_price_profit >= 0:
                        # 正常情況下，根據距離止損價格的遠近進行懲罰
                        reward = 0
                        # reward = entry_price_profit
                        # self.missed_trades += 1
                        # self.wrong_trades += 1
                    else:
                        reward = 0
            # 如果觸發止盈
            elif max_future_high >= tp_price:
                # 使用 entry_price 來計算實際的盈利
                actual_profit = (tp_price - entry_price) / entry_price - fee
                
                if current_price * (1 - sl) >= min_future_low:
                    reward = actual_profit * 2
                else:
                    # 根據盈利給予獎勵
                    reward = actual_profit  # 獎勵

                # self.correct_trades += 1
            else:
                # 計算未觸發止損或止盈的情況下的結果
                actual_profit = ((final_price - entry_price) / entry_price) - fee
                # reward = actual_profit
                reward = 0
        else:
            # 沒有進場的狀態
            # 檢查是否觸及止盈 (TP)
            if max_future_high >= tp_price:
                if entry_index < tp_index:
                    # 如果有進場少賺的距離(最大化)
                    entry_price_to_tp_price_distance = ((tp_price - min_future_low) / min_future_low)

                    # entry price跟這段期間的最低點相差的距離，意旨錯了多遠
                    distance_from_min_to_entry = ((min_future_low - entry_price) / entry_price)
                    
                    # 懲罰未進場但有盈利的情況，懲罰程度根據最低價格與進場價格的距離
                    reward = (-abs(entry_price_to_tp_price_distance) - abs(distance_from_min_to_entry) - sl) * 2  # 可以調整懲罰比例
                else:
                    # 如果有進場少賺的距離(最大化)
                    entry_price_to_tp_price_distance = ((tp_price - min_future_low) / min_future_low)

                    # entry price跟這段期間的最低點相差的距離，意旨錯了多遠
                    distance_from_min_to_entry = ((min_future_low - entry_price) / entry_price)
                    
                    # 懲罰未進場但有盈利的情況，懲罰程度根據最低價格與進場價格的距離
                    reward = (-abs(entry_price_to_tp_price_distance) - abs(distance_from_min_to_entry) - sl) * 2# 可以調整懲罰比例
                # self.missed_trades += 1
            # elif potential_profit >= 0.1:
            #     # 如果未觸發TP，但有潛在盈利，進行輕微懲罰
            #     reward = -abs(potential_profit)

            #     # self.missed_trades += 1
            # else:
            #     # 無盈利或虧損時，正常給予輕微獎勵或懲罰
            #     reward = potential_profit * 0.2  # 給予虧損或0的獎勵
                # reward = 0
        print(f"actual_profit: {actual_profit}, reward: {reward}")

lookahead = 24
percentage = 0.015
prob_threshold = 0

df = modules_data.generate_kline(lookahead + 1, 2238, 2238 * (1 + percentage), 3, 2, 2230)

# validation_env = TradingEnv(input_data=df, lookahead=lookahead, prob_threshold=prob_threshold, tp=percentage)
# rl_model = PPO("MlpPolicy", validation_env, verbose=1)

# # 開始訓練
# rl_model.learn(total_timesteps=10000)
# # 測試模型策略並進行回測
# obs = validation_env.reset()
# done = False
# total_profit = 0
# total_reward = 0  # 用來累計總reward
# profits = []
# while not done:
#     action, _states = rl_model.predict(obs)
#     obs, reward, done, info = validation_env.step(action)
#     total_profit += info['profit']  # 使用 `profit` 而不是 `reward`
#     total_reward += reward  # 累計每步的 reward
#     profits.append(total_profit)
# validation_env.print_summary()
# print(f"總收益：{total_profit}")
# print(f"總獎勵：{total_reward}")

check_logic(df, df['Open'].iloc[1], percentage, lookahead)
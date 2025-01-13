import sys
import os
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import datetime
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import model as model_process
from modules import custom_model_fit_indicators as custom_model_fit_indicators
from keras.callbacks import ReduceLROnPlateau, TensorBoard

import xgboost as xgb
import csv
from sklearn.preprocessing import OneHotEncoder

import gym
from stable_baselines3 import PPO  # 使用 Proximal Policy Optimization (PPO) 代理
from gym import spaces
import matplotlib.pyplot as plt
from scipy.stats import skew
import torch
from gym.spaces import MultiDiscrete

# 定義交易環境
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

#         self.lowest_entry_price_percentage = -0.015

#         # 行動空間：模型將選擇具體的進場價格 (以當前價格上下波動為範圍)
#         self.action_space = spaces.Box(low=self.lowest_entry_price_percentage, high=0, shape=(1,), dtype=np.float32)  # 行動空間為價格的變動範圍
        
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
#         prob = self.input_data['prob_class_1'].iloc[self.current_step]
#         # prob = 0.5
#         current_price = self.input_data['Close'].iloc[self.current_step]
#         volume = self.input_data['Volume'].iloc[self.current_step]
#         # 選擇的進場價格是當前價格的變動
#         entry_price = current_price * (1 + action[0])
#         # 設置動態止損
#         sl = np.random.uniform(self.sl_min, self.sl_max)

#         if self.current_step >= len(self.input_data) - self.lookahead:
#             done = True
#             return self.input_data.iloc[self.current_step].values, 0, done, {"profit": 0, "entry_price": entry_price, "correct_trades": self.correct_trades, "missed_trades": self.missed_trades}
            
#         lookahead_end = min(self.current_step + self.lookahead, len(self.input_data) - 1)
#         future_highs = self.input_data['High'].iloc[self.current_step+1:lookahead_end + 1]
#         future_lows = self.input_data['Low'].iloc[self.current_step+1:lookahead_end + 1]

#         max_future_high = future_highs.max()
#         min_future_low = future_lows.min()

#         actual_profit = 0
#         done = False
#         reward = 0

#         entry_index = None
#         tp_index = None

#         # 找出未來高點達到 TP 價格的位置
#         tp_price = current_price * (1 + self.tp)
#         tp_condition = (future_highs >= tp_price)

#         is_long_position = False  # 初始化
#         future_lows_until_tp = future_lows
#         future_highs_until_tp = future_highs
#         has_tp = tp_condition.any()
#         # 確保有一個符合條件的 TP index
#         if has_tp:
#             tp_index = tp_condition.idxmax()  # 最早触发 TP 的位置

#             # 获取从当前时刻到 tp_index 的 future_lows
#             future_lows_until_tp = future_lows.iloc[:tp_index+1]
#             future_highs_until_tp = future_highs.iloc[:tp_index+1]

#             max_future_high = future_highs_until_tp.max()
#             min_future_low = future_lows_until_tp.min()

#             entry_condition = (future_lows_until_tp <= entry_price)
#             # 检查在此期间是否有机会以 entry_price 进场
#             if entry_condition.any():
#                 entry_index = entry_condition.idxmax()
#                 if entry_index <= tp_index:
#                     future_lows_until_tp = future_lows.iloc[entry_index:tp_index+1]
#                     future_highs_until_tp = future_highs.iloc[entry_index:tp_index+1]
#                     max_future_high = future_highs_until_tp.max()
#                     min_future_low = future_lows_until_tp.min()
#                     is_long_position = True
#             else:
#                 is_long_position = False
                
#         else:
#             pass
#             # 先不判定沒有TP的進場
#             # 如果没有触发 TP，我们依然需要检查是否有机会进场
#             # if (future_lows <= entry_price).any():
#             #     is_long_position = True
#             # else:
#             #     is_long_position = False

#         sl_price = entry_price * (1 - sl)
#         tp_price = current_price * (1 + self.tp)
#         final_price = self.input_data['Close'].iloc[self.current_step + self.lookahead]
#         entry_price_profit = (final_price - entry_price) / entry_price
#         distance_from_min_future_low_to_tp = (tp_price - min_future_low) / min_future_low
#         distance_from_min_to_entry = ((min_future_low - entry_price) / entry_price)
#         entry_price_to_tp_price_distance = ((tp_price - entry_price) / entry_price)


#         if is_long_position:
#             # 如果觸發止損
#             if min_future_low <= sl_price:
#                 actual_profit = -sl - self.fee
#                 if action[0] <= self.lowest_entry_price_percentage:
#                     reward = 0
#                     self.must_loss_trades += 1
#                 else:
#                     # 檢查未來是否觸發 TP
#                     if max_future_high >= tp_price:
#                         reward = -abs(distance_from_min_to_entry) * 3
#                         self.entry_price_not_good_trades += 1
#                     # elif entry_price_profit >= 0:
#                         # 正常情況下，根據距離止損價格的遠近進行懲罰
#                         # reward = 0
#                         # reward = entry_price_profit
#                         # self.missed_trades += 1
#                         # self.wrong_trades += 1
#                     else:
#                         reward = 0
#             # 如果觸發止盈
#             elif max_future_high >= tp_price:
#                 # 使用 entry_price 來計算實際的盈利
#                 actual_profit = entry_price_to_tp_price_distance - self.fee
                
#                 if current_price * (1 - sl) >= min_future_low:
#                     reward = -abs(distance_from_min_to_entry)
#                 else:
#                     # 根據盈利給予獎勵
#                     reward = -abs(distance_from_min_to_entry) * 1.5  # 獎勵

#                 self.correct_trades += 1
#             else:
#                 # 計算未觸發止損或止盈的情況下的結果
#                 actual_profit = entry_price_profit - self.fee
#                 # reward = actual_profit
#                 reward = 0
            
#         else:
#             # 沒有進場的狀態
#             potential_profit = (final_price - min_future_low) / min_future_low
#             # 檢查是否觸及止盈 (TP)
#             if max_future_high >= tp_price:
#                 # 懲罰未進場但有盈利的情況，懲罰程度根據最低價格與進場價格的距離
#                 # reward = (-abs(distance_from_min_future_low_to_tp) - abs(distance_from_min_to_entry) - sl) * 2 # 可以調整懲罰比例
#                 reward = -abs(distance_from_min_to_entry) * 5
#                 self.missed_trades += 1
#             # elif (final_price - min_future_low) / min_future_low >= 0:
#             #     # 如果未觸發TP，但有潛在盈利，進行輕微懲罰
#             #     reward = -abs(distance_from_min_to_entry) * 2.5
#                 # self.missed_trades += 1
#             # else:
#             #     # 無盈利或虧損時，正常給予輕微獎勵或懲罰
#             #     reward = -abs(distance_from_min_to_entry) * 2.7 # 給予虧損或0的獎勵

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



# class TradingEnv(gym.Env):
#     def __init__(self, price_data, balance=100000, lookback_window=20, missed_window=10, pending_order_duration=12, order_amount=1000):
#         super(TradingEnv, self).__init__()
        
#         self.price_data = price_data.reset_index(drop=True)  # 确保索引连续，从0开始
#         self.lookback_window = lookback_window
#         self.current_step = self.lookback_window
#         self.position = 0  # 0：无持仓，1：持有多头
#         self.balance = balance  # 初始资金
#         self.entry_price = 0
#         self.tp_price = 0
#         self.sl_price = 0
#         self.order_amount = order_amount  # 每次交易的下单金额

#         self.total_trades = 0          # 总交易次数
#         self.winning_trades = 0        # 盈利交易次数
#         self.balance_history = []      # 用于记录资金余额的列表
#         self.total_profit_percent = [] # 用于记录总利润百分比的列表

#         self.transaction_cost = 0.0005  # 手续费比例，例如万分之五
#         self.missed_window = missed_window  # 用于计算错过的潜在利润的窗口大小
#         self.pending_order_duration = pending_order_duration  # 挂单有效期，12 个时间步

#         # 动作空间设置
#         # 动作空间：进场/出场决策，入场价格调整，TP 调整，SL 调整
#         # decision: 0-不操作，1-买入，2-平仓
#         # entry_adjustment: 0% - 3%，步长为0.1%，共有31个级别
#         # tp_adjustment: 0.8% - 3%，步长为0.1%，共有23个级别
#         # sl_adjustment: 0.5% - 1.5%，步长为0.1%，共有11个级别

#         entry_steps = int((3.0 - 0.0) / 0.1) + 1  # 共有31个级别（0%到3%）
#         tp_steps = int((3.0 - 0.8) / 0.1) + 1     # 共有23个级别
#         sl_steps = int((1.5 - 0.5) / 0.1) + 1     # 共有11个级别

#         self.action_space = gym.spaces.MultiDiscrete([3, entry_steps, tp_steps, sl_steps])
        
#         # 观察空间：回溯窗口的价格数据和技术指标等
#         n_features = self.price_data.shape[1]  # 例如，包含 OHLC 和技术指标
#         self.observation_space = gym.spaces.Box(
#             low=-np.inf, high=np.inf, shape=(self.lookback_window, n_features), dtype=np.float32)
        
#         # 新增 pending_order 变量，用于存储挂单信息
#         self.pending_order = None
        
#     def reset(self):
#         self.current_step = self.lookback_window
#         self.position = 0
#         self.balance = 100000  # 与初始资金一致
#         self.entry_price = 0
#         self.tp_price = 0
#         self.sl_price = 0
#         self.order_amount = 1000  # 固定下单金额
#         self.pending_order = None
#         self.total_trades = 0
#         self.winning_trades = 0
#         self.balance_history = []
#         self.total_profit_percent = []
#         return self._next_observation()
    
#     def _next_observation(self):
#         # 获取回溯窗口内的价格数据和技术指标
#         obs = self.price_data.iloc[self.current_step - self.lookback_window:self.current_step]
#         # 将 DataFrame 转换为 numpy 数组
#         obs = obs.values
#         return obs
    
#     def step(self, action):
#         decision, entry_adjustment, tp_adjustment, sl_adjustment = action
#         current_price = self.price_data.iloc[self.current_step]['Close']  # 当前收盘价

#         # 入场价格调整：最多下调 3%
#         entry_adjust_percent = entry_adjustment * 0.1  # 0% 到 3%，步长 0.1%
#         adjusted_entry_price = current_price - current_price * entry_adjust_percent / 100

#         # TP 和 SL 调整
#         tp_percent = 0.8 + tp_adjustment * 0.1  # 0.8% 到 3%，步长 0.1%
#         sl_percent = 0.5 + sl_adjustment * 0.1  # 0.5% 到 1.5%，步长 0.1%

#         reward = 0
#         done = False

#         # 检查是否有未成交的挂单
#         if self.pending_order:
#             order_age = self.current_step - self.pending_order['order_step']
#             # 检查当前最低价是否达到挂单价格
#             if self.price_data.iloc[self.current_step]['Low'] <= self.pending_order['price']:
#                 # 挂单成交，开仓
#                 self.position = 1
#                 self.entry_price = self.pending_order['price']
#                 self.tp_price = self.entry_price + self.entry_price * self.pending_order['tp_percent'] / 100
#                 self.sl_price = self.entry_price - self.entry_price * self.pending_order['sl_percent'] / 100
#                 self.balance -= self.pending_order['total_entry_cost']  # 扣除下单金额和交易手续费
#                 self.pending_order = None  # 清除挂单

#                 # 评估挂单后的表现
#                 order_reward = self._evaluate_order_limit()
#                 reward += order_reward
#             elif order_age >= self.pending_order_duration:
#                 # 挂单过期，未成交，给予惩罚
#                 reward -= 1  # 未成交的轻微惩罚
#                 self.pending_order = None  # 清除挂单

#                 # 计算错过的潜在利润作为惩罚
#                 missed_profit_percent = self._calculate_missed_profit()
#                 if missed_profit_percent > 0:
#                     penalty = self._calculate_missed_profit_penalty(missed_profit_percent)
#                     reward -= penalty  # 惩罚是错过的潜在利润

#         if self.position == 0 and not self.pending_order:
#             if decision == 1:  # 开限价订单
#                 # 计算交易手续费（假设使用限价单）
#                 entry_fee = self.transaction_cost * self.order_amount
#                 total_entry_cost = self.order_amount + entry_fee

#                 # 检查是否有足够的资金开仓
#                 if self.balance >= total_entry_cost:
#                     # 创建挂单，记录挂单信息
#                     self.pending_order = {
#                         'price': adjusted_entry_price,
#                         'tp_percent': tp_percent,
#                         'sl_percent': sl_percent,
#                         'total_entry_cost': total_entry_cost,
#                         'order_step': self.current_step
#                     }

#                     reward = self._evaluate_create_order_limit_decision()
#                 else:
#                     # 资金不足，不能开仓，给予惩罚
#                     reward = 0
#                     # reward -= 10  # 资金不足的惩罚
#             else:
#                 # 未开单，检查错过的潜在利润作为惩罚
#                 missed_profit_percent = self._calculate_missed_profit()
#                 if missed_profit_percent > 0.6:
#                     penalty = self._calculate_missed_profit_penalty(missed_profit_percent)
#                     reward -= penalty  # 惩罚是错过的潜在利润
#         else:
#             # 不操作，可以给予轻微惩罚或奖励
#             reward -= 0.1

        

#         # 在每个时间步记录余额
#         self.balance_history.append(self.balance)

#         self.current_step += 1
#         if self.current_step >= len(self.price_data):
#             done = True
#         else:
#             done = False

#         obs = self._next_observation()
#         return obs, reward, done, {}

#     def _calculate_missed_profit(self):
#         future_window = self.price_data.iloc[self.current_step:self.current_step + self.missed_window]
#         if future_window.empty:
#             return 0
#         future_high = future_window['High'].max()
#         potential_profit = future_high - self.price_data.iloc[self.current_step]['Close']
#         potential_profit_percent = potential_profit / self.price_data.iloc[self.current_step]['Close'] * 100
#         potential_profit_percent -= self.transaction_cost * 100  # 考虑交易成本，转换为百分比
#         return max(potential_profit_percent, 0)  # 确保不为负值
    
    # def _calculate_missed_profit_penalty(self, missed_profit_percent):
    #     """
    #     根据错过的利润百分比，计算对应的惩罚值。
    #     """
    #     if missed_profit_percent < 0.7:
    #         penalty = missed_profit_percent * 0.5  # 较小的惩罚
    #     elif 0.7 <= missed_profit_percent < 1.5:
    #         penalty = missed_profit_percent  # 正常惩罚
    #     elif 1.5 <= missed_profit_percent < 2:
    #         penalty = missed_profit_percent * 1.5  # 较重的惩罚
    #     elif 2 <= missed_profit_percent < 3:
    #         penalty = missed_profit_percent * 2  # 加重惩罚
    #     elif 3 <= missed_profit_percent < 4:
    #         penalty = missed_profit_percent * 2.5  # 更高的惩罚
    #     elif 4 <= missed_profit_percent <= 5:
    #         penalty = missed_profit_percent * 3  # 最大惩罚
    #     else:
    #         penalty = missed_profit_percent * 3  # 对于超过5%的错过利润，维持最大惩罚
    #     return penalty
    
#     def _calculate_missed_profit_after_exit(self, exit_price, reason):
#         """
#         计算在平仓后错过的潜在利润。
#         对于 TP，计算价格继续上涨的部分。
#         对于 SL，计算价格反弹的部分。
#         """
#         future_window = self.price_data.iloc[self.current_step:self.current_step + self.missed_window]
#         if future_window.empty:
#             return 0

#         if reason == 'TP':
#             # 对于达到 TP 的情况，计算未来最高价
#             future_high = future_window['High'].max()
#             potential_profit = future_high - exit_price
#         elif reason == 'SL':
#             # 对于达到 SL 的情况，计算未来最高价
#             future_high = future_window['High'].max()
#             potential_profit = future_high - exit_price
#         else:  # 'Manual' 平仓
#             future_high = future_window['High'].max()
#             potential_profit = future_high - exit_price

#         potential_profit_percent = potential_profit / self.entry_price * 100
#         potential_profit_percent -= self.transaction_cost * 100  # 考虑交易成本
#         return max(potential_profit_percent, 0)  # 确保不为负值

#     def _calculate_reward(self, profit_percent, exit_reason, missed_profit_percent):
#         """
#         根据平仓原因计算奖励。

#         参数：
#         - profit_percent：实际的利润百分比。
#         - exit_reason：平仓原因（'TP'、'SL'、'Manual'）。
#         - missed_profit_percent：错过的潜在利润百分比。

#         返回：
#         - reward：计算后的奖励值。
#         """
#         reward = 0

#         if exit_reason == 'TP':
#             # 如果是达到止盈平仓，奖励等于实际利润百分比
#             reward += profit_percent
#         elif exit_reason == 'SL':
#             # 如果是触发止损平仓，给予较大的惩罚
#             reward += profit_percent  # 通常 profit_percent 为负值
#             # 可以额外给予惩罚，例如：
#             reward -= 5  # 额外的固定惩罚
#         elif exit_reason == 'Manual':
#             # 如果是手动平仓，奖励等于实际利润百分比
#             reward += profit_percent
#             # 可以根据需要添加额外的奖励或惩罚

#         # 减去错过的潜在利润作为惩罚
#         if missed_profit_percent > 0:
#             penalty = self._calculate_missed_profit_penalty(missed_profit_percent)
#             reward -= penalty  # 惩罚是错过的潜在利润

#         return reward

#     def _evaluate_order_limit(self):
#         """
#         评估挂单操作的合理性。
#         如果未来窗口内的潜在利润未达到 0.5%，给予加重的惩罚。
#         """
#         future_window = self.price_data.iloc[self.current_step:self.current_step + self.missed_window]
#         if future_window.empty:
#             return -5  # 如果未来数据不足，给予固定惩罚

#         future_high = future_window['High'].max()
#         potential_profit = (future_high - self.entry_price) / self.entry_price * 100  # 计算潜在利润百分比

#         if potential_profit < 0.5:
#             # 潜在利润未达到 0.5%，给予加重的惩罚
#             penalty = -5  # 您可以根据需要调整惩罚值
#         else:
#             # 潜在利润达到或超过 0.5%，给予适当奖励
#             penalty = 0  # 或者给予正奖励，例如 potential_profit * 0.1

#         return penalty
    
    # def _evaluate_create_order_limit_decision(self):
    #     """
    #     评估创建挂单的决策。
    #     """
    #     future_window = self.price_data.iloc[self.current_step:self.current_step + self.missed_window]
    #     if future_window.empty:
    #         return 0  # 没有未来数据，不给予惩罚

    #     future_high = future_window['High'].max()
    #     potential_profit = (future_high - self.price_data.iloc[self.current_step]['Close']) / self.price_data.iloc[self.current_step]['Close'] * 100
        
    #     if potential_profit < 0.6:
    #         penalty = -potential_profit  # 较小profit, 直接給予懲罰
    #     elif 0.6 <= potential_profit < 1.5:
    #         penalty = potential_profit  # 正常獎勵
    #     elif 1.5 <= potential_profit < 2:
    #         penalty = potential_profit * 1.5  # 较重的獎勵
    #     elif 2 <= potential_profit < 3:
    #         penalty = potential_profit * 2  # 加重獎勵
    #     elif 3 <= potential_profit < 4:
    #         penalty = potential_profit * 2.5  # 更高的獎勵
    #     elif 4 <= potential_profit <= 5:
    #         penalty = potential_profit * 3  # 最大獎勵
    #     else:
    #         penalty = potential_profit * 3  # 对于超过5%的错过利润，维持最大獎勵

    #     return penalty

    # def caculate_entry_price_reward(self, action, is_long_position, current_price, entry_price, min_future_low, max_future_high, tp_price, sl_price, final_price):
    #     entry_price_to_final_profit = (final_price - entry_price) / entry_price
    #     entry_price_to_tp_price_distance = ((tp_price - entry_price) / entry_price)
    #     distance_from_min_to_entry = ((entry_price - min_future_low) / entry_price)
    #     min_low_to_final_price = (final_price - min_future_low) / min_future_low
    #     distance_from_min_future_low_to_tp = (tp_price - min_future_low) / min_future_low
    #     current_price_to_min_future_low = (min_future_low - current_price) / current_price
        
    #     reward = 0

    #     if is_long_position:
    #         if min_future_low <= sl_price:
    #             actual_profit = -self.sl - self.fee
    #             entry_price_to_tp_price_distance
    #             if ((min_future_low - current_price) / current_price) <= self.lowest_entry_price_percentage:
    #                 reward = 0
    #                 self.must_loss_trades += 1
    #             else:
    #                 # 檢查未來是否觸發 TP
    #                 if max_future_high >= tp_price:
    #                     reward = actual_profit - abs(entry_price_to_tp_price_distance) - abs(distance_from_min_to_entry) * 5
    #                     self.can_tp_but_loss_price_not_good_trades += 1
    #                 elif min_low_to_final_price >= self.at_least_profit:
    #                     # 正常情況下，根據距離止損價格的遠近進行懲罰
    #                     reward = actual_profit - abs(entry_price_to_tp_price_distance) - abs(distance_from_min_to_entry) * 3
    #                     self.can_at_least_profit_but_loss_price_not_good_trades += 1
    #                 elif min_low_to_final_price >= 0:
    #                     reward = actual_profit - abs(entry_price_to_tp_price_distance) - abs(distance_from_min_to_entry) * 2.5
    #                     self.can_profit_but_loss_price_not_good_trades += 1
    #                 else:
    #                     # 盡量輸少
    #                     if abs(distance_from_min_to_entry) <= 0.002:
    #                         reward = 0
    #                     else:
    #                         reward = actual_profit - abs(entry_price_to_tp_price_distance) - abs(distance_from_min_to_entry)
    #                     self.hold_to_final_price_must_loss_trades += 1
    #         # 如果觸發止盈
    #         elif max_future_high >= tp_price:
    #             # 使用 entry_price 來計算實際的盈利
    #             actual_profit = entry_price_to_tp_price_distance - self.fee
                
    #             if current_price * (1 - self.sl) >= min_future_low:
    #                 # 當距離最低點非常小時（例如 <= 0.1%），增強獎勵
    #                 if abs(distance_from_min_to_entry) <= 0.001:  # 0.1% 表示的百分比距離
    #                     reward = actual_profit * 20.22 - abs(distance_from_min_to_entry)
    #                 elif abs(distance_from_min_to_entry) <= 0.002:  # 0.2% 表示的百分比距離
    #                     reward = actual_profit * 10.11 - abs(distance_from_min_to_entry) * 1.5
    #                 elif abs(distance_from_min_to_entry) <= 0.003:  # 0.3% 表示的百分比距離
    #                     reward = actual_profit * 5.05 - abs(distance_from_min_to_entry) * 2
    #                 else:
    #                     reward = actual_profit * 3.37 - abs(distance_from_min_to_entry) * 2.5
    #                 self.take_tp_but_avoid_loss_trades += 1
    #             else:
    #                 # 根據盈利給予獎勵
    #                 if abs(distance_from_min_to_entry) <= 0.001:  # 當距離最低點小於 0.1%
    #                     reward = actual_profit * 13.5 - abs(distance_from_min_to_entry)
    #                 elif abs(distance_from_min_to_entry) <= 0.002:  # 當距離最低點小於 0.2%
    #                     reward = actual_profit * 6.75 - abs(distance_from_min_to_entry) * 1.5
    #                 elif abs(distance_from_min_to_entry) <= 0.003:  # 當距離最低點小於 0.3%
    #                     reward = actual_profit * 3.37 - abs(distance_from_min_to_entry) * 2
    #                 else:
    #                     reward = actual_profit * 2.25 - abs(distance_from_min_to_entry) * 2.5
    #                 self.take_tp_trades += 1
    #         else:
    #             # 計算未觸發止損或止盈的情況下的結果
    #             actual_profit = entry_price_to_final_profit - self.fee
    #             if entry_price_to_final_profit >= 0:
    #                 if current_price * (1 - self.sl) >= min_future_low:
    #                     if abs(distance_from_min_to_entry) <= 0.001:  # 當距離最低點小於 0.1%
    #                         reward = actual_profit * 9 - abs(distance_from_min_to_entry)
    #                     elif abs(distance_from_min_to_entry) <= 0.002:  # 當距離最低點小於 0.2%
    #                         reward = actual_profit * 4.5 - abs(distance_from_min_to_entry) * 1.5
    #                     elif abs(distance_from_min_to_entry) <= 0.003:  # 當距離最低點小於 0.3%
    #                         reward = actual_profit * 2.25 - abs(distance_from_min_to_entry) * 2
    #                     else:
    #                         reward = actual_profit * 1.5 - abs(distance_from_min_to_entry) * 2.5
    #                 else:
    #                     if abs(distance_from_min_to_entry) <= 0.001:  # 當距離最低點小於 0.1%
    #                         reward = actual_profit * 6 - abs(distance_from_min_to_entry)
    #                     elif abs(distance_from_min_to_entry) <= 0.002:  # 當距離最低點小於 0.2%
    #                         reward = actual_profit * 3 - abs(distance_from_min_to_entry) * 1.5
    #                     elif abs(distance_from_min_to_entry) <= 0.003:  # 當距離最低點小於 0.3%
    #                         reward = actual_profit * 1.5 - abs(distance_from_min_to_entry) * 2
    #                     else:
    #                         reward = actual_profit - abs(distance_from_min_to_entry) * 2.5
    #             else:
    #                 # 當虧損時
    #                 if abs(distance_from_min_to_entry) <= 0.002:
    #                     reward = 0
    #                 else:
    #                     reward = (actual_profit - abs(distance_from_min_to_entry)) * 0.5
    #             self.other_trades += 1
    #     else:
    #         # 距離最低價格的距離 - 少賺 - 現在價格到最低價格的距離
    #         if max_future_high >= tp_price:
    #             reward = -abs(distance_from_min_to_entry) * 6 - abs(entry_price_to_tp_price_distance) - abs(current_price_to_min_future_low)
    #             self.missed_tp_trades += 1
    #         elif min_low_to_final_price >= self.at_least_profit:
    #             reward = -abs(distance_from_min_to_entry) * 4 - abs(entry_price_to_final_profit) - abs(current_price_to_min_future_low)
    #             self.missed_at_least_profit_trades += 1
    #         elif min_low_to_final_price >= 0:
    #             reward = -abs(distance_from_min_to_entry) * 1.5 - abs(entry_price_to_final_profit) - abs(current_price_to_min_future_low)
    #             self.missed_profit_trade += 1
    #         else:
    #             reward = -abs(distance_from_min_to_entry) - abs(entry_price_to_final_profit) - abs(current_price_to_min_future_low)
    #             self.missed_hold_to_final_price_trade += 1

    #     return reward
    
    # def render(self):
    #     pass
    
    # def print_summary(self):
    #     # 打印總結結果
    #     print(f"MUST LOSS TRADES: {self.must_loss_trades}")
    #     print(f"CAN TP BUT LOSS PRICE NOT GOOD TRADES: {self.can_tp_but_loss_price_not_good_trades}")
    #     print(f"CAN AT LEAST PROFIT BUT LOSS PRICE NOT GOOD TRADES: {self.can_at_least_profit_but_loss_price_not_good_trades}")
    #     print(f"CAN PROFIT BUT LOSS PRICE NOT GOOD TRADES: {self.can_profit_but_loss_price_not_good_trades}")
    #     print(f"HOLD TO FINAL PRICE MUST LOSS TRADES: {self.hold_to_final_price_must_loss_trades}")
    #     print(f"TAKE TP BUT AVOID LOSS TRADES: {self.take_tp_but_avoid_loss_trades}")
    #     print(f"TAKE TP TRADES: {self.take_tp_trades}")
    #     print(f"MISSED TP TRADES: {self.missed_tp_trades}")
    #     print(f"MISSED AT LEAST PROFIT TRADES: {self.missed_at_least_profit_trades}")
    #     print(f"MISSED PROFIT TRADE: {self.missed_profit_trade}")
    #     print(f"MISSED HOLD TO FINAL PRICE TRADE: {self.missed_hold_to_final_price_trade}")
    #     print(f"OTHER TRADES: {self.other_trades}")
    #     print("-----------------------------------")

class TradingEnv(gym.Env):
    def __init__(self, input_data, balance=100000, lookback_window=20, missed_window=12):
        super(TradingEnv, self).__init__()
        
        self.input_data = input_data.reset_index(drop=True)  # 确保索引连续，从0开始
        self.lookback_window = lookback_window
        self.current_step = self.lookback_window
        self.balance = balance  # 初始资金
        self.order_amount = 1000  # 每次交易的下单金额
        self.total_steps = len(self.input_data) - self.lookback_window  # 总时间步数

        # 持仓信息
        self.position = 0  # 0：无持仓，1：持有多头
        self.entry_price = 0

        # 记录列表
        self.balance_history = []      # 用于记录资金余额的列表
        self.total_profit_percent = [] # 用于记录每次开单的利润百分比
        self.totla_profit = 0

        self.transaction_cost = 0.0005  # 手续费比例，例如万分之五
        self.missed_window = missed_window  # 用于计算错过的潜在利润的窗口大小

        self.max_profit_percent = 0.02

        # 动作空间设置
        # 动作空间：决策
        # decision: 0-不操作，1-开单
        # 定义 TP_price 的取值范围
        tp_levels = np.arange(0.006, self.max_profit_percent + 0.001, 0.001)  # 从 0.3% 到 3%，步长为 0.1%
        num_tp_levels = len(tp_levels)
        self.action_space = MultiDiscrete([2, num_tp_levels])

        # action_low = np.array([0.0, self.max_profit_percent / 10])  # decision 的下限为 0.0
        # action_high = np.array([1.0, self.max_profit_percent])  # decision 的上限为 1.0
        # self.action_space = gym.spaces.Box(
        #     low=action_low,
        #     high=action_high,
        #     dtype=np.float32
        # )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.lookback_window, len(self.input_data.columns)), dtype=np.float32)
        
        # 新增统计计数器
        self.total_opportunities = 0     # 实际超过最低利润（0.6%）的机会总数
        self.correct_trades = 0          # 开单且未来窗口内利润确实超过 0.6% 的次数（命中）
        self.missed_opportunities = 0    # 错过的超过最低利润（0.6%）的机会次数
        self.total_trades = 0            # 开单总次数
        self.rewards = []
        
    def reset(self):
        self.current_step = self.lookback_window
        self.balance = self.balance  # 重置账户余额
        self.position = 0
        self.entry_price = 0
        self.balance_history = []
        self.total_profit_percent = []
        self.totla_profit = 0
        self.rewards = []
        # 重置统计计数器
        self.total_opportunities = 0
        self.correct_trades = 0
        self.missed_opportunities = 0
        self.total_trades = 0
        return self._next_observation()
    
    def _next_observation(self):
        # 获取回溯窗口内的价格数据
        obs = self.input_data.iloc[self.current_step - self.lookback_window:self.current_step]
        # 将 DataFrame 转换为 numpy 数组
        obs = obs.values
        return obs
    
    def step(self, action):
        decision = action[0]  # 0：不操作，1：开单
        tp_price_offset = action[1]  # 从 Box 中取出 TP_price 偏移量
        tp_price_offset_percent = tp_price_offset * 100  # 将 TP_price 偏移量转换为百分比
        current_price = self.input_data.iloc[self.current_step]['Close']  # 当前收盘价
        tp_price = current_price * (1 + tp_price_offset)  # 计算止盈价格

        reward = 0
        done = False

        potential_max_profit = self.calculate_future_potential_max_profit()
        # 检查是否有实际超过最低利润的机会
        if potential_max_profit >= 0.6:
            self.total_opportunities += 1  # 增加实际机会计数

        
        # 计算奖励或惩罚
        TP_price_reward = self._evaluate_TP_reward(tp_price, tp_price_offset * 100)
        decision_reward = self._evaluate_trade_decision_reward(decision) # 大於0.6%才值得開單



        reward = decision_reward + TP_price_reward
        if decision == 1:  # 开单
            self.total_trades += 1  # 增加开单次数

            # 检查是否命中
            if potential_max_profit >= 0.6:
                self.correct_trades += 1  # 增加命中次数

            # 想辦法讓decision在有機會的時候才開單，並且獲取最大的利潤
            if potential_max_profit >= tp_price_offset_percent:
                self.totla_profit += tp_price_offset_percent - 0.1
            else:
                self.totla_profit += self.calculate_future_final_return() - 0.1
                
                # self.totla_profit += (max(-0.6, self.calculate_future_final_return()) - 0.1)
                
        else:
            if potential_max_profit >= 0.6:
                self.missed_opportunities += 1  # 增加错过的机会计数



        # 在每个时间步记录余额
        self.balance_history.append(self.balance)

        self.current_step += 1
        if self.current_step >= len(self.input_data):
            done = True

        obs = self._next_observation()
        self.rewards.append(reward)

        self.total_profit_percent.append(self.totla_profit)
        return obs, reward, done, {}
    
    def _evaluate_trade_decision_reward(self, decision):
        potential_max_profit = self.calculate_future_potential_max_profit()
        potential_max_loss = self.calculate_future_potential_max_loss()
        
        if decision == 1:  # 模型选择开仓
            if potential_max_profit >= 0.6:
                reward = potential_max_profit
                # reward = 1
            elif potential_max_loss >= 1.5:
                reward = potential_max_profit * 1.5
            else:
                reward = -abs(potential_max_loss)  # 适度减少惩罚力度
                # reward = -1
        else:
            if potential_max_profit >= 0.6:
                reward = -potential_max_profit  # 增加错过机会的惩罚
                # reward = -1
            else:
                reward = abs(potential_max_loss)
                # reward = 1            
        return reward
    
    def _evaluate_TP_reward(self, TP_price, tp_price_offset):
        reward = 0
        current_price = self.input_data.iloc[self.current_step]['Close']
        potential_max_profit = self.calculate_future_potential_max_profit()  # 以百分比表示

        # 计算 TP_price 对应的利润百分比
        tp_profit_percentage = (TP_price - current_price) / current_price * 100

        # 计算 TP_price 与 potential_max_profit 的差值（百分比）
        profit_difference = tp_profit_percentage - potential_max_profit

        if potential_max_profit <= 0.6:
            # 如果未来没有潜在利润，任何 TP_price 都没有意义，给予适当的惩罚
            reward = -1 * 0.5  # 或者您认为合适的惩罚值
        else:
            if profit_difference > 0:
                # # TP_price 超过 potential_max_profit，给予惩罚
                # if profit_difference > 0.5:
                #     penalty = profit_difference * 5
                # elif profit_difference > 0.2:
                #     penalty = profit_difference * 3
                # else:
                #     penalty = profit_difference * 2
                # reward = -penalty

                # TP_price 超过 potential_max_profit，给予惩罚
                penalty = profit_difference * 2
                reward = -penalty
            else:
                # TP_price 在 potential_max_profit 范围内，给予奖励
                profit_ratio = tp_profit_percentage / potential_max_profit  # 获取潜在利润的比例

                if tp_price_offset >= (self.max_profit_percent * 100) or profit_ratio >= 0.8:
                    reward = 1 * 3 # 代表已經全拿了
                # elif profit_ratio >= 0.7:
                #     reward = profit_ratio * 8
                # elif profit_ratio >= 0.5:
                #     reward = profit_ratio * 3
                else:
                    reward = profit_ratio
        
        return reward

    
    def calculate_future_potential_max_profit(self):
        future_window = self.input_data.iloc[self.current_step:self.current_step + self.missed_window]
        if future_window.empty:
            return 0  # 没有未来数据

        future_high = future_window['High'].max()
        potential_profit = (future_high - self.input_data.iloc[self.current_step]['Close']) / self.input_data.iloc[self.current_step]['Close'] * 100

        return potential_profit
    
    def calculate_future_potential_max_loss(self):
        future_window = self.input_data.iloc[self.current_step:self.current_step + self.missed_window]
        if future_window.empty:
            return 0  # 没有未来数据

        future_low = future_window['Low'].min()
        current_price = self.input_data.iloc[self.current_step]['Close']
        potential_loss = (future_low - current_price) / current_price * 100  # 结果为负值
        return potential_loss  # 返回负值表示亏损
    
    def calculate_future_final_return(self):
        future_window = self.input_data.iloc[self.current_step:self.current_step + self.missed_window]
        if future_window.empty:
            return 0  # 没有未来数据

        final_close = future_window['Close'].iloc[-1]
        current_price = self.input_data.iloc[self.current_step]['Close']
        final_return = (final_close - current_price) / current_price * 100
        return final_return

    def summary(self):
        # 模拟结束后，计算混淆矩阵和指标
        TP = self.correct_trades  # 模型正确开单次数
        FN = self.missed_opportunities  # 模型错过的机会次数
        FP = self.total_trades - self.correct_trades  # 模型错误开单次数
        TN = self.total_steps - self.total_trades - self.missed_opportunities  # 模型正确避开的次数
        # 计算 Precision、Recall 和 F1-score
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        mean_reward, variance_reward = self.calculate_mean_and_variance(self.rewards)
        reward_to_risk = self.calculate_reward_to_risk_ratio(self.rewards)
        positive_negative_ratio = self.calculate_positive_negative_ratio(self.rewards)
        reward_skewness = self.calculate_reward_skewness(self.rewards)

        print(f"平均獎勵: {mean_reward}")
        print(f"獎勵方差: {variance_reward}")
        print(f"獎勵-風險比: {reward_to_risk}")
        print(f"正負獎勵比: {positive_negative_ratio}")
        print(f"獎勵偏度: {reward_skewness}")

        print(f"Precision（精确率）：{precision:.4f}")
        print(f"Recall（召回率）：{recall:.4f}")
        print(f"F1-score：{f1_score:.4f}")
        # 打印总结结果
        print(f"Total trades: {self.total_trades}")
        print(f"Total opportunities: {self.total_opportunities}")
        print(f"Correct trades: {self.correct_trades}")
        print(f"Missed opportunities: {self.missed_opportunities}")
        print("-----------------------------------")
        print(f"Total profit: {self.totla_profit}")

    def calculate_mean_and_variance(self, rewards):
        mean_reward = np.mean(rewards)
        variance_reward = np.var(rewards)
        return mean_reward, variance_reward

    def calculate_reward_to_risk_ratio(self, rewards):
        mean_reward = np.mean(rewards)
        std_dev_reward = np.std(rewards)
        reward_to_risk_ratio = mean_reward / std_dev_reward
        return reward_to_risk_ratio

    def calculate_positive_negative_ratio(self, rewards):
        positive_rewards = [r for r in rewards if r > 0]
        negative_rewards = [r for r in rewards if r < 0]
        
        if len(negative_rewards) == 0:
            return float('inf')  # 如果沒有負獎勵，返回無窮大
        else:
            return np.sum(positive_rewards) / np.abs(np.sum(negative_rewards))
        
    def calculate_reward_skewness(self, rewards):
        return skew(rewards)

    def clip_rewards(rewards, min_reward, max_reward):
        return np.clip(rewards, min_reward, max_reward)

def customized_specific_period_col(df):
    customized_cols_infos = []
    
    # 紀錄原先的cols
    original_cols = set(df.columns)
    price_look_back = 672
    
    # indicator_look_back = 288
    
    # df, lower_low_higher_high_info = indicators.add_lower_low_higher_high(df, 0.04, look_back=indicator_look_back, is_need_return_function_info=True)
    # customized_cols_infos.append(lower_low_higher_high_info)
    
    # df, price_indicator_info = indicators.add_price_indicator(df, look_back=price_look_back, is_need_return_function_info=True)
    # customized_cols_infos.append(price_indicator_info)

    # df, look_back_24h_min_max_price_info = indicators.add_24h_min_max_price(df, look_back=96, is_need_return_function_info=True)
    # customized_cols_infos.append(look_back_24h_min_max_price_info)

    df = calculate_volatility(df, 20)
    di_len = 14
    adx_len = 14
    df = calculate_adx(df, di_len, adx_len)
    df = calculate_moving_averages(df)
    df = calculate_bollinger_bands(df)
    df = calculate_momentum(df)
    df = calculate_ppo(df)
    # df = calculate_trend(df, 288, 0.04)
    df = indicators.trend_min_max_price_indicator(df, 192)
    df = indicators.calculate_candelstick_patterns(df)
    df = calculate_VWAP(df, window=20)

    # # 對數收益率
    # df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    # df['Volatility'] = df['Log_Returns'].rolling(window=20).std()
    
    # df['SMA_5'] = indicators.calculate_sma(df['Close'], 5)
    # df['SMA_10'] = indicators.calculate_sma(df['Close'], 10)

    df['EMA_7'] = indicators.calculate_ema(df['Close'], 7)
    df['EMA_25'] = indicators.calculate_ema(df['Close'], 25)
    df['EMA_99'] = indicators.calculate_ema(df['Close'], 99)

    
    # 紀錄新的cols
    modified_cols = set(df.columns)
    # 篩選多出來的cols
    new_cols = list(modified_cols - original_cols)
    return df, customized_cols_infos, new_cols

def calculate_VWAP(df, window=20):
    df['Typical_Price'] = (df['Close'] + df['High'] + df['Low']) / 3
    # df['Typical_Price'] = df['Close']
    df['VP'] = df['Typical_Price'] * df['Volume']

    # df['Cumulative_VP'] = df['VP'].cumsum()
    # df['Cumulative_Volume'] = df['Volume'].cumsum()
    # 使用滚动窗口计算 VP 和 Volume 的累积和
    df['Cumulative_VP'] = df['VP'].rolling(window=window).sum()
    df['Cumulative_Volume'] = df['Volume'].rolling(window=window).sum()

    df['VWAP'] = df['Cumulative_VP'] / df['Cumulative_Volume']

    return df


def calculate_dm(df):
    df['up'] = df['High'] - df['High'].shift(1)
    df['down'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0)
    df['-DM'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0)
    return df

def calculate_rma(series, period):
    rma = series.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return rma

def calculate_di(df, period):
    df['TR_sum'] = calculate_rma(df['TR'], period)
    df['+DM_sum'] = calculate_rma(df['+DM'], period)
    df['-DM_sum'] = calculate_rma(df['-DM'], period)
    df['+DI'] = 100 * (df['+DM_sum'] / df['TR_sum'])
    df['-DI'] = 100 * (df['-DM_sum'] / df['TR_sum'])
    return df

def calculate_dx(df):
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    return df

def calculate_adx(df, di_len, adx_len):
    df = calculate_dm(df)
    df = calculate_di(df, di_len)
    df = calculate_dx(df)
    df['ADX'] = calculate_rma(df['DX'], adx_len)
    return df

def calculate_volatility(df, look_back):
    df['Close_Volatility'] = df['Close'].rolling(window=look_back).std()
    df['High_Volatility'] = df['High'].rolling(window=look_back).std()
    df['Low_Volatility'] = df['Low'].rolling(window=look_back).std()
    df['Open_Volatility'] = df['Open'].rolling(window=look_back).std()
    return df

def calculate_moving_averages(df, short_window=50, long_window=200):
    df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_window).mean()
    return df

def calculate_bollinger_bands(df):
    df['BB_Width'] = df['Upper Band'] - df['Lower Band']
    df['BB_Pos'] = (df['Close'] - df['Lower Band']) / (df['Upper Band'] - df['Lower Band'])
    return df

def calculate_momentum(df, window=10):
    df['Momentum'] = df['Close'].diff(window)
    return df

def calculate_ppo(df, short_window=12, long_window=26):
    short_ema = df['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['PPO'] = (short_ema - long_ema) / long_ema * 100
    return df

# 假设df是你的数据框，包含所有需要的列
def calculate_rolling_stats(df, window=20):
    rolling_mean = df.rolling(window=window).mean()
    rolling_std = df.rolling(window=window).std()
    return rolling_mean, rolling_std

def set_y_label(df, lookahead=8, percentage=0.05):
    # set_y_label
    
    for i in range(len(df) - lookahead):
        current_close = df['Close'].iloc[i]
        future_window = df['Close'].iloc[i+1:i+lookahead+1]

        future_high_window = df['High'].iloc[i+1:i+lookahead+1]
        future_low_window = df['Low'].iloc[i+1:i+lookahead+1]
        
        # max_future_price = future_window.max()
        # min_future_price = future_window.min()

        max_future_price = future_high_window.max()
        min_future_price = future_low_window.min()

        up_price = current_close * (1 + percentage)
        down_price = current_close * (1 - percentage)
        
        if max_future_price >= up_price and min_future_price <= down_price:
            df.at[i, 'y'] = 3  # 未来同时出现上涨和下跌5%的可能性
        elif max_future_price >= up_price:
            df.at[i, 'y'] = 1  # 未来有上涨5%的可能性
        elif min_future_price <= down_price:
            df.at[i, 'y'] = 2  # 未来有下跌5%的可能性
        else:
            df.at[i, 'y'] = 0  # 未来没有明显的上涨或下跌

        df.at[i, 'up_percentage'] = up_price
        df.at[i, 'down_percentage'] = down_price
    df['lookahead'] = lookahead
    # df['up_percentage'] = percentage
    # df['down_percentage'] = -percentage
    return df

def set_y_only_up_label(df, lookahead=8, percentage=0.05):
    for i in range(len(df) - lookahead):
        current_close = df['Close'].iloc[i]
        future_high_window = df['High'].iloc[i+1:i+lookahead+1]

        max_future_price = future_high_window.max()

        up_price = current_close * (1 + percentage)
        
        if max_future_price >= up_price:
            df.at[i, 'y'] = 1  # 未来有上涨5%的可能性
        else:
            df.at[i, 'y'] = 0  # 未来没有明显的上涨或下跌

        df.at[i, 'up_percentage'] = up_price
    df['lookahead'] = lookahead
    # df['up_percentage'] = percentage
    # df['down_percentage'] = -percentage
    return df


def set_y_label_sequence(df, lookahead=8, percentage=0.05):
    for i in range(len(df) - lookahead):
        current_close = df['Close'].iloc[i]
        up_price = current_close * (1 + percentage)
        down_price = current_close * (1 - percentage)

        y_label = 0  # 默认为没有明显的上涨或下跌

        for j in range(1, lookahead + 1):
            future_high = df['High'].iloc[i + j]
            future_low = df['Low'].iloc[i + j]

            if future_high >= up_price and future_low <= down_price:
                # 如果同时达到up_price和down_price，判断谁先到达
                y_label = 3
                break
            elif future_high >= up_price:
                y_label = 1  # 先上涨
                break
            elif future_low <= down_price:
                y_label = 2  # 先下跌
                break

        df.at[i, 'y'] = y_label
        df.at[i, 'up_percentage'] = up_price
        df.at[i, 'down_percentage'] = down_price

    df['lookahead'] = lookahead
    return df

def set_y_win_lose_label_sequence(df, lookahead=8, tp_percentage=0.008, sl_percentage=0.004):
    for i in range(len(df) - lookahead):
        current_close = df['Close'].iloc[i]
        
        # 计算多头的止盈和止损价格
        long_tp_price = current_close * (1 + tp_percentage)
        long_sl_price = current_close * (1 - sl_percentage)
        
        # 计算空头的止盈和止损价格
        short_tp_price = current_close * (1 - tp_percentage)
        short_sl_price = current_close * (1 + sl_percentage)

        y_label = 0  # 默认为没有达到止盈或止损的情况
        long_hit = False  # 标记多头是否触发止盈或止损
        short_hit = False  # 标记空头是否触发止盈或止损

        for j in range(1, lookahead + 1):
            future_high = df['High'].iloc[i + j]
            future_low = df['Low'].iloc[i + j]

            # 检查多头是否触发止盈或止损
            if not long_hit:
                if future_low <= long_sl_price:
                    y_label = 3  # 多头止损（long lose）
                    long_hit = True  # 多头已经处理
                elif future_high >= long_tp_price:
                    y_label = 1  # 多头获胜（long win）
                    long_hit = True  # 多头已经处理
                    break
                

            # 检查空头是否触发止盈或止损
            if not short_hit:
                if future_high >= short_sl_price:
                    y_label = 4  # 空头止损（short lose）
                    short_hit = True  # 空头已经处理
                elif future_low <= short_tp_price:
                    y_label = 2  # 空头获胜（short win）
                    short_hit = True  # 空头已经处理
                    break
                
            # 如果多头和空头都触发了止损，则标记为0
            if long_hit and short_hit:
                y_label = 0  # 两边都碰到止损，标记为0
                break

        # 标记止盈和止损价格
        df.at[i, 'y'] = y_label
        df.at[i, 'long_tp_price'] = long_tp_price
        df.at[i, 'long_sl_price'] = long_sl_price
        df.at[i, 'short_tp_price'] = short_tp_price
        df.at[i, 'short_sl_price'] = short_sl_price

    df['lookahead'] = lookahead
    return df

def set_y_multiple_label(df, min_lookahead = 1, max_lookahead=8, percentage=0.05):
    def calculate_y_label(i):
        current_close = df['Close'].iloc[i]
        labels = []

        for lookahead in range(min_lookahead, max_lookahead + 1):
            future_window = df['Close'].iloc[i+1:i+lookahead+1]

            future_high_window = df['High'].iloc[i+1:i+lookahead+1]
            future_low_window = df['Low'].iloc[i+1:i+lookahead+1]

            max_future_price = future_window.max()
            min_future_price = future_window.min()

            up_price = current_close * (1 + percentage)
            down_price = current_close * (1 - percentage)

            if max_future_price >= up_price and min_future_price <= down_price:
                labels.append(3)
            elif max_future_price >= up_price:
                labels.append(1)
            elif min_future_price <= down_price:
                labels.append(2)
            else:
                labels.append(0)

        print('i:', i)
        return labels

    # 计算所有的标签
    df['y'] = df.index.to_series().apply(calculate_y_label)

    # 展开 'y' 列为单独的行
    df = df.explode('y')
    df['lookahead'] = df.groupby(df.index).cumcount() + 1

    # 只保留完整的 lookahead 数据
    df = df[df['lookahead'] + df.index < len(df)]
    
    df.reset_index(drop=True, inplace=True)

    return df

def backtesting(input_data, lookahead, percentage, prob, fee=0.0012):
    total_profit_percentage = 0  # 初始化总利润
    trades = []  # 记录每笔交易的利润

    # 遍历每一行数据，进行回测
    for i in range(len(input_data) - lookahead):
        entry_price = input_data['Close'].iloc[i]  # 进场价格
        predicted_class = input_data['predicted_class'].iloc[i]  # 预测类别
        prob_class_1 = input_data['prob_class_1'].iloc[i]  # target 1（多头）的预测概率
        prob_class_2 = input_data['prob_class_2'].iloc[i]  # target 2（空头）的预测概率
        final_close_price = input_data['Close'].iloc[i + lookahead]  # 使用lookahead之后的下一个收盘价

        # 进场多头交易
        if predicted_class == 1 and prob_class_1 >= prob:
            future_high = input_data['High'].iloc[i+1:i+lookahead+1].max()  # 未来的最高价
            future_low = input_data['Low'].iloc[i+1:i+lookahead+1].min()  # 未来的最低价
            tp_price = input_data['long_tp_price'].iloc[i]  # 多头止盈目标
            sl_price = input_data['long_sl_price'].iloc[i]  # 多头止损目标
            # tp_price = entry_price * (1 + percentage)  # 多头止盈目标
            # sl_price = entry_price * (1 - percentage)  # 多头止损目标

            if future_low <= sl_price:
                # 计算止损时的亏损百分比
                profit_percentage = (sl_price - entry_price) / entry_price - fee
            elif future_high >= tp_price:
                # 计算止盈时的收益百分比
                profit_percentage = (tp_price - entry_price) / entry_price - fee
            else:
                # 如果没有达到止盈或止损，则使用 entry 与最后一个 close 的差额来计算利润
                profit_percentage = (final_close_price - entry_price) / entry_price - fee


            total_profit_percentage += profit_percentage
            trades.append(profit_percentage)

        # 进场空头交易
        elif predicted_class == 2 and prob_class_2 >= prob:
            future_high = input_data['High'].iloc[i+1:i+lookahead+1].max()  # 未来的最高价
            future_low = input_data['Low'].iloc[i+1:i+lookahead+1].min()  # 未来的最低价
            tp_price = input_data['short_tp_price'].iloc[i]  # 多头止盈目标
            sl_price = input_data['short_sl_price'].iloc[i]  # 多头止损目标
            # tp_price = entry_price * (1 - percentage)  # 空头止盈目标
            # sl_price = entry_price * (1 + percentage)  # 空头止损目标

            if future_high >= sl_price:
                # 计算空头止损时的亏损百分比
                profit_percentage = (entry_price - sl_price) / entry_price - fee
            elif future_low <= tp_price:
                # 计算空头止盈时的收益百分比
                profit_percentage = (entry_price - tp_price) / entry_price - fee
            else:
                # 如果没有达到止盈或止损，则使用 entry 与最后一个 close 的差额来计算利润
                profit_percentage = (entry_price - final_close_price) / entry_price - fee

            total_profit_percentage += profit_percentage
            trades.append(profit_percentage)

    # 返回总利润和每笔交易的记录
    return total_profit_percentage, trades



def validation(test_data, model, preprocessor, close_scaler, cols, lookahead, percentage, price_related_features, robust_features, prob_threshold):
    # 预测并存储结果
    input_data = test_data.iloc[:len(test_data) - lookahead].copy()
    input_data = input_data[cols]
    input_data.reset_index(drop=True, inplace=True)

    input_data = set_y_only_up_label(input_data, lookahead, percentage)
    # input_data = set_y_label_sequence(input_data, lookahead, percentage)
    # input_data = set_y_win_lose_label_sequence(input_data, lookahead, percentage, percentage / 3)
    y = input_data['y']
    input_data.drop('y', axis=1, inplace=True)

    # 先筛选出 Volume 大于 10,000 的数据
    # input_data = input_data[input_data['Volume'] >= 15000]
    
    # preprocessed_data, _, _ = scaler(input_data, price_related_features, robust_features, close_scaler=close_scaler, preprocessor=preprocessor)
    # preprocessed_data = input_data
    preprocessed_data = preprocessor.transform(input_data)
    probs = []
    if model is not None:
        model.set_params(device='cuda')
        probs = model.predict_proba(preprocessed_data)
        predicted_classes = model.predict(preprocessed_data)
        input_data['predicted_class'] = predicted_classes
        # 将每个类别的概率保存到 input_data 中
        for i in range(probs.shape[1]):
            input_data[f'prob_class_{i}'] = probs[:, i]
    rl_data = input_data[:-300]
    # 將 test_data 分成 validation set 和 test set
    # validation_data = rl_data
    validation_data, test_data = train_test_split(rl_data, test_size=0.5, random_state=42, shuffle=False)

    # 訓練強化學習模型
    # policy_kwargs = dict(
    #     net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # 增加神經元數量
    # )
    # policy_kwargs = dict(
    #     net_arch=[dict(pi=[512, 512, 256], vf=[512, 512, 256])]  # 大型網絡，適合複雜的金融環境
    # )
    # rl_model = PPO("MlpPolicy", validation_env, verbose=1)
    # rl_model = PPO("MlpPolicy", validation_env, learning_rate=0.0001, n_steps=4096, gamma=0.999, gae_lambda=0.98, clip_range=0.2, ent_coef=0.01, batch_size=128, policy_kwargs=policy_kwargs, verbose=1)

    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # 大型網絡，適合複雜的金融環境
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    validation_env = TradingEnv(input_data=validation_data, lookback_window=288, missed_window=24)
    # model = PPO('MlpPolicy', validation_env, policy_kwargs=policy_kwargs, gamma=0.999, clip_range=0.2, verbose=1)
    # model = PPO('MlpPolicy', validation_env, policy_kwargs=policy_kwargs, verbose=1, device=device)
    model = PPO('MlpPolicy', validation_env, learning_rate=0.00005, gamma=0.995, n_steps=4096, gae_lambda=0.98, batch_size=256, clip_range=0.2, ent_coef=0.01, policy_kwargs=policy_kwargs, verbose=1, device=device)
    model.learn(total_timesteps=1000000)
    # 测试模型
    obs = validation_env.reset()
    for _ in range(len(validation_data) - validation_env.lookback_window):
        action, _states = model.predict(obs)
        obs, reward, done, info = validation_env.step(action) 
        if done:
            break
    validation_env.summary()
    
    # 測試模型策略並進行回測
    test_env = TradingEnv(input_data=test_data, lookback_window=288, missed_window=24)
    obs = test_env.reset()
    for _ in range(len(test_data) - test_env.lookback_window):
        action, _states = model.predict(obs)
        obs, reward, done, info = test_env.step(action) 
        if done:
            break
    test_env.summary()
    # 繪製每根K線的 total profit 變化摺線圖
    plt.figure(figsize=(10, 6))
    plt.plot(test_env.total_profit_percent, label='Total Profit')
    plt.xlabel('K線步數')
    plt.ylabel('Total Profit')
    plt.title('每根K線的總收益變化')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    input_data['y'] = y

    prob_threshold = prob_threshold
    # 过滤出预测概率大于prob_threshold的行
    high_confidence_data = input_data[(probs.max(axis=1) >= prob_threshold)]

    class_accuracies = []
    class_denominators = []
    class_numerators = []
    # 计算每个类别的准确率
    for class_label in range(probs.shape[1]):
        # 获取当前类别的所有样本
        class_data = high_confidence_data[high_confidence_data['predicted_class'] == class_label]

        # 分母: prob >= prob_threshold 的样本数量
        denominator = len(class_data)

        # 分子: predicted_class == y 且 prob >= prob_threshold 的样本数量
        numerator = (class_data['predicted_class'] == class_data['y']).sum()

        # 计算准确率，考虑分母为0的情况
        if denominator > 0:
            class_accuracy = numerator / denominator
        else:
            class_accuracy = 0

        class_accuracies.append(class_accuracy)
        class_denominators.append(denominator)
        class_numerators.append(numerator)

    total_accuracy = np.mean(class_accuracies)
    total_profit_percentage = 0
    
    # total_profit_percentage, trades = backtesting(input_data, lookahead, percentage, prob_threshold)
    return total_accuracy, class_accuracies, class_denominators, class_numerators, total_profit_percentage
    

def get_test_data():
    # predict目前最新的資料
    end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

    end_time_seconds = end_time / 1000
    end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
    end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
    df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 20000, '', is_need_save_original_data=False, is_need_calculated=True)
    df = modules_data.clean_data(df)
    df, customized_cols_infos, new_cols = customized_specific_period_col(df)
    df = modules_data.clean_data(df)
    # 将 'datetime' 列转换为 datetime 类型
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 将 'datetime' 列转换为 UTC+8
    df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')

    # 如果需要移除时区信息，可以使用 .dt.tz_localize(None)
    df['datetime'] = df['datetime'].dt.tz_localize(None)
    
    drop_front_data_count = 1000
    df = df[drop_front_data_count:]
    df.reset_index(drop=True, inplace=True)

    return df

def scaler(df, price_related_features, robust_features, close_scaler=None, preprocessor=None):
    robust_features = robust_features
    standard_features = []
    minMax_features = []

    # 假设 X 是你的数据集
    # 定义需要以 Close 尺度缩放的特征
    close_feature = ['Close']
    price_related_features = price_related_features

    if close_scaler is None:
        # 初始化 StandardScaler 并仅在 Close 上进行拟合
        close_scaler = StandardScaler()
        X_close_scaled = close_scaler.fit_transform(df[close_feature])
    else:
        X_close_scaled = close_scaler.transform(df[close_feature])

    X_scaled_without_price = df.drop(price_related_features + close_feature, axis=1)
    if preprocessor is None:
        # 列出每个缩放器/转换器对应的特征
        preprocessor = ColumnTransformer(
            transformers=[
                ('robust', RobustScaler(), robust_features),
                ('standard', StandardScaler(), standard_features),
                ('minMax', MinMaxScaler(feature_range=(0, 1)), minMax_features),
            ],
            remainder='passthrough'  # 不需要缩放的特征保持原样
        )
        X_scaled = preprocessor.fit_transform(X_scaled_without_price)
    else:
        X_scaled = preprocessor.transform(X_scaled_without_price)

    # 使用 Close 的均值和标准差手动缩放其他相关特征
    price_related_scaled = (df[price_related_features] - close_scaler.mean_[0]) / close_scaler.scale_[0]

    # 将缩放后的 price_related_features 替换回 X_scaled
    X_scaled_df = pd.DataFrame(X_scaled, columns=preprocessor.get_feature_names_out())

    # 将 price_related_scaled 替换回 X_scaled_df 中
    for i, feature in enumerate(price_related_features):
        X_scaled_df[feature] = price_related_scaled.iloc[:, i]

    X_scaled_df['Close'] = X_close_scaled
    # 导出特定列到 CSV 文件
    X_scaled = X_scaled_df.to_numpy()

    return X_scaled, close_scaler, preprocessor


symbol = "ETHUSDT"
interval = "15m"
look_back = 1 #使用回看n根數據
epochs = 150
batch_size = 128
total_klines = 150000
get_local_file_name = 'ETHUSDT_15m_2023-12-31_23-59-59_150000_calculated.csv'

input_model_infos = []

# end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# end_time_seconds = end_time / 1000
# end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
# end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# end_time_string = "2023-12-31 23:59:59"
# df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
# df = modules_data.clean_data(df)

# df, customized_cols_infos, new_cols = customized_specific_period_col(df)

# df = modules_data.clean_data(df)

# # 将 'datetime' 列转换为 datetime 类型
# df['datetime'] = pd.to_datetime(df['datetime'])

# # 将 'datetime' 列转换为 UTC+8
# df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')

# # 如果需要移除时区信息，可以使用 .dt.tz_localize(None)
# df['datetime'] = df['datetime'].dt.tz_localize(None)

# df.to_csv('original_df_15m.csv', index=False)

df = pd.read_csv('original_df_15m.csv')

lookahead = 24
percentage = 0.015
prob_threshold = 0.9

cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'fibonacci_1', 'fibonacci_0',
        'Open_Close_pct', 'High_Low_pct', 'Up_Shadow_pct', 'Down_Shadow_pct',
        'EMA_7', 'EMA_25', 'EMA_99', 
        'BB_Width', 'Upper Band', 'Lower Band', 'Middle Band',
        'Close_Volatility', 'High_Volatility', 'Low_Volatility', 'Open_Volatility', 'ATR', 'VWAP', 'Cumulative_VP', 'Cumulative_Volume', 
    ]
X = df[cols]

X = set_y_only_up_label(X, lookahead, percentage)
# X = set_y_label_sequence(X, lookahead, percentage)
# X = set_y_win_lose_label_sequence(X, lookahead, percentage, percentage / 2)
drop_front_data_count = 1000
X = X[drop_front_data_count:-300]
X.reset_index(drop=True, inplace=True)

# X = set_y_multiple_label(X, 96, lookahead, percentage)
# X.to_csv('X.csv', index=False)

# X, test_data = train_test_split(X, test_size=0.2, random_state=42, shuffle=False)


y = X['y']
X.drop('y', axis=1, inplace=True)

print(y.value_counts())

robust_features = ['Volume', 'Open_Close_pct', 'High_Low_pct', 'Up_Shadow_pct', 'Down_Shadow_pct', 'ATR', 'BB_Width', 'Close_Volatility', 'High_Volatility', 'Low_Volatility', 'Open_Volatility', 'Cumulative_VP', 'Cumulative_Volume']
standard_features = ['Close', 'High', 'Low', 'Open', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'fibonacci_1', 'fibonacci_0', 'EMA_7', 'EMA_25', 'EMA_99', 'Upper Band', 'Lower Band', 'Middle Band', 'VWAP', 'up_percentage']
minMax_features = []

# 假设 X 是你的数据集
# 定义需要以 Close 尺度缩放的特征
### 這部分得到的成果沒有比較好
# close_feature = ['Close']
# price_related_features = ['High', 'Low', 'Open', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'fibonacci_1', 'fibonacci_0', 'EMA_7', 'EMA_25', 'EMA_99', 'Upper Band', 'Lower Band', 'Middle Band', 'VWAP', 'up_percentage', 'down_percentage']
# X_scaled, close_scaler, preprocessor = scaler(X, price_related_features, robust_features)

preprocessor = ColumnTransformer(
    transformers=[
        ('robust', RobustScaler(), robust_features),
        ('standard', StandardScaler(), standard_features),
        ('minMax', MinMaxScaler(feature_range=(0, 1)), minMax_features),
    ],
    remainder='passthrough'  # 不需要缩放的特征保持原样
)
X_scaled = preprocessor.fit_transform(X)

n = 200
d = 5
lr = 0.3
# 打开CSV文件准备写入
# with open('model_results_confidence08.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     # 写入列名
#     writer.writerow(['n_estimators', 'max_depth', 'max_leaves', 'learning_rate', 'total_accuracy', 'class_0_accuracy', 'class_1_accuracy', 'class_2_accuracy', 'class_3_accuracy', 'class_4_accuracy', 'class_0_denominator', 'class_1_denominator', 'class_2_denominator', 'class_3_denominator', 'class_4_denominator', 'class_0_numerator', 'class_1_numerator', 'class_2_numerator', 'class_3_numerator', 'class_4_numerator', 'total_profit_percentage'])

#     for n in range(100, 1500, 100):
#         for d in range(3, 13, 2):
#             for lr in np.arange(0.05, 0.55, 0.05):
model_big_trend = xgb.XGBClassifier(n_estimators=n, max_depth=d, learning_rate=lr, random_state=42, device='cuda', verbosity=1)

model_big_trend.fit(X_scaled, y)

test_data = get_test_data()
# total_accuracy, class_accuracies, class_denominators, class_numerators, total_profit_percentage = validation(test_data, model_big_trend, preprocessor, close_scaler, cols, lookahead, percentage, price_related_features, robust_features)
total_accuracy, class_accuracies, class_denominators, class_numerators, total_profit_percentage = validation(test_data, model_big_trend, preprocessor, None, cols, lookahead, percentage, None, robust_features, prob_threshold)
print(total_accuracy, class_accuracies, class_denominators, class_numerators, total_profit_percentage)
import sys
import os
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam

from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, PReLU, Conv1D, MaxPooling1D, Flatten, LeakyReLU, ReLU, Bidirectional, Attention, LayerNormalization, Input, Activation, RepeatVector, Permute, Multiply, Layer, concatenate, Concatenate
from tensorflow.keras.initializers import GlorotUniform

from keras.regularizers import l1_l2
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

from scipy import stats

from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression

import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import csv
import matplotlib.pyplot as plt

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

    # df, supertrend_delta_info = indicators.super_trend_delta_and_risk(df, is_need_return_function_info=True)
    # customized_cols_infos.append(supertrend_delta_info)
    # df, supertrend_delta_info = indicators.super_trend_delta_and_risk_v1(df, is_need_return_function_info=True)
    # customized_cols_infos.append(supertrend_delta_info)
    # df, supertrend_risk_score_info = indicators.calculate_supertrend_risk(df, is_need_return_function_info=True)
    # customized_cols_infos.append(supertrend_risk_score_info)

    # df, look_back_24h_min_max_price_info = indicators.add_24h_min_max_price(df, look_back=96, is_need_return_function_info=True)
    # customized_cols_infos.append(look_back_24h_min_max_price_info)

    # df = calculate_volatility(df, 8)
    # di_len = 14
    # adx_len = 14
    # df = calculate_adx(df, di_len, adx_len)
    # df = calculate_moving_averages(df)
    # df = calculate_bollinger_bands(df)
    # df = calculate_momentum(df)
    # df = calculate_ppo(df)
    # df = calculate_trend(df, 288, 0.04)
    df = indicators.trend_min_max_price_indicator(df, 192)
    df = indicators.calculate_candelstick_patterns(df)

    # 计算价格变化和对数收益率
    # df['Open_Change_Rate'] = df['Open'].pct_change().fillna(0)
    # df['High_Change_Rate'] = df['High'].pct_change().fillna(0)
    # df['Low_Change_Rate'] = df['Low'].pct_change().fillna(0)
    # df['Close_Change_Rate'] = df['Close'].pct_change().fillna(0)
    # # 计算成交量变化率
    # df['Volume_Change'] = df['Volume'].pct_change().fillna(0)

    # df['Close_Open_Ratio'] = df['Close'] / df['Open']
    # df['High_Low_Ratio'] = df['High'] / df['Low']
    # df['Close_High_Ratio'] = df['Close'] / df['High']
    # df['Close_Low_Ratio'] = df['Close'] / df['Low']

    # # 计算布林带宽度
    # df['BB_Width'] = df['Upper Band'] - df['Lower Band']
    # # 计算布林带宽度变化率
    # df['BB_Width_Ratio'] = df['BB_Width'].pct_change().fillna(0)
    
    # # 對數收益率
    # df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    # df['Volatility'] = df['Log_Returns'].rolling(window=20).std()
    
    # df['SMA_5'] = indicators.calculate_sma(df['Close'], 5)
    # df['SMA_10'] = indicators.calculate_sma(df['Close'], 10)

    df['EMA_7'] = indicators.calculate_ema(df['Close'], 7)
    df['EMA_25'] = indicators.calculate_ema(df['Close'], 25)
    df['EMA_99'] = indicators.calculate_ema(df['Close'], 99)

    # df = calculate_VWAP(df)

    # 紀錄新的cols
    modified_cols = set(df.columns)
    # 篩選多出來的cols
    new_cols = list(modified_cols - original_cols)
    return df, customized_cols_infos, new_cols




# 最大前瞻窗口
max_lookahead = 24

# # 生成特征和目标
# X, _, _ = signals.prepare_features_and_targets(df, max_lookahead, cols, min_increase=0.003, max_increase=0.02, step=0.001)
cols = [
        'Open', 'High', 'Low', 'Close', 'Future_High', 'Future_Low', 'Volume', 'fibonacci_0.382', 
        'fibonacci_0.5', 'fibonacci_0.618', 'Open_Close_pct', 'High_Low_pct', 
        'Up_Shadow_pct', 'Down_Shadow_pct', 'EMA_7', 'EMA_25', 'EMA_99', 'RSI', 'Middle Band', 'Upper Band', 'Lower Band', 'ATR', 'lookahead', 'increase'
    ]
X = pd.read_csv('original_features.csv')

robust_features = ['Volume', 'Open_Close_pct', 'High_Low_pct', 'Up_Shadow_pct', 'Down_Shadow_pct', 'ATR']
standard_features = ['Low', 'Close', 'High', 'Open', 'fibonacci_0.618', 'fibonacci_0.5', 'fibonacci_0.382', 'Future_High', 'EMA_7', 'EMA_25', 'EMA_99', 'Middle Band', 'Upper Band', 'Lower Band']
minMax_features = []

# 列出每个缩放器/转换器对应的特征
preprocessor = ColumnTransformer(
    transformers=[
        ('price', RobustScaler(), robust_features),
        ('percent', StandardScaler(), standard_features),
        ('bounded', MinMaxScaler(feature_range=(0, 1)), minMax_features),
    ],
    remainder='passthrough'  # 不需要缩放的特征保持原样
)
preprocessor.fit(X)

model_up = xgb.XGBClassifier()  # or xgb.XGBRegressor() depending on your model type
model_up.load_model('xgboost_up.model')

model_down = xgb.XGBClassifier()  # or xgb.XGBRegressor() depending on your model type
model_down.load_model('xgboost_down.model')

# symbol = "ETHUSDT"
# interval = "15m"
# look_back = 1 #使用回看n根數據
# epochs = 150
# batch_size = 128
# total_klines = 150000
# get_local_file_name = ''

# # predict目前最新的資料
# end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# end_time_seconds = end_time / 1000
# end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
# end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
# df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 20000, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)

# # 将 'datetime' 列转换为 datetime 类型
# df['datetime'] = pd.to_datetime(df['datetime'])

# # 将 'datetime' 列转换为 UTC+8
# df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')

# # 如果需要移除时区信息，可以使用 .dt.tz_localize(None)
# df['datetime'] = df['datetime'].dt.tz_localize(None)
# df, customized_cols_infos, new_cols = customized_specific_period_col(df)

# drop_front_data_count = 1000
# df = df[drop_front_data_count:]
# df.reset_index(drop=True, inplace=True)

# df.fillna(0, inplace=True)
# df.replace([np.inf, -np.inf], np.nan, inplace=True)
# df.interpolate(method='linear', inplace=True)

# numeric_cols = df.select_dtypes(include=[np.number]).columns

# # 检查 NaN 值的数量
# nan_counts = df[numeric_cols].isna().sum()
# print("NaN Counts:\n", nan_counts)

# inf_counts = np.isinf(df[numeric_cols]).sum()
# print("Infinity Counts:\n", inf_counts)

# total_nan_inf = nan_counts + inf_counts
# print("Total NaN and Infinity Counts:\n", total_nan_inf)

# # 检查所有列中 NaN 值的行
# nan_volume_rows = df[df[numeric_cols].isna().any(axis=1)]
# inf_volume_rows = df[np.isinf(df[numeric_cols]).any(axis=1)]
# print("Rows with NaN column:\n", nan_volume_rows)
# print("Rows with inf column:\n", inf_volume_rows)
# df.reset_index(drop=True, inplace=True)
df = pd.read_csv('backtesting_data.csv')
df = df[2000:]

increase_range = np.arange(0.001, 0.021, 0.001)
lookahead_range = range(8, 25)
prob_threshold = 0.8  # 概率阈值
increase_threshold = 0.008  # 最小增量阈值
fee = 0.0012

total_profit = 0  # 总利润
tp_price = None
sl_price = None
entry_price = None
long_position = False
short_position = False
best_lookahead = 0
candles_held = 0  # Number of candles held
highest_prob = None
lowest_prob = None
best_increase = None
stop_loss_increase = None
win_count = 0
lose_count = 0

continue_long = 0

is_importance = False
importance_price = None

up_finish = False
down_finish = False

profits_over_time = []  # To store total profit at each K-line
# 打开CSV文件以进行写入
with open(f'backtest_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # 写入表头
    writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'TP', 'SL', 'Prob_up', 'Prob_down', 'Total Profit', 'Long', 'Short'])
    
    for i in range(len(df) - max(lookahead_range)):
        print(i)
        # Check if currently in a long position
        if long_position:
            candles_held += 1  # Increment the number of candles held
            if df['Low'].iloc[i] <= sl_price:
                # Stop-loss triggered, record the loss
                profit = ((sl_price - entry_price) / entry_price) - fee
                total_profit += profit
                long_position = False  # Exit position
                candles_held = 0  # Reset the candles held
                tp_price = None
                sl_price = None
                entry_price = None
                highest_prob = None
                lowest_prob = None
                best_increase = None
                stop_loss_increase = None
                if profit < 0:
                    lose_count += 1
                elif profit >= 0:
                    win_count += 1
            elif df['High'].iloc[i] >= tp_price:
                # Take-profit triggered, record the profit
                profit = ((tp_price - entry_price) / entry_price) - fee
                total_profit += profit
                long_position = False  # Exit position
                candles_held = 0  # Reset the candles held
                tp_price = None
                sl_price = None
                entry_price = None
                highest_prob = None
                lowest_prob = None
                best_increase = None
                stop_loss_increase = None
                if profit < 0:
                    lose_count += 1
                elif profit >= 0:
                    win_count += 1
            elif candles_held >= best_lookahead:
                # Reached the lookahead limit, close the position
                profit = ((df['Close'].iloc[i] - entry_price) / entry_price) - fee
                total_profit += profit
                long_position = False  # Exit position
                candles_held = 0  # Reset the candles held
                tp_price = None
                sl_price = None
                entry_price = None
                highest_prob = None
                lowest_prob = None
                best_increase = None
                stop_loss_increase = None
                if profit < 0:
                    lose_count += 1
                elif profit >= 0:
                    win_count += 1

            profits_over_time.append(total_profit)
        # Check if currently in a short position
        elif short_position:
            candles_held += 1  # Increment the number of candles held
            if df['High'].iloc[i] >= sl_price:
                # Stop-loss triggered, record the loss
                profit = ((entry_price - sl_price) / entry_price) - fee
                total_profit += profit
                short_position = False  # Exit position
                candles_held = 0  # Reset the candles held
                tp_price = None
                sl_price = None
                entry_price = None
                highest_prob = None
                lowest_prob = None
                best_increase = None
                stop_loss_increase = None
                if profit < 0:
                    lose_count += 1
                elif profit >= 0:
                    win_count += 1
            elif df['Low'].iloc[i] <= tp_price:
                # Take-profit triggered, record the profit
                profit = ((entry_price - tp_price) / entry_price) - fee
                total_profit += profit
                short_position = False  # Exit position
                candles_held = 0  # Reset the candles held
                tp_price = None
                sl_price = None
                entry_price = None
                highest_prob = None
                lowest_prob = None
                best_increase = None
                stop_loss_increase = None
                if profit < 0:
                    lose_count += 1
                elif profit >= 0:
                    win_count += 1
            elif candles_held >= best_lookahead:
                # Reached the lookahead limit, close the position
                profit = ((entry_price - df['Close'].iloc[i]) / entry_price) - fee
                total_profit += profit
                short_position = False  # Exit position
                candles_held = 0  # Reset the candles held
                tp_price = None
                sl_price = None
                entry_price = None
                highest_prob = None
                lowest_prob = None
                best_increase = None
                stop_loss_increase = None
                if profit < 0:
                    lose_count += 1
                elif profit >= 0:
                    win_count += 1

            profits_over_time.append(total_profit)
        else:
            profits_over_time.append(total_profit)
            
            all_combinations = []
            # 遍历所有 increase 和 lookahead 的组合
            for increase in increase_range:
                for lookahead in lookahead_range:
                    future_high = df['Close'].iloc[i] * (1 + increase)
                    future_low = df['Close'].iloc[i] * (1 - increase)
                    # 将所有组合添加到列表中
                    all_combinations.append([
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

            # 将所有组合转换为 DataFrame
            input_data = pd.DataFrame(all_combinations, columns=cols)

            # 对输入数据进行预处理
            input_data = preprocessor.transform(input_data)

            # 一次性预测所有组合的概率
            probs_up = model_up.predict_proba(input_data)[:, 1]
            probs_down = model_down.predict_proba(input_data)[:, 1]

            # 将预测结果存储回对应的组合
            up_candidates = [(prob_up, combination[-1], combination[-2]) for prob_up, combination in zip(probs_up, all_combinations)]
            down_candidates = [(prob_down, combination[-1], combination[-2]) for prob_down, combination in zip(probs_down, all_combinations)]

            # 筛选符合条件的组合
            up_valid = [x for x in up_candidates if x[1] >= increase_threshold and x[0] >= prob_threshold]
            down_valid = [x for x in down_candidates if x[1] >= increase_threshold and x[0] >= (prob_threshold - 0.1)]
            sl_threshold = 2
            sl_prob_threshold = 0.6
            
            # if continue_long >= 2:
            #     down_candidates_for_sl = [
            #         (prob, increase, lookahead) for prob, increase, lookahead in down_candidates
            #         if increase <= best_increase / sl_threshold and lookahead == best_lookahead and prob < sl_prob_threshold
            #     ]
            #     if down_candidates_for_sl:
            #         continue_long = 0
            #         long_position = True
            #         lowest_prob, stop_loss_increase, stop_loss_lookahead = min(down_candidates_for_sl, key=lambda x: x[0])
            #         tp_price = df['Close'].iloc[i] * (1 + best_increase)
            #         sl_price = df['Close'].iloc[i] * (1 - stop_loss_increase)
            #         entry_price = df['Close'].iloc[i]
            # else:
            #     if len(up_valid) > 0:
            #         continue_long += 1
            #         best_increase, best_lookahead = max(up_valid, key=lambda x: x[1])[1:]
            #     else:
            #         continue_long = 0
            stop_loss = 0.01

            if len(up_valid) > 0 and len(down_valid) > 0:
                importance_price = df['Close'].iloc[i]
                best_increase, best_lookahead = max(up_valid, key=lambda x: x[0])[1:]
                is_importance = True
            else:
                if is_importance:
                    if df['Low'].iloc[i] <= importance_price * (1 - best_increase):
                        down_finish = True
                    if df['High'].iloc[i] >= importance_price * (1 + best_increase):
                        up_finish = True

                    if down_finish and up_finish:
                        is_importance = False
                        importance_price = None
                    elif down_finish:
                        if len(down_valid) == 0:
                            long_position = True
                            stop_loss_increase = stop_loss if (best_increase / sl_threshold) <= stop_loss else best_increase / sl_threshold
                            stop_loss_increase = min(0.015, stop_loss_increase)
                            
                            tp_price = importance_price * (1 + best_increase)
                            sl_price = df['Close'].iloc[i] * (1 - stop_loss_increase)
                            entry_price = df['Close'].iloc[i]
                            is_importance = False

                            down_finish = False
                    elif up_finish:
                        if len(up_valid) == 0:
                            short_position = True
                            stop_loss_increase = stop_loss if (best_increase / sl_threshold) <= stop_loss else best_increase / sl_threshold
                            stop_loss_increase = min(0.015, stop_loss_increase)

                            tp_price = importance_price * (1 - best_increase)
                            sl_price = df['Close'].iloc[i] * (1 + stop_loss_increase)
                            entry_price = df['Close'].iloc[i]
                            is_importance = False

                            up_finish = False
                

            


            # if len(up_valid) > 0 and len(down_valid) == 0:
            #     best_increase, best_lookahead = max(up_valid, key=lambda x: x[1])[1:]
            #     highest_prob = max(up_valid, key=lambda x: x[1])[0]
            #     lowest_prob = 1 - highest_prob
                
            #     # 查找 SL
            #     down_candidates_for_sl = [
            #         (prob, increase, lookahead) for prob, increase, lookahead in down_candidates
            #         if increase <= best_increase / sl_threshold and lookahead == best_lookahead and prob < sl_prob_threshold
            #     ]

            #     if down_candidates_for_sl:
            #         long_position = True
            #         lowest_prob, stop_loss_increase, stop_loss_lookahead = min(down_candidates_for_sl, key=lambda x: x[0])
            #         stop_loss_increase = 0.006 if stop_loss_increase <= 0.006 else best_increase / sl_threshold

                    # tp_price = df['Close'].iloc[i] * (1 + best_increase)
                    # sl_price = df['Close'].iloc[i] * (1 - stop_loss_increase)
                    # entry_price = df['Close'].iloc[i]
                # else:
                #     stop_loss_increase = best_increase / sl_threshold
                #     stop_loss_lookahead = best_lookahead
                #     lowest_prob = 1 - highest_prob
                
                # stop_loss_increase = 0.006 if stop_loss_increase <= 0.006 else best_increase / sl_threshold

                # tp_price = df['Close'].iloc[i] * (1 + best_increase)
                # sl_price = df['Close'].iloc[i] * (1 - stop_loss_increase)
                # entry_price = df['Close'].iloc[i]


            # elif len(down_valid) > 0 and len(up_valid) == 0:
            #     short_position = True
            #     best_increase, best_lookahead = max(down_valid, key=lambda x: x[1])[1:]
            #     highest_prob = max(down_valid, key=lambda x: x[0])[0]
            #     lowest_prob = 1 - highest_prob

            #     # 查找 SL
            #     up_candidates_for_sl = [
            #         (prob, increase, lookahead) for prob, increase, lookahead in up_candidates
            #         if increase <= best_increase / sl_threshold and lookahead == best_lookahead and prob < sl_prob_threshold
            #     ]

            #     if up_candidates_for_sl:
            #         short_position = True
            #         lowest_prob, stop_loss_increase, stop_loss_lookahead = min(up_candidates_for_sl, key=lambda x: x[0])
                    
            #     else:
            #         stop_loss_increase = best_increase / sl_threshold
            #         stop_loss_lookahead = best_lookahead
            #         lowest_prob = 1 - highest_prob

            #     tp_price = df['Close'].iloc[i] * (1 - best_increase)
            #     sl_price = df['Close'].iloc[i] * (1 + stop_loss_increase)
            #     entry_price = df['Close'].iloc[i]
                
        # 将数据写入CSV文件
        writer.writerow([
            df['datetime'].iloc[i],
            df['Open'].iloc[i], 
            df['High'].iloc[i], 
            df['Low'].iloc[i], 
            df['Close'].iloc[i], 
            tp_price, 
            sl_price, 
            highest_prob, 
            lowest_prob, 
            total_profit,
            long_position,
            short_position,
            best_increase,
            best_lookahead,
            stop_loss_increase,
            win_count,
            lose_count,
        ])

    profits_over_time.append(total_profit)

    

print(f"Total Profit: {total_profit}")

# Plot the total profit over time
plt.figure(figsize=(10, 6))
plt.plot(profits_over_time, label="Total Profit")
plt.title("Total Profit Over Time")
plt.xlabel("K-lines")
plt.ylabel("Total Profit")
plt.legend()
plt.grid(True)
plt.show()
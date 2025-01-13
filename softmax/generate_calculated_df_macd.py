import os, sys
sys.path.append(os.getcwd())

import pandas as pd

current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.join(current_path, '..')

from modules import data as modules_data

symbol = 'ETHUSDT'
interval = '15m'
end_time_string = '2023-11-30 23:59:59'
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'MACD', 'Signal', 'Hist', 'Hist', 'MACD_Cross', 'SMA_200', 'Stochastic_Oscillator_9', 'ATR', 'RSI', 'Up Trend', 'Down Trend', 'Super Trend', 'Middle Band', 'Upper Band', 'Lower Band', 'Target']

df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 150000, is_need_save_original_data=False, is_read_local=False, is_need_calculated=False)
df = modules_data.calculate_df_all_data(df, interval, False, '', True)
target_counts = df['Target'].value_counts()
print(target_counts)

df = df[cols]
print(df)

df.to_csv(f'{parent_path}/local_data/ETHUSDT_15m_MACD_CROSS_original.csv')

df = df[df['Target'] >= 0]
df.reset_index(drop=True, inplace=True)
print(df)

df.to_csv(f'{parent_path}/local_data/ETHUSDT_15m_MACD_CROSS.csv')

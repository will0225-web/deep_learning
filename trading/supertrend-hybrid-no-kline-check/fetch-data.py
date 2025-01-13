import os, sys
sys.path.append(os.getcwd())

from datetime import datetime, timedelta
from modules import indicators as indicators

current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.join(current_path, '..')

from modules import data as modules_data
from trading.utils.data import customized_specific_period_col

symbol = 'ETHUSDT'
interval = '15m'

v1_model_name = 'supertrend'
v2_model_name = 'supertrend_v2'

v1_data_file_name = 'ETHUSDT_15m_supertrend_hybrid_v1.csv'
v2_data_file_name = 'ETHUSDT_15m_supertrend_hybrid_v2.csv'

# 透過計算，確保資料只抓到上一根 closed 的 K 線
current_time = datetime.now()
time_elapsed_last_bar = current_time.minute % 15
if current_time.timestamp() % 900 == 899:
    # 若執行快了 0.0xxxx 秒，會因時間差沒有更新到最新資料
    last_bar_close_time = current_time -  timedelta(minutes=time_elapsed_last_bar)
else:
    last_bar_close_time = current_time - timedelta(minutes=time_elapsed_last_bar + 15)
end_time_string = last_bar_close_time.strftime('%Y-%m-%d %H:%M:%S')

df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 200000, is_need_calculated=True)

df_v1 = df
df_v2 = df.copy()

df_v1, _, _ = customized_specific_period_col(df_v1, model_type=v1_model_name, calculate_target=True)
df_v2, _, _ = customized_specific_period_col(df_v2, model_type=v2_model_name, calculate_target=True)

target_dir = os.path.join(os.path.abspath(os.path.dirname('__file__')), 'local_data/')

if not os.path.exists(target_dir):
    print('Target directory does not exist')
    os.mkdir(target_dir)

df_v1.to_csv(f'local_data/{v1_data_file_name}')
df_v2.to_csv(f'local_data/{v2_data_file_name}')

print(f'v1 target counts: {df_v1["Target"].value_counts()}')
print(f'v2 target counts: {df_v2["Target"].value_counts()}')

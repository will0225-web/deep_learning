import os, sys
sys.path.append(os.getcwd())

from datetime import datetime, timedelta
from modules import indicators as indicators

current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.join(current_path, '..')

from modules import data as modules_data
from utils.data import customized_specific_period_col

symbol = 'ETHUSDT'
interval = '15m'

model_name='supertrend_v2'
data_file_name = 'ETHUSDT_15m_realtime_caculated_614.csv'

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
df, _, _ = customized_specific_period_col(df, model_type=model_name, calculate_target=True)

target_dir = os.path.join(os.path.abspath(os.path.dirname('__file__')), 'local_data/')
csv_file = ''

if not os.path.exists(target_dir):
    print('Target directory does not exist')
    os.mkdir(target_dir)

csv_file = target_dir + data_file_name

df.to_csv(f'local_data/{data_file_name}')

target_counts = df['Target'].value_counts()
print(target_counts)

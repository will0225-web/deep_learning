import sys
import os
import datetime

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules import data as modules_data

symbol = "ETHUSDT"
interval = "1m"
total_klines = 2500000
end_time_string = '2023-12-31 23:59:59'
# get_local_file_name = 'ETHUSDT_5m_2023-12-31_23-59-59_450000_calculated.csv'
get_local_file_name = ''
# end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# end_time_seconds = end_time / 1000
# end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
# end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_calculated_need_save=True, is_need_calculated=True)
df, target_function_info = modules_data.set_target(df)
target_counts = df['Target'].value_counts()
print(target_counts)
print(target_function_info)

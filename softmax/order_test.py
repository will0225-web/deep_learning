import sys
import os

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import datetime
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import custom_model_fit_indicators as custom_model_fit_indicators

from tensorflow.keras.utils import to_categorical
from binance.client import Client
from binance.enums import *
import json

def calculate_hold_count(entry_datetime, current_datetime, interval):
    """
    计算持仓时长（以数据点数量计算）。
    entry_datetime_str: 入场时间字符串，格式为"YYYY-MM-DD HH:MM:SS"。
    current_datetime_str: 当前时间字符串，同上。
    interval: 数据点的时间间隔，以分钟为单位。例如，15m为15。
    """
    if entry_datetime == '':
        return 0
    # 计算时间差
    delta = current_datetime - entry_datetime
    
    # 将时间差转换为分钟
    delta_minutes = delta.total_seconds() / 60
    
    # 计算持仓的数据点数量
    hold_count = delta_minutes / interval
    
    return hold_count

def save_state(state_dict, file_path='trading_state.json'):
    """保存当前状态到文件"""
    for key, value in state_dict.items():
        if isinstance(value, pd.Timestamp):
            state_dict[key] = value.isoformat()
    with open(file_path, 'w') as file:
        json.dump(state_dict, file, indent=4)

def load_state(file_path='trading_state.json'):
    """从文件加载状态"""
    try:
        with open(file_path, 'r') as file:
            state_dict = json.load(file)
            return state_dict
    except FileNotFoundError:
        return {}  # 如果文件不存在，返回空字典

def get_original_data(symbol, interval, cols):
    original_df = modules_data.get_binance_klines_backward(symbol, interval, "2023-12-31 23:59:59", 70080, cols, is_need_save_original_data=False, is_read_local=True, is_need_calculated=True)

    # 预处理数据
    original_X = original_df[cols]
    original_y = original_df['Target']

    
    indicator_look_back = 192
    price_look_back = 480
    original_X = indicators.add_lower_low_higher_high(original_X, 0.06, look_back=indicator_look_back)
    original_X = indicators.add_price_indicator(original_X, look_back=price_look_back)
    original_X = indicators.add_24h_min_max_price(original_X, look_back=96)
    # 再拿掉前後無參考性資料
    original_X = original_X[(price_look_back + 1 + 1600):-300]
    original_X.reset_index(drop=True, inplace=True)
    original_y = original_y[(price_look_back + 1 + 1600):-300]
    original_y.reset_index(drop=True, inplace=True)
    return original_X, original_y


def multiclass_f1_score(y_true, y_pred):
    return custom_model_fit_indicators.multiclass_one_hot_f1_score(y_true, y_pred)


# 特徵欄位
cols = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'MACD', 'Signal', 'Hist', 'MACD_Long_Short', 'MACD_Stronger', 'MACD_Cross', 'Up Trend', 'Down Trend', 'Super Trend', 'ATR', 'TR', 'SMA_55', 'SMA_200', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_26', 'EMA_50', 'green_three_soldiers', 'red_three_soldiers', 'rsi_overbought', 'rsi_oversold']
symbol = "ETHUSDT"
interval = "15m"
look_back = 48 #使用回看n根數據
epochs = 100
batch_size = 128

# real
api_key = 'ZcYLtM5WiH9qlWWRLP1z9xh4DmAsorJOfwECb5IlS1jIOihffRWrdir5kHPRIAA9'
api_secret = 'X3fSAnqrJL6r2hfG1Q88FCI1tjmJHjcHmOfFjeBXacSumtMGYd65UVvdMn5kdzTh'
client = Client(api_key, api_secret)

symbol = "ETHUSDT"
leverage = 40  # 设置杠杆倍数为40倍

client.futures_change_leverage(symbol=symbol, leverage=leverage)

# 拿原始資料
original_X, original_y = get_original_data(symbol, interval, cols)
# 當前
end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

end_time_seconds = end_time / 1000
end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# 特定
# end_time_string = "2024-03-09 07:00:01"

# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 3000, cols, is_need_save_original_data=False, is_read_local=False, is_need_calculated=True)

# 预处理数据
X = df[cols]
y = df['Target']


indicator_look_back = 192
price_look_back = 480
X = indicators.add_lower_low_higher_high(X, 0.06, look_back=indicator_look_back)
X = indicators.add_price_indicator(X, look_back=price_look_back)
X = indicators.add_24h_min_max_price(X, look_back=96)


# 再拿掉前後無參考性資料
X = X[(price_look_back + 1 + 1600):]
X.reset_index(drop=True, inplace=True)
y = y[(price_look_back + 1 + 1600):]
y.reset_index(drop=True, inplace=True)

# 选择要缩放的列
ochl_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'TR', 'Upper Band', 'Lower Band', 'Middle Band']
ochl_data = X[ochl_cols]

robust_features = []
standard_features = ['Close', 'High', 'Low', 'Open', 'Volume', 'TR', 'Upper Band', 'Lower Band', 'Middle Band']
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
preprocessor.fit(original_X[ochl_cols])  # original_X 是原始数据
# 对特征进行缩放
X_ochl_scaled = preprocessor.transform(ochl_data)


current_cols = ['red_three_soldiers', 'green_three_soldiers', 'MACD_Stronger', 'rsi_oversold', 'rsi_overbought']
current_data = X[current_cols]

robust_features = []
standard_features = []
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
preprocessor.fit(original_X[current_cols])  # original_X 是原始数据
# 对特征进行缩放
X_current_scaled = preprocessor.transform(current_data)
X_current_scaled = X_current_scaled[look_back:]

# 将目标变量转换为分类格式
y_categorical = to_categorical(y, num_classes=5)

# 使用look_back创建数据集
np_X, np_y = modules_data.create_look_back_dataset(X_ochl_scaled, y_categorical, look_back)

# 載入已經訓練好的模型
# 獲取當前文件的絕對路徑
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.join(current_path, '..')
trained_model_name = f'{parent_path}/trained_models/1709987084_softmax_loss-0.7022_accuracy-0.7377_f1-0.3519'
model = tf.keras.models.load_model(trained_model_name, custom_objects={'multiclass_f1_score': multiclass_f1_score})

df = df[(price_look_back + 1 + 1600):]
df.reset_index(drop=True, inplace=True)
X['High'] = df['High']
X['Low'] = df['Low']
X['Close'] = df['Close']
X['datetime'] = df['datetime']

# 評估模型
loss, accuracy, f1 = model.evaluate([np_X, X_current_scaled], np_y, batch_size=batch_size)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1: {f1:.4f}")

predicted = model.predict([np_X, X_current_scaled])

# 加载之前的状态
state = load_state()
position = state.get('position', 0)  # 如果文件中没有position信息，默认为0（无仓位）
entry_price = state.get('entry_price', 0)  # 默认入场价格为0
entry_predicted_target = state.get('entry_predicted_target', 0)
entry_datetime = state.get('entry_datetime', '')  # 默认索引为-1，表示未持仓
max_profit = state.get('max_profit', 0)
if entry_datetime != '':
    entry_datetime = pd.to_datetime(entry_datetime)

stop_loss = 0.008
take_profit = 0.012
stop_loss_3_4 = 0.012

entry_confidence = 0.4
entry_confidence_3_4 = 0.7

predicted_index = -2
hold_count_limit = 16

# 取最新一笔预测结果（假设predicted是按时间顺序排列的）
latest_prediction = predicted[predicted_index]
predicted_target = np.argmax(latest_prediction)
confidence = np.max(latest_prediction)

# 获取最新的市场数据
original_data = X.iloc[predicted_index]
original_close = original_data['Close']
original_high = original_data['High']
original_low = original_data['Low']
original_datetime = original_data['datetime']

print(original_close, original_datetime, predicted_target, confidence)

hold_count = calculate_hold_count(entry_datetime, original_datetime, 15)

info = client.futures_exchange_info()
symbol_info = [s for s in info['symbols'] if s['symbol'] == 'ETHUSDT'][0]

price_precision = symbol_info['pricePrecision']
quantity_precision = symbol_info['quantityPrecision']

if position != 0:
    position_info = client.futures_position_information(symbol=symbol)
    position_amt = 0
    for pos in position_info:
        if pos['symbol'] == symbol:
            position_amt = abs(float(pos['positionAmt']))
            break

    if position == 1:
        if (entry_price - original_low) / entry_price >= stop_loss:
            # 止損
            position = 0
            max_profit = 0
            entry_price = 0
            entry_datetime = ''

            print('Long Stop Loss')
        else:
            current_profit = indicators.calculate_percentage_change(original_high, entry_price)
            max_profit = max(max_profit, current_profit)
            if current_profit >= take_profit or hold_count >= hold_count_limit:
                # 結單
                position = 0
                max_profit = 0
                entry_price = 0
                entry_datetime = ''
                print('Long Close Position')

                # 获取当前所有订单
                open_orders = client.futures_get_open_orders(symbol=symbol)
                # 遍历并取消每个订单
                for order in open_orders:
                    result = client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])

                order = client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=position_amt
                )
                

    elif position == -1:
        if (original_high - entry_price) / entry_price >= stop_loss:
            # 止損
            position = 0
            max_profit = 0
            entry_price = 0
            entry_datetime = ''

            print('Short Stop Loss')
        else:
            current_profit = -indicators.calculate_percentage_change(original_low, entry_price)
            max_profit = max(max_profit, current_profit)
            if current_profit >= take_profit or hold_count >= hold_count_limit:
                # 結單
                position = 0
                max_profit = 0
                entry_price = 0
                entry_datetime = ''
                print('Short Close Position')

                # 获取当前所有订单
                open_orders = client.futures_get_open_orders(symbol=symbol)
                # 遍历并取消每个订单
                for order in open_orders:
                    result = client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])

                order = client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quantity=position_amt
                )
        # if position == 1 and entry_predicted_target == 4:
        #     if (original_high - entry_price) / entry_price >= stop_loss:
        #         position = 0
        #         max_profit = 0
        #         entry_price = 0
        #         entry_datetime = ''
                
        #     else:
        #         current_profit = calculate_percentage_change(entry_price, original_close)
        #         max_profit = max(max_profit, current_profit)
        #         if current_profit >= take_profit or hold_count >= hold_count_limit:
        #             # 結單
        #             position = 0
        #             max_profit = 0
        #             entry_price = 0
        #             entry_datetime = ''
        #             print('Long Close Position')

        #             # 获取当前所有订单
        #             open_orders = client.futures_get_open_orders(symbol=symbol)
        #             # 遍历并取消每个订单
        #             for order in open_orders:
        #                 result = client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                        
        #             order = client.futures_create_order(
        #                 symbol=symbol,
        #                 side=SIDE_SELL,
        #                 type=ORDER_TYPE_MARKET,
        #                 closePosition='true'
        #             )
        #     if position == -1 and entry_predicted_target == 3:
        #         if (entry_price - original_low) / entry_price >= stop_loss:
        #             position = 0
        #             max_profit = 0
        #             entry_price = 0
        #             entry_datetime = ''
        #         else:
        #             current_profit = -calculate_percentage_change(entry_price, original_close)
        #             max_profit = max(max_profit, current_profit)
        #             if current_profit >= take_profit or hold_count >= hold_count_limit:
        #                 # 結單
        #                 position = 0
        #                 max_profit = 0
        #                 entry_price = 0
        #                 entry_datetime = ''
        #                 print('Short Close Position')

        #                 # 获取当前所有订单
        #                 open_orders = client.futures_get_open_orders(symbol=symbol)
        #                 # 遍历并取消每个订单
        #                 for order in open_orders:
        #                     result = client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])

        #                 order = client.futures_create_order(
        #                     symbol=symbol,
        #                     side=SIDE_BUY,
        #                     type=ORDER_TYPE_MARKET,
        #                     closePosition='true'
        #                 )

else:
    max_profit = 0
    entryU = 66
    quantity = round(entryU * leverage / original_close, quantity_precision)

    # 开仓逻辑
    if (predicted_target == 1 and confidence >= entry_confidence) or (predicted_target == 3 and confidence >= entry_confidence_3_4) or (predicted_target == 3 and confidence < entry_confidence):  # 预测为多头且当前无仓位
        position = 1
        entry_price = original_close
        entry_datetime = original_datetime
        entry_predicted_target = predicted_target
        print('Long')

        order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=quantity  # 需要根据您的资金管理策略来计算合约数量
        )

        # if predicted_target == 1:
            # 计算止损价格
        stop_loss_price = entry_price * (1 - stop_loss)
        take_profit_price = entry_price * (1 + take_profit)

        stop_loss_price = round(stop_loss_price, price_precision)
        take_profit_price = round(take_profit_price, price_precision)
        stop_loss_order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=FUTURE_ORDER_TYPE_STOP_MARKET,
            quantity=quantity,
            stopPrice=str(stop_loss_price),  # 注意：API需要的是字符串格式
            closePosition="true"
        )

        take_profit_order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
            quantity=quantity,
            stopPrice=str(take_profit_price),  # 注意：API需要的是字符串格式
            closePosition="true"
        )
        # else:
        #     stop_loss_price = entry_price * (1 - take_profit - 0.005)
        #     stop_loss_price = round(stop_loss_price, price_precision)
            
        #     take_profit_price = entry_price * (1 + stop_loss)
        #     take_profit_price = round(take_profit_price, price_precision)

            
        #     stop_loss_order = client.futures_create_order(
        #         symbol=symbol,
        #         side=SIDE_SELL,
        #         type=FUTURE_ORDER_TYPE_STOP_MARKET,
        #         quantity=quantity,
        #         stopPrice=str(stop_loss_price),  # 注意：API需要的是字符串格式
        #         closePosition="true"
        #     )

        #     take_profit_order = client.futures_create_order(
        #         symbol=symbol,
        #         side=SIDE_SELL,
        #         type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
        #         quantity=quantity,
        #         stopPrice=str(take_profit_price),  # 注意：API需要的是字符串格式
        #         closePosition="true"
        #     )

    elif (predicted_target == 2 and confidence >= entry_confidence) or (predicted_target == 4 and confidence >= entry_confidence_3_4) or (predicted_target == 4 and confidence < entry_confidence):  # 预测为空头且当前无仓位
        position = -1
        entry_price = original_close
        entry_datetime = original_datetime
        entry_predicted_target = predicted_target
        print('Short')

        order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=quantity  # 持仓数量，需要与您的开仓买入合约数量相匹配
        )

        # if predicted_target == 2:
        stop_loss_price = entry_price * (1 + stop_loss)
        take_profit_price = entry_price * (1 - take_profit)

        stop_loss_price = round(stop_loss_price, price_precision)
        take_profit_price = round(take_profit_price, price_precision)

        stop_loss_order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=FUTURE_ORDER_TYPE_STOP_MARKET,
            quantity=quantity,
            stopPrice=str(stop_loss_price),
            closePosition="true"
        )
        take_profit_order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
            quantity=quantity,
            stopPrice=str(take_profit_price),
            closePosition="true"
        )
        # else:
            # stop_loss_price = entry_price * (1 + take_profit + 0.005)
            # stop_loss_price = round(stop_loss_price, price_precision)

            # take_profit_price = entry_price * (1 - stop_loss)
            # take_profit_price = round(take_profit_price, price_precision)

            # stop_loss_order = client.futures_create_order(
            #     symbol=symbol,
            #     side=SIDE_BUY,
            #     type=FUTURE_ORDER_TYPE_STOP_MARKET,
            #     quantity=quantity,
            #     stopPrice=str(stop_loss_price),
            #     closePosition="true"
            # )

            # take_profit_order = client.futures_create_order(
            #     symbol=symbol,
            #     side=SIDE_BUY,
            #     type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
            #     quantity=quantity,
            #     stopPrice=str(take_profit_price),
            #     closePosition="true"
            # )
    else:
        print('Nothing')

# 保存当前状态
save_state({'entry_datetime': entry_datetime, 'position': position, 'entry_price': entry_price, 'entry_predicted_target': int(entry_predicted_target), 'max_profit': max_profit})
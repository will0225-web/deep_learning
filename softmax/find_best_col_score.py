import sys
import os
# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import model as model_process
from modules import custom_model_fit_indicators as custom_model_fit_indicators

from tensorflow.keras.utils import to_categorical

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 特徵欄位
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'MACD', 'Signal', 'Hist', 'MACD_Long_Short', 'MACD_Stronger', 'MACD_Cross', 'Up Trend', 'Down Trend', 'Super Trend', 'ATR', 'TR', 'SMA_55', 'SMA_200', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_26', 'EMA_50', 'green_three_soldiers', 'red_three_soldiers', 'rsi_overbought', 'rsi_oversold', 'macd_hist_stronger_continuity', 'macd_hist_weaker_continuity']
symbol = "ETHUSDT"
interval = "15m"
# look_back = 144 #使用回看n根數據
epochs = 100
batch_size = 128

selected_cols = ['Upper Band', 'MACD', 'Lower Band', 'Middle Band', 'SMA_200', 'RSI', 'MACD_Cross', 'MACD_Long_Short', 'SMA_55', 'kline_color']
# selected_cols = ['Upper Band', 'Down Trend', 'Lower Band', 'MACD', 'RSI', 'MACD_Cross', 'MACD_Long_Short', 'SMA_200', 'Hist', 'SMA_55']


# 當前, 'Close', 'MACD',
# end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# end_time_seconds = end_time / 1000
# end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
# end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# 特定
end_time_string = "2023-11-30 23:59:59"

# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 52560, cols, is_need_save_original_data=False, is_read_local=True, is_need_calculated=True)

# 预处理数据
X = df[cols]
y = df['Target']


indicator_look_back = 672
price_look_back = 1440
X = indicators.add_lower_low_higher_high(X, 0.05, look_back=indicator_look_back)
X = indicators.add_price_indicator(X, look_back=price_look_back)

# 再拿掉前後無參考性資料
X = X[1600:-300]
X.reset_index(drop=True, inplace=True)
y = y[1600:-300]
y.reset_index(drop=True, inplace=True)

X['Target'] = y
X.to_csv('X_original.csv')
X.drop(['Target'], axis=1, inplace=True)

X = X[selected_cols]


# 補齊所有欄位
cols = X.columns.to_list()

# 选择要缩放的列
# robust_features = ['Open', 'High', 'Low', 'Close', 'Up Trend', 'Down Trend', 'SMA_55', 'SMA_200', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_26', 'EMA_50', 'max_high_look_back', 'min_low_look_back', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'MACD', 'Signal', 'Hist']
# standard_features = ['Volume', 'ATR', 'TR']
# minMax_features = ['RSI']
robust_features = ['Upper Band', 'Middle Band', 'Lower Band', 'SMA_200', 'SMA_55', 'MACD']
standard_features = []
minMax_features = ['RSI']

# # 列出每个缩放器/转换器对应的特征
preprocessor = ColumnTransformer(
    transformers=[
        ('price', RobustScaler(), robust_features),
        ('percent', StandardScaler(), standard_features),
        ('bounded', MinMaxScaler(feature_range=(0, 1)), minMax_features),
    ],
    remainder='passthrough'  # 不需要缩放的特征保持原样
)

# 对特征进行缩放
X_scaled = preprocessor.fit_transform(X)
# y_categorical = y
y_categorical = to_categorical(y, num_classes=5)

# 使用look_back创建数据集
# X_lagged, y = modules_data.create_look_back_dataset(X_scaled, y_categorical, look_back_value)
# X_lagged = X_lagged.reshape(X_lagged.shape[0], -1) 
# 假设X_scaled是缩放后的完整特征集，y是目标变量

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 特征选择
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
feature_importances = rf.feature_importances_

# 选择前10个最重要的特征
top_10_features_indices = feature_importances.argsort()[-10:][::-1]
top_10_features_names = [cols[i] for i in top_10_features_indices]

print(top_10_features_names)


# 假设 X_scaled 和 y 已经准备好
look_back_values = [12, 24, 48, 96, 144, 288, 672]  # 示例 look_back 值
best_score = 0
best_look_back = None

for look_back in look_back_values:
    X_look_back, y_look_back = modules_data.create_look_back_dataset(X_scaled, y_categorical, look_back)
    X_look_back = X_look_back.reshape(X_look_back.shape[0], -1)  # 将数据转换为适合逻辑回归的二维形式
    y_labels = np.argmax(y_look_back, axis=1)  # 从独热编码转换回类别标签

    # X_train, X_test, y_train, y_test = train_test_split(X_look_back, y_labels, test_size=0.2, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X_look_back, y_labels, test_size=0.2, random_state=42, stratify=y_labels)

    # 使用逻辑回归作为简化模型
    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    print(f"Look back: {look_back}, Accuracy: {score}")
    if score > best_score:
        best_score = score
        best_look_back = look_back

print(f"Best look_back: {best_look_back} with accuracy: {best_score}")
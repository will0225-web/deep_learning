import sys
import os
from sklearn.linear_model import LogisticRegression
import numpy as np

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import datetime

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def customized_specific_period_col(df):
    customized_cols_infos = []
    
    # 紀錄原先的cols
    original_cols = set(df.columns)
    price_look_back = 672
    
    # indicator_look_back = 288
    
    # df, lower_low_higher_high_info = indicators.add_lower_low_higher_high(df, 0.04, look_back=indicator_look_back, is_need_return_function_info=True)
    # customized_cols_infos.append(lower_low_higher_high_info)
    
    df, price_indicator_info = indicators.add_price_indicator(df, look_back=price_look_back, is_need_return_function_info=True)
    customized_cols_infos.append(price_indicator_info)

    # df, supertrend_delta_info = indicators.super_trend_delta_and_risk(df, is_need_return_function_info=True)
    # customized_cols_infos.append(supertrend_delta_info)
    df, supertrend_delta_info = indicators.super_trend_delta_and_risk_v1(df, is_need_return_function_info=True)
    customized_cols_infos.append(supertrend_delta_info)
    # df, supertrend_risk_score_info = indicators.calculate_supertrend_risk(df, is_need_return_function_info=True)
    # customized_cols_infos.append(supertrend_risk_score_info)

    # df, look_back_24h_min_max_price_info = indicators.add_24h_min_max_price(df, look_back=96, is_need_return_function_info=True)
    # customized_cols_infos.append(look_back_24h_min_max_price_info)

    df['EMA_9'], ema_9_info = indicators.calculate_ema(df['Close'], 9, is_need_return_function_info=True)
    customized_cols_infos.append(ema_9_info)
    df['SMA_15'], sma_15_info = indicators.calculate_sma(df['Close'], 15, is_need_return_function_info=True)
    customized_cols_infos.append(sma_15_info)
    df['SMA_30'], sma_30_info = indicators.calculate_sma(df['Close'], 30, is_need_return_function_info=True)
    customized_cols_infos.append(sma_30_info)
    ### export customized col
    # df['EMA_9'], ema_9_function_info = indicators.calculate_ema(df['Close'], 9, is_need_return_function_info=True)
    # customized_cols_infos.append(ema_9_function_info)
    # df['EMA_21'], ema_21_function_info = indicators.calculate_ema(df['Close'], 21, is_need_return_function_info=True)
    # customized_cols_infos.append(ema_21_function_info)
    # df['ATR_35'], atr_35_function_info  = indicators.calculate_atr(df['TR'], 35, is_need_return_function_info=True)
    # customized_cols_infos.append(atr_35_function_info)

    # df['EMA_14'], ema_14_function_info  = indicators.calculate_ema(df['Close'], 14, is_need_return_function_info=True)
    # customized_cols_infos.append(ema_14_function_info)

    # 紀錄新的cols
    modified_cols = set(df.columns)
    # 篩選多出來的cols
    new_cols = list(modified_cols - original_cols)
    return df, customized_cols_infos, new_cols

# SMA: 29904 0.360701262564175, 100000
# EMA: 27856 0.31852584568261033

# MACD: (100, 99) 0.13944518839632775
# Signal: (1, 2) -0.01153927461989506
# Hist: (46, 10) 0.022661390676630325
# MACD_Cross: (10, 77) 0.01689270114450725
# MACD_Stronger: (20, 127) 0.022909215852763384
# MACD_Long_Short: (10, 43) 0.004374199855314253

# rsi_overbought: (14, 65) 0.052930529737699517
# rsi_oversold: (48, 44) 0.09708001372257367

# TR: (999, 0) 0.316485999076136 i越高越好
# ATR: (35, 0) 0.3939505768666517

# Super Trend: (22, 3) 0.01846115354937622 沒屁用
# Up Trend: (247, 2) 0.13979000108525744, i越高越好
# Down Trend: (4, 4) 0.15410804296063565

# Upper Band: (10003, 2) 0.3497736860207934 i越高越好
# Lower Band: (2, 2) 0.14128573700571848 i越低越好

# fibonacci_0.382: (29953, 0) 0.3327351748028141
# fibonacci_0.5: (28261, 0) 0.34720627274280924
# fibonacci_0.618: (28261, 0) 0.35533297091428473 # fibonnacci在28261都出現一制性的關聯性

# lower_low: (282, 0) 0.21156119215882468 ,0.06
# higher_high: (123, 0) 0.14746789202189375 ,0.06

# 特徵欄位
symbol = "ETHUSDT"
interval = "15m"
look_back = 288 #使用回看n根數據
epochs = 300
batch_size = 128
total_klines = 100000
get_local_file_name = 'ETHUSDT_15m_2023-12-31_23-59-59_150000_calculated.csv'
input_model_infos = []

# 當前
# end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# end_time_seconds = end_time / 1000
# end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
# end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# 特定
end_time_string = "2023-12-31 23:59:59"


# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
df, customized_cols_infos, new_cols = customized_specific_period_col(df)
df, target_function_info = modules_data.set_target(df)

target_counts = df['Target'].value_counts()
print(target_counts)

X = df

# 觀測所有的cols
## 排除datetime
cols = X.columns
cols = cols[cols != 'datetime']

# cols = X.columns

# 只檢查想要觀測的cols
# cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'kline_color', 'percentage', 'MACD', 'Signal', 'Hist', 'MACD_Long_Short', 'MACD_Cross', 'MACD_Stronger', 'K', 'D', 'SMA_200', 'SMA_55', 'SMA_27', 'SMA_21', 'SMA_12', 'EMA_12', 'EMA_26', 'EMA_50', 'TR', 'ATR', 'RSI', 'Up Trend', 'Down Trend', 'Super Trend', 'Middle Band', 'Upper Band', 'Lower Band', 'red_three_soldiers', 'green_three_soldiers', 'rsi_overbought', 'rsi_oversold', 'Target'] + new_cols
X_features_only = X[cols]
# # 选择要缩放的列
# robust_features = ['Open', 'High', 'Low', 'Close', 'Up Trend', 'Down Trend', 'SMA_55', 'SMA_200', 'Middle Band', 'Upper Band', 'Lower Band', 'EMA_26', 'EMA_50', 'max_high_look_back', 'min_low_look_back', 'MACD', 'Signal', 'Hist', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618']
# standard_features = ['Volume', 'ATR', 'TR', 'max_high_look_back_24h', 'min_low_look_back_24h']
# minMax_features = ['RSI']

# # 列出每个缩放器/转换器对应的特征
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('price', RobustScaler(), robust_features),
#         ('percent', StandardScaler(), standard_features),
#         ('bounded', MinMaxScaler(feature_range=(0, 1)), minMax_features),
#     ],
#     remainder='passthrough'  # 不需要缩放的特征保持原样
# )

# # 对特征进行缩放
# X_scaled = preprocessor.fit_transform(X_features_only)
# X = X_scaled

# 邏輯回歸找關聯
# log_reg = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
# # 训练模型
# log_reg.fit(X_scaled, y)

# # 获取特征系数
# coefficients = log_reg.coef_[0]
# # 将特征系数转换为DataFrame
# coeff_df = pd.DataFrame(coefficients, index=X_features_only.columns, columns=['Coefficient'])

# # 根据系数的绝对值排序并选取关联最大的前10个特征
# top_10_features = coeff_df.abs().sort_values(by='Coefficient', ascending=False).head(10)

# # 使用热图展示这些特征系数
# plt.figure(figsize=(8, 6))
# sns.heatmap(top_10_features, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'label': 'Coefficient'})
# plt.title('Top 10 Features Correlated with Target = 1')
# plt.show()
# y_categorical = to_categorical(y, num_classes=5)

# X, y = modules_data.create_look_back_dataset(X_scaled, y, look_back=144)

# 切分數據集
# test_and_validation_size = 0.3
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_and_validation_size, shuffle=False)




# n = 100
# max_depth = None
# min_samples_leaf = 1
# min_samples_split = 2
# # 随机森林模型
# rf = RandomForestClassifier(n_estimators=n, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features='sqrt', min_samples_split=min_samples_split, class_weight='balanced', random_state=42)
# # X_train = X_train.reshape(X_train.shape[0], -1)
# rf.fit(X_train, y_train)

# # 訓練集預測
# train_predictions = rf.predict(X_train)
# train_accuracy = accuracy_score(y_train, train_predictions)
# f1_macro = f1_score(y_train, train_predictions, average='macro')  # 全局计算真阳性、假阳性和假阴性
# print("Training Accuracy:", train_accuracy)
# print("Training F1:", f1_macro)


# # X_test = X_test.reshape(X_test.shape[0], -1)
# # 預測和評估
# predictions = rf.predict(X_test)
# test_f1_macro = f1_score(y_test, predictions, average='macro')  # 全局计算真阳性、假阳性和假阴性
# print("Accuracy:", accuracy_score(y_test, predictions))
# print("Training F1:", test_f1_macro)

# # 获取特征重要性
# importances = rf.feature_importances_
# # 获取特征重要性并转换为DataFrame
# feature_importances = pd.DataFrame(importances, index=X_features_only.columns, columns=['Importance']).sort_values('Importance', ascending=False)
# # 选择前10个最重要的特征的重要性
# top_10_feature_importances = feature_importances.head(15)
# # 展开为适合热图显示的格式
# top_10_for_heatmap = top_10_feature_importances.T

# # 使用热图展示这些特征系数
# plt.figure(figsize=(8, 6))
# sns.heatmap(top_10_for_heatmap, annot=True, cmap='viridis', cbar_kws={'label': 'Importance'})
# plt.title('Top 10 Features Correlated with Target = 1')
# plt.show()


X = X_features_only
# X = X[X['Target'].isin([1, 5])]
# X = X[X['Target'] != 0] # 0量體太大, 所以篩選掉
# 计算特征间的相关性
corr = X.corr()

# 定义我们关注的特征
focus_features = ['Target']

# 找出与每个关注特征相关性最高的前10个特征
top_correlations = pd.DataFrame()
for feature in focus_features:
    top_features = corr[feature].sort_values(ascending=False).index[1:31]  # 排除自身，取前10
    top_correlations[feature] = corr.loc[top_features, feature]


# 绘制与Target的相关性热图
plt.figure(figsize=(8, 10))
sns.heatmap(top_correlations, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation of Features with Target')
plt.show()


# 用來儲存異常值比例的字典
outlier_fraction = {}

# 計算每個特徵的異常值比例
for col in cols:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    outlier_count = X[(X[col] < (Q1 - 1.5 * IQR)) | (X[col] > (Q3 + 1.5 * IQR))].shape[0]
    outlier_fraction[col] = outlier_count / X.shape[0]

# 將異常值比例轉換成DataFrame以便於繪圖
outlier_fraction_df = pd.DataFrame(list(outlier_fraction.items()), columns=['Feature', 'Outlier Fraction'])

# 設定圖表尺寸
plt.figure(figsize=(10, 8))

# 繪製條形圖顯示每個特徵的異常值比例
sns.barplot(x='Outlier Fraction', y='Feature', data=outlier_fraction_df.sort_values('Outlier Fraction', ascending=False))

# 設定標題和軸標籤
plt.title('Outlier Fraction by Feature')
plt.xlabel('Fraction of Outliers')
plt.ylabel('Feature')

# 顯示圖表
plt.show()





# RSI關聯性很低
# correlations = {}
# check_col = 'Super Trend'

# best_corr = -1  # 初始化最佳相关性为一个很小的数
# best_params = None  # 用于存储最佳参数组合
# sequence = np.arange(0.05, 0.2, 0.01)
# for i in range(1, 200):
#     for j in np.arange(1, 11, 1):
#         X = df[cols]
#         X['Target'] = df['Target']

#         X['Up Trend'], X['Down Trend'], X['Super Trend'] = indicators.calculate_supertrend(X, i, j)

#         # 再拿掉前後無參考性資料
#         X = X[max(1600, (i + 1600)):-300]
#         X.reset_index(drop=True, inplace=True)

#         X = X[X['Target'].isin([1, 2, 3, 4])]

#         corr = X[[check_col, 'Target']].corr().iloc[0, 1]  # 获取SMA和Target之间的相关性

#         correlations[(i, j)] = corr  # 存储每一对参数组合的相关性

#         # 更新最佳参数组合
#         if corr > best_corr:
#             best_corr = corr
#             best_params = (i, j)

#         print(i, j, best_params, best_corr)

# # 找到相关性最高的SMA窗口
# best_window = max(correlations, key=correlations.get)
# best_corr = correlations[best_window]

# print(best_window, best_corr)

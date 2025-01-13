import sys
import os

# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import datetime
from keras.regularizers import l1_l2, l1, l2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import classification_report
import os
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils import class_weight

from joblib import dump

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV

def evaluate_model(estimator, X_train, y_train, X_val, y_val):
    estimator.fit(X_train, y_train)
    y_pred_val = estimator.predict(X_val)
    f1_val = f1_score(y_val, y_pred_val, average='macro')
    return f1_val

# 特徵欄位
# cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'MACD', 'Signal', 'Hist', 'MACD_Cross', 'Up Trend', 'Down Trend', 'Super Trend', 'TR', 'ATR', 'SMA_27', 'SMA_55', 'SMA_200', 'kline_color', 'Middle Band', 'Upper Band', 'Lower Band']
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'percentage', 'RSI', 'MACD', 'Signal', 'Hist', 'Up Trend', 'Down Trend', 'Super Trend', 'MACD_Stronger', 'ATR', 'SMA_21', 'Middle Band', 'Upper Band', 'Lower Band', 'Stochastic_Oscillator_14', 'EMA_26', 'kline_color']

symbol = "ETHUSDT"
interval = "15m"
look_back = 288 #使用回看n根數據
# 當前
# end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# 特定
end_time_string = "2023-10-31 23:59:59"

# 轉毫秒
# end_time = int(datetime.datetime.timestamp(datetime.datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S"))) * 1000
# Step 1: 獲取數據
df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 35040, cols, is_need_save_original_data=False, is_read_local=True, is_need_calculated=True)

new_data_end_time_string = "2023-11-30 23:59:59"
df_new = modules_data.get_binance_klines_backward(symbol, interval, new_data_end_time_string, 4000, cols, is_need_save_original_data=False, is_read_local=False, is_need_calculated=True) 

# 再拿掉前後無參考性資料
df = df[1000:-300]
df_new = df_new[1000:-300]

df.reset_index(drop=True, inplace=True)
df.to_csv('test.csv')

df_new.reset_index(drop=True, inplace=True)

# macd_cross_indices = df.index[df['MACD_Cross'] != 0].tolist()
# df_new_macd_cross_indices = df_new.index[df_new['MACD_Cross'] != 0].tolist()

# 使用Scaler對數據進行縮放
scaled_data = df[cols]
scaled_new_data = df_new[cols]

X, y = [], []
for i in range(look_back, len(scaled_data) + 1):
    X.append(scaled_data[i - look_back:i])
    y.append(df.iloc[i - 1]['Target'])

y = [element.astype(int) for element in y]
X, y = np.array(X), np.array(y)

X_new, y_new = [], []
for new_index in range(look_back, len(scaled_new_data) + 1):
    X_new.append(scaled_new_data[new_index - look_back:new_index])
    y_new.append(df_new.iloc[new_index - 1]['Target'])
y_new = [element.astype(int) for element in y_new]
X_new, y_new = np.array(X_new), np.array(y_new)

# class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y.ravel())
# class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}   

# rf = RandomForestClassifier(random_state=42)
# param_distributions = { 
#     'n_estimators': [100, 200, 300, 500, 1000, 1500, 2000, 3000],
#     'max_depth': [10, 20, 30, 50, 100, None],
#     'min_samples_split': [2, 5, 10, 20, 50, 100],
#     'min_samples_leaf': [1, 2, 4, 8 ,16, 32, 64],
#     'max_features': ['sqrt', 'log2', 3, 5, 8, 10, 15, 16],
#     'class_weight': ['balanced', None, class_weights_dict]
# }
# n_iter_search = 50  # 定义迭代次数

# rf = RandomForestClassifier(random_state=42)

# random_search = RandomizedSearchCV(rf, param_distributions=param_distributions, 
#                                 n_iter=n_iter_search, scoring='f1_macro', 
#                                 cv=5, verbose=2, n_jobs=-1, random_state=42)
# X_reshape = X.reshape(X.shape[0], -1)
# # 在训练集上执行随机搜索
# random_search.fit(X_reshape, y)

# # 获取最佳参数
# best_params = random_search.best_params_
# print("Best Parameters: ", best_params)

# # 使用最佳参数在整个训练集上训练模型
# best_rf = RandomForestClassifier(**best_params, random_state=42)
# best_rf.fit(X_reshape, y)

# # 在测试集上评估模型
# y_pred_test = best_rf.predict(X_new.reshape(X_new.shape[0], -1))
# f1_test = f1_score(y_new, y_pred_test, average='macro')
# print("Test F1 Score: ", f1_test)

# 將數據分割為訓練集、驗證集和測試集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.07, stratify=y)
for n in [100, 300, 500]:
# for n in [100, 500, 1500, 3000, 5000]:
    for max_depth in [None , 8, 10, 15, 20]: # 更深的树可以捕获更复杂的模式，但可能导致过拟合。相反，较浅的树可以提高泛化能力
        for min_samples_leaf in [1, 10, 20, 30]:
        # for min_samples_leaf in [10]: #增加这个参数值可以减少模型的复杂性，提高泛化能力
            for min_samples_split in [2, 50]:
                for max_feature in ['sqrt']:
                    # 初始化随机森林分类器
                    model = RandomForestClassifier(n_estimators=n, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_feature, min_samples_split=min_samples_split, class_weight='balanced', criterion='entropy', random_state=42)
                    
                    X_reshape = X.reshape(X.shape[0], -1)
                    # 训练模型
                    model.fit(X_reshape, y)  # 重塑X_train以适应模型
                    s = model.score(X_reshape, y)
                    
                    test_s = model.score(X.reshape(X.shape[0], -1), y)
                    # predict
                    predicted = model.predict(X_new.reshape(X_new.shape[0], -1))
                    # y_pred = np.argmax(predicted, axis=1)

                    # 在测试集上评估模型
                    score = model.score(X_new.reshape(X_new.shape[0], -1), y_new)
                    f1_macro = f1_score(y_new, predicted, average='macro')  # 全局计算真阳性、假阳性和假阴性
                    # repo = classification_report(X_new.reshape(X_new.shape[0], -1), y_new)
                    print(f'Model Accuracy: train-{s}, test-{test_s}, new:{score}, f1:{f1_macro}')
                    
                    # 獲取當前文件的絕對路徑
                    current_path = os.path.abspath(os.path.dirname(__file__))

                    # 定義上層目錄的路徑
                    parent_path = os.path.join(current_path, '..')
                    time = int(datetime.datetime.timestamp(datetime.datetime.now()))
                    model_directory = f'{parent_path}/trained_models/grid_forest/{time}_trainAcc-{s:.4f}_test-{test_s:.4f}_accuracy-{score:.4f}_f1:{f1_macro:.4f}_n-{n}_maxd-{max_depth}_minsl-{min_samples_leaf}_minss-{min_samples_split}_max_feature-{max_feature}'
                    model_filename = f'{model_directory}/softmax_trainAcc-{s:.4f}_test-{test_s:.4f}_accuracy-{score:.4f}_f1:{f1_macro:.4f}_n-{n}_maxd-{max_depth}_minsl-{min_samples_leaf}_minss-{min_samples_split}_max_feature-{max_feature}.joblib'

                    # 创建目录
                    os.makedirs(model_directory, exist_ok=True)

                    dump(model, model_filename, indent=4)


                    reshapre_X = X_new.reshape(X_new.shape[0], -1)
                    predicted = model.predict(reshapre_X)


                    # 假设class_names是全部可能的类别名称
                    class_names = ['0', '1', '2', '3', '4']

                    # 确定预测和真实标签中存在的类别
                    unique_labels = np.unique(np.concatenate((y_new, predicted)))

                    # 为这些类别创建标签列表
                    labels = [class_names[label] for label in unique_labels]

                    cm = confusion_matrix(y_new, predicted, labels=unique_labels)

                    
                    plt.figure(figsize=(11, 11))
                    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title('Confusion Matrix')
                    plt.xticks(rotation=90)
                    plt.yticks(rotation=0)
                    plt.savefig(fname=f'{model_directory}/cm-{time}.png')
                    plt.close()
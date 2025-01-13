import sys, os
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

def save_model(trained_model_name, model, loss, accuracy):
    model.save(trained_model_name)

    record_valoss_valaccuracy(loss, accuracy, trained_model_name, -1)

def save_to_csv(df, filename):
    # 檢查文件是否存在和是否為空
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        # 如果文件存在且非空，追加模式且不加入header
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        # 如果文件不存在或為空，寫入模式且加入header
        df.to_csv(filename, mode='w', index=False)

def record_valoss_valaccuracy(loss, accuracy, trained_model_name, backtesting_len):
    # backtesting_len -1 等於第一次訓練完的時候跑的loss, accuracy
    data = {
        "loss": [loss],
        "accuracy": [accuracy],
        "backtesting_len": [backtesting_len]
    }

    df = pd.DataFrame(data)
    # 追加模式存入CSV
    save_to_csv(df, f'{trained_model_name}/loss_accuracy.csv')


def export_epoch_info(history, trained_model_name, use_f05_score=False):
    time = int(datetime.timestamp(datetime.now()))

    if use_f05_score == False:
        # 將每個 epoch 的訓練數據繪製成圖表與 csv
        train_loss = history.history['loss']
        train_accuracy = history.history['accuracy']
        train_f1 = history.history['multiclass_f1_score']
        val_loss = history.history['val_loss']
        val_accuracy = history.history['val_accuracy']
        val_f1 = history.history['val_multiclass_f1_score']

        fig, axs = plt.subplots(3, sharex=True)
        fig.suptitle('Training and Validation Metrics')
        axs[0].set_title('Loss')
        axs[0].plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss')
        axs[0].plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
        axs[0].set(ylabel='Loss')
        axs[1].set_title('Accuracy')
        axs[1].plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Train Accuracy')
        axs[1].plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation Accuracy')
        axs[1].set(xlabel='Epoch', ylabel='Accuracy')
        axs[2].set_title('F1')
        axs[2].plot(range(1, len(train_f1) + 1), train_f1, label='Train F1')
        axs[2].plot(range(1, len(val_f1) + 1), val_f1, label='Validation F1')
        axs[2].set(ylabel='F1')

        plt.savefig(fname=f'{trained_model_name}/epoch-metrics-{time}.png')

        d = {'loss': train_loss, 'accuracy': train_accuracy, 'F1': train_f1, 'val_loss': val_loss, 'val_accuracy': val_accuracy, 'val_F1': val_f1}
        df_epoch = pd.DataFrame(data=d)
        df_epoch.to_csv(f'{trained_model_name}/epoch-metrics.csv', float_format='%.4f')
    else:
        # 將每個 epoch 的訓練數據繪製成圖表與 csv
        train_loss = history.history['loss']
        train_accuracy = history.history['accuracy']
        train_f05 = history.history['multiclass_f05_score']
        val_loss = history.history['val_loss']
        val_accuracy = history.history['val_accuracy']
        val_f05 = history.history['val_multiclass_f05_score']

        fig, axs = plt.subplots(3, sharex=True)
        fig.suptitle('Training and Validation Metrics')
        axs[0].set_title('Loss')
        axs[0].plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss')
        axs[0].plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
        axs[0].set(ylabel='Loss')
        axs[1].set_title('Accuracy')
        axs[1].plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Train Accuracy')
        axs[1].plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation Accuracy')
        axs[1].set(xlabel='Epoch', ylabel='Accuracy')
        axs[2].set_title('F0.5')
        axs[2].plot(range(1, len(train_f05) + 1), train_f05, label='Train F0.5')
        axs[2].plot(range(1, len(val_f05) + 1), val_f05, label='Validation F0.5')
        axs[2].set(ylabel='F0.5')

        plt.savefig(fname=f'{trained_model_name}/epoch-metrics-{time}.png')

        d = {'loss': train_loss, 'accuracy': train_accuracy, 'F0.5': train_f05, 'val_loss': val_loss, 'val_accuracy': val_accuracy, 'val_F0.5': val_f05}
        df_epoch = pd.DataFrame(data=d)
        df_epoch.to_csv(f'{trained_model_name}/epoch-metrics.csv', float_format='%.4f')        


def export_epoch_sigmoid_info(history, trained_model_name):
    time = int(datetime.timestamp(datetime.now()))

    # 將每個 epoch 的訓練數據繪製成圖表與 csv
    train_loss = history.history['loss']
    train_accuracy = history.history['accuracy']
    train_f1 = history.history['f1_score']
    val_loss = history.history['val_loss']
    val_accuracy = history.history['val_accuracy']
    val_f1 = history.history['val_f1_score']

    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle('Training and Validation Metrics')
    axs[0].set_title('Loss')
    axs[0].plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss')
    axs[0].plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
    axs[0].set(ylabel='Loss')
    axs[1].set_title('Accuracy')
    axs[1].plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Train Accuracy')
    axs[1].plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation Accuracy')
    axs[1].set(xlabel='Epoch', ylabel='Accuracy')
    axs[2].set_title('F1')
    axs[2].plot(range(1, len(train_f1) + 1), train_f1, label='Train F1')
    axs[2].plot(range(1, len(val_f1) + 1), val_f1, label='Validation F1')
    axs[2].set(ylabel='F1')

    plt.savefig(fname=f'{trained_model_name}/epoch-metrics-{time}.png')

    d = {'loss': train_loss, 'accuracy': train_accuracy, 'F1': train_f1, 'val_loss': val_loss, 'val_accuracy': val_accuracy, 'val_F1': val_f1}
    df_epoch = pd.DataFrame(data=d)
    df_epoch.to_csv(f'{trained_model_name}/epoch-metrics.csv', float_format='%.4f')


def export_macd_cm(trained_model_name, model, X, y):
    time = int(datetime.timestamp(datetime.now()))

    # 4. 預測
    predicted = model.predict(X)
    predicted_class = np.argmax(predicted, axis=1)

    # 從測試數據中提取真實的類別
    true_class = y
    
    # 計算混淆矩陣
    cm = confusion_matrix(true_class, predicted_class)

    # 繪製混淆矩陣
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['long_loss', 'short_loss', 'long_win', 'short_win'], yticklabels=['long_loss', 'short_loss', 'long_win', 'short_win'])
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix')
    plt.savefig(fname=f'{trained_model_name}/cm-{time}.png')

def export_cm(trained_model_name, model, X, y, labels, threshold=0.7, batch_size=32):
    time = int(datetime.timestamp(datetime.now()))

    # 4. 預測
    predicted = model.predict(X, batch_size)
    predicted_class = np.argmax(predicted, axis=1)
    confidence_scores = np.max(predicted, axis=1)  # 计算置信度

    # 從測試數據中提取真實的類別
    # true_class = np.argmax(y, axis=1)

    high_confidence_indices = np.where(confidence_scores >= threshold)[0]
    print(f'Cases above confidence level {threshold}: {len(high_confidence_indices)}')
    filtered_predicted_class = predicted_class[high_confidence_indices]
    filtered_true_class = np.argmax(y[high_confidence_indices], axis=1)
    
    # 計算混淆矩陣
    # cm = confusion_matrix(true_class, predicted_class, labels=labels)
    cm = confusion_matrix(filtered_true_class, filtered_predicted_class, labels=labels)

    
    # 繪製混淆矩陣
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix')
    plt.savefig(fname=f'{trained_model_name}/cm-{threshold}.png')
    plt.close('all')

def export_sigmoid_cm(trained_model_name, model, X, y, labels, threshold=0.5, batch_size=32):
    time = int(datetime.timestamp(datetime.now()))

    # 预测
    predicted_prob = model.predict(X, batch_size=batch_size)
    # 使用阈值判断类别
    predicted_class = (predicted_prob >= threshold).astype(int).flatten()

    # 真实类别已经是一维的，不需要转换
    true_class = y.flatten()
    
    # 计算混淆矩阵
    labels = [0, 1]  # 二元分类的标签
    cm = confusion_matrix(true_class, predicted_class, labels=labels)

    # 繪製混淆矩陣
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix')
    plt.savefig(fname=f'{trained_model_name}/cm-{time}.png')


def export_ROC(model, trained_model_name, X_test, y_test):
    time = int(datetime.timestamp(datetime.now()))
    # Predict probabilities
    y_probs = model.predict(X_test)
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    # Plotting
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(fname=f'{trained_model_name}/ROC-{time}.png')

def export_layer_parameters(trained_model_name, model, symbol, interval, look_back, df, batch_size, epochs, end_time_string, evaluation_results, test_and_validation_size, validation_ratio_of_test, class_weight_dict, get_local_file_name):
    time = int(datetime.timestamp(datetime.now()))

    
    # 將模型結構轉換為JSON格式
    model_json = model.to_json()
    # 將JSON寫入文件
    with open(f'{trained_model_name}/model-{time}.json', "w") as json_file:
        json_file.write(model_json)
    model.save(f'{trained_model_name}/model-{time}.h5')

    # 构造"Model Compile Config"
    model_compile_config = {
        "jit_compile": None,  # 目前TensorFlow暂无直接方法获取，需手动记录
        "loss": model.loss,  # 如果是直接传入的字符串
        "metrics": ['accuracy', {"class_name": "function", "config": "multiclass_f1_score"}],  # 手动记录
        "optimizer": {
            "class_name": model.optimizer.__class__.__name__,
            "config": model.optimizer.get_config(),  # 使用get_config获取优化器配置
            "module": model.optimizer.__module__,
        }
    }

    time_string = datetime.fromtimestamp(time).strftime("%Y-%m-%d %H:%M:%S")
    # 紀錄模型訓練參數
    model_parameters = {
        'Train Time': time_string,
        'End Time': end_time_string,
        'Symbol': symbol,
        'Interval': interval,
        'Look Back': look_back,
        'Data Length': len(df.index),
        'Model Compile Config': model_compile_config,
        'Model Metrics Result': evaluation_results,
        'Batch Size': batch_size,
        'Epoch': epochs,
        'Test And Validation Size': test_and_validation_size,
        'Validation Ratio Of Test': validation_ratio_of_test,
        'Class Weight Dict': class_weight_dict,
        'Get Local File Name': get_local_file_name
    }
    # 寫入JSON文件
    with open(f'{trained_model_name}/model_params-{time}.json', "w") as file:
        json.dump(model_parameters, file, default=str, indent=4)

def export_connection_info(trained_model_name, batch_size, symbol, interval, look_back, end_time_string, get_original_klines_count, target_function_info, input_model_infos, target_types, customized_cols_infos, drop_front_data_count, drop_back_data_count, get_local_file_name, new_cols):
    connection_info = {
        'End Time': end_time_string,
        'Batch Size': batch_size,
        'Symbol': symbol,
        'Interval': interval,
        'Look Back': look_back,
        'Get Original Klines Count': get_original_klines_count,
        'Target Function Info': target_function_info,
        'Input Model Infos': input_model_infos,
        'Target Types': target_types,
        'Customized Cols Infos': customized_cols_infos,
        'Drop Front Data Count': drop_front_data_count,
        'Drop Back Data Count': drop_back_data_count,
        'Get Local File Name': get_local_file_name, 
        'Customized Cols': new_cols
    }
    with open(f'{trained_model_name}/model_connection_info.json', "w") as file:
        json.dump(connection_info, file, default=str, indent=4)

def export_total_profit_info(df, trained_model_name, time, csv_path):
    df = pd.read_csv(csv_path)
    # 提取total_profit列
    total_profits = df['Total Profit']

    # 绘制折线图
    plt.figure(figsize=(10, 6))  # 设置图表大小
    plt.plot(total_profits, linestyle='-', markersize=5)  # 画出total_profit的变化
    plt.title('Total Profit Over Trades')  # 设置图表标题
    plt.xlabel('Trade Number')  # 设置X轴标签
    plt.ylabel('Total Profit')  # 设置Y轴标签
    plt.grid(True)  # 显示网格
    plt.savefig(fname=f'{trained_model_name}/total_profit-{time}.png')
    plt.close('all')

def export_model_summary(trained_model_name, model):
    with open(f'{trained_model_name}/model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

import numpy as np
import os
import sys
import supertrend_backtesting_v2_support_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.join(current_path, '..')
local_data_path = os.path.join(parent_path, 'local_data')

def get_predict_original_data(model_connection_info):
    symbol = model_connection_info['Symbol']
    interval = model_connection_info['Interval']
    get_local_file_name = model_connection_info['Get Local File Name']
    total_klines = model_connection_info['Get Original Klines Count']
    end_time_string = model_connection_info['End Time']
    drop_front_data_count = model_connection_info['Drop Front Data Count']
    drop_back_data_count = model_connection_info['Drop Back Data Count']
    df, trained_model_path, target_y, X, predicted = supertrend_backtesting_v2_support_model.get_predict(total_klines, end_time_string, '')

    # 假设 target_y 是 one-hot 编码，我们需要转换为标签
    target_labels = np.argmax(target_y, axis=1)
    # 假设 predicted 也是概率数组，我们需要转换为预测的标签
    predicted_labels = np.argmax(predicted, axis=1)
    # 假设 predicted 也是概率数组，我们需要转换为预测的标签
    predicted_labels = np.argmax(predicted, axis=1)
    # 比较 target_labels 和 predicted_labels，不一样则为0，否则为1
    # is_predicted_correct = (target_labels == predicted_labels).astype(int)


    # 初始化一个与target_labels相同长度的数组，用于存储新的编码
    is_predicted_correct = np.zeros_like(target_labels)

    for i in range(len(predicted_labels)):
    # if target_labels[i] in [0, 2, 4, 7]:
        if predicted_labels[i] in [0, 2, 4]:
            # 如果属于0, 2, 4组
            is_predicted_correct[i] = 0 if target_labels[i] == 6 else 1
        elif predicted_labels[i] in [1, 3, 5]:
            # 如果属于1, 3, 5组
            is_predicted_correct[i] = 0 if target_labels[i] == 7 else 1
        elif predicted_labels[i] in [6]:
            is_predicted_correct[i] = 1 if target_labels[i] != predicted_labels[i] else 0
        elif predicted_labels[i] in [7]:
            is_predicted_correct[i] = 1 if target_labels[i] != predicted_labels[i] else 0

    predicted_confidence = np.max(predicted, axis=1)
    X['is_predicted_correct'] = is_predicted_correct
    X['Predicted'] = predicted_labels
    X['Confidence'] = predicted_confidence
    return X, target_y

model_connection_info = {}
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.join(current_path, '..')
trained_model_path = os.path.join(parent_path, 'trained_models/1712129241_softmax_loss-0.7342_accuracy-0.5567_f1-0.5241')
# 构建完整的文件路径
model_info_path = os.path.join(trained_model_path, 'model_connection_info.json')

# 检查文件是否存在
if os.path.exists(model_info_path):
    # 这里可以添加读取或处理文件的代码
    # 例如，读取JSON文件
    import json
    with open(model_info_path, 'r') as file:
        model_connection_info = json.load(file)
else:
    raise FileNotFoundError(f"File not found: {model_info_path}")

X, y = get_predict_original_data(model_connection_info)

symbol = model_connection_info['Symbol']
interval = model_connection_info['Interval']
get_local_file_name = model_connection_info['Get Local File Name']
total_klines = model_connection_info['Get Original Klines Count']
end_time_string = model_connection_info['End Time']
drop_front_data_count = model_connection_info['Drop Front Data Count']
drop_back_data_count = model_connection_info['Drop Back Data Count']
temp_end_time_string = end_time_string.replace(':', '-').replace(' ', '_')
calculate_file_path = os.path.join(local_data_path, f'{symbol}_{interval}_{temp_end_time_string}_{total_klines}_calculated_predicted.csv')

X.to_csv(calculate_file_path)
import numpy as np
import matplotlib.pyplot as plt
from modules.data import denormalize

def plot(prediction, answer, dir):
    plt.plot(prediction, color='red', label='Prediction')
    plt.plot(answer, color='blue', label='Answer')
    plt.savefig(f'linear/{dir}/figure')
    plt.legend(loc='best')
    plt.show()

def plot_subplots(amount, original_data, prediction, test_data):
    fig, axs = plt.subplots(amount)
    fig.suptitle('ETH Price y_test vs. prediction')
    for i in range(amount):
        # print(prediction[:, - i - 1].reshape(-1, 1))
        axs[i].plot(denormalize(original_data.values.reshape(-1, 1), prediction[:, - i - 1].reshape(-1, 1)), color='red', label='Prediction')
        axs[i].plot(denormalize(original_data.values.reshape(-1, 1), test_data[:, - i - 1].reshape(-1, 1)), color='blue', label='Answer')
        axs[i].set_title(f'Last {i + 1} candle answer vs. prediction')      
    plt.savefig('linear/trained_model_directory/figure')
    plt.legend(loc='best')
    plt.show()

def print_predict(model_path, dataframe, prediction, close_column_index):
    np.set_printoptions(precision=2, suppress=True)
    if 'sliding' in model_path:
        i = 1
        for k in prediction:
            close = denormalize(dataframe, k.reshape(1, -1))[0, close_column_index]
            print(f'預測下{i}根K線的收盤價為：%.2f' % close)
            print(f'預測下{i}根K線的ohlc為：{denormalize(dataframe, k.reshape(1, -1))[0]}')
            i += 1
    elif 'multi' in model_path:
        klines = denormalize(dataframe['Close'].values.reshape(-1, 1), prediction[0].reshape(-1, 1))
        for i in range(len(klines)):
            print(f'預測下{i + 1}根K線的收盤價為：%.2f' % klines[i])    
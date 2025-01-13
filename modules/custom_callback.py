import numpy as np
from keras.callbacks import Callback

class CustomEarlyStopping(Callback):
    def __init__(self, patience=20, delta=0.01, log_metric=''):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.delta = delta
        self.best_score = -np.Inf
        self.wait = 0
        self.log_metric = log_metric

    def on_epoch_end(self, epoch, logs=None):
        current_score = logs.get(self.log_metric)
        if (current_score - self.delta) > self.best_score:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"\nStopping training at epoch {epoch + 1}")


class ThresholdMetric(Callback):
    def __init__(self, max_val_loss=1):
        super(ThresholdMetric, self).__init__()
        self.best_score = 0
        self.max_val_loss = max_val_loss

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        score = 0

        # 加入 accuracy, auc 和 val_auc 到分數中
        score += logs.get('accuracy', 0) * 0.05
        score += logs.get('auc', 0) * 0.1
        score += logs.get('f1_score', 0) * 0.15

        score += logs.get('val_auc', 0) * 0.15
        score += logs.get('val_f1_score', 0) * 0.3

        # 根據 val_loss 計算得分
        val_loss = logs.get('val_loss', float('inf'))
        if val_loss >= self.max_val_loss:
            # overfitting的懲罰
            score -= 0.05
        else:
            val_loss_score = max(0, (self.max_val_loss - val_loss) / self.max_val_loss) * 0.25
            score += val_loss_score
        
        # 存儲得分以便於後續訪問
        # self.combined_metric_history.append(score)

        logs['threshold_metric'] = score
        print(f"\nCombined Metric for epoch {epoch + 1}: {score} / 1")

    # def get_final_score(self):
    #     return self.combined_metric_history[-1] if self.combined_metric_history else 0
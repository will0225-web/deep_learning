from keras import backend as K
import tensorflow as tf

def f1_score(y_true, y_pred):
    """计算 F1 分数"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def multiclass_f1_score(y_true, y_pred, num_classes):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.int32)

    f1_scores = []
    for i in range(num_classes):
        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, i), tf.equal(y_pred, i)), tf.float32))
        false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_true, i), tf.equal(y_pred, i)), tf.float32))
        false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, i), tf.not_equal(y_pred, i)), tf.float32))

        # Avoid division by zero
        precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon())
        recall = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())

        # Calculate F1 score, or set to 0 if both precision and recall are 0
        f1 = tf.cond(tf.greater(precision + recall, 0),
                     lambda: 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon())),
                     lambda: tf.constant(0.0))
        
        f1_scores.append(f1)

    f1_scores = tf.stack(f1_scores)
    return tf.reduce_mean(f1_scores)

def multiclass_one_hot_f1_score(y_true, y_pred):
    # 将 y_pred 转换为 one-hot 编码形式
    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=K.shape(y_true)[-1])

    # 计算每个类别的 true positives, false positives 和 false negatives
    true_positives = K.sum(y_true * y_pred, axis=0)
    false_positives = K.sum((1 - y_true) * y_pred, axis=0)
    false_negatives = K.sum(y_true * (1 - y_pred), axis=0)

    # 计算每个类别的 precision 和 recall
    precision = true_positives / (true_positives + false_positives + K.epsilon())
    recall = true_positives / (true_positives + false_negatives + K.epsilon())

    # 计算每个类别的 F1 分数
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    # 返回所有类别 F1 分数的平均值
    return K.mean(f1)

def multiclass_one_hot_fbeta_score(y_true, y_pred, beta=0.5):
    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=K.shape(y_true)[-1])

    true_positives = K.sum(y_true * y_pred, axis=0)
    false_positives = K.sum((1 - y_true) * y_pred, axis=0)
    false_negatives = K.sum(y_true * (1 - y_pred), axis=0)

    precision = true_positives / (true_positives + false_positives + K.epsilon())
    recall = true_positives / (true_positives + false_negatives + K.epsilon())

    # 计算每个类别的 F-beta score
    beta_squared = beta ** 2
    f_beta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

    # 返回所有类别 F-beta score 的平均值
    return K.mean(f_beta)

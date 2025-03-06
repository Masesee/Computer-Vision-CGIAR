import tensorflow as tf

def precision(y_true, y_pred):
    return tf.keras.metrics.Precision()(y_true, y_pred)

def recall(y_true, y_pred):
    return tf.keras.metrics.Recall()(y_true, y_pred)

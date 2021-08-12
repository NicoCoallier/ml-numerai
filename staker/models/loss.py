import tensorflow as tf

def rmsle(y_true, y_pred):
    return tf.sqrt(
        tf.reduce_mean((tf.math.log(y_pred + 1.0) - tf.math.log(y_true + 1.0)) ** 2)
    )


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean((y_pred - y_true) ** 2))

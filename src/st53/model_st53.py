import tensorflow as tf
def build_st53_model(w):
    m=tf.keras.Sequential([
        tf.keras.layers.Input((w,1)),
        tf.keras.layers.LSTM(64,return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    m.compile(optimizer="adam",loss="mse")
    return m

import keras
def build_st39_model(w):
    m=keras.Sequential([
        keras.layers.Input((w,1)),
        keras.layers.LSTM(64,return_sequences=True),
        keras.layers.LSTM(32),
        keras.layers.Dense(1)
    ])
    m.compile(optimizer="adam",loss="mse")
    return m

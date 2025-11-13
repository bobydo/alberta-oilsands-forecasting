import numpy as np, joblib, tensorflow as tf
def load_st39_model(path):
    return tf.keras.models.load_model(f"{path}/st39_model"), joblib.load(f"{path}/st39_meta.pkl")["window"]
def predict_st39(model,window,values):
    arr=np.array(values).reshape(1,window,1)
    return float(model.predict(arr)[0][0])

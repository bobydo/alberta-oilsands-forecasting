import numpy as np, joblib, tensorflow as tf
def load_st53_model(path):
    return tf.keras.models.load_model(f"{path}/st53_model"), joblib.load(f"{path}/st53_meta.pkl")["window"]
def predict_st53(model,window,values):
    arr=np.array(values).reshape(1,window,1)
    return float(model.predict(arr)[0][0])

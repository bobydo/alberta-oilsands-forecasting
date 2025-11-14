import numpy as np, joblib, keras
def load_st53_model(path):
    return keras.models.load_model(f"{path}/st53_model.keras"), joblib.load(f"{path}/st53_meta.pkl")["window"]
def predict_st53(model,window,values):
    arr=np.array(values).reshape(1,window,1)
    return float(model.predict(arr)[0][0])

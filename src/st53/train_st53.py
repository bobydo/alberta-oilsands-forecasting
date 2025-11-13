import numpy as np, joblib
from .preprocess_st53 import load_st53
from .model_st53 import build_st53_model
from src.common.window import create_windows

def train_st53(xls,out):
    df=load_st53(xls)
    values=df["Bitumen"].astype(float).values
    WINDOW=6
    X,y=create_windows(values,WINDOW)
    X=X.reshape((-1,WINDOW,1))
    model=build_st53_model(WINDOW)
    model.fit(X,y,epochs=40,batch_size=16)
    model.save(f"{out}/st53_model")
    joblib.dump({"window":WINDOW},f"{out}/st53_meta.pkl")

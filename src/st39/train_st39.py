import numpy as np, joblib
from .preprocess_st39 import load_st39
from .model_st39 import build_st39_model
from src.common.window import create_windows

def train_st39(xls,out):
    df=load_st39(xls)
    values=df["Production"].astype(float).values
    WINDOW=6
    X,y=create_windows(values,WINDOW)
    X=X.reshape((-1,WINDOW,1))
    model=build_st39_model(WINDOW)
    model.fit(X,y,epochs=40,batch_size=16)
    model.save(f"{out}/st39_model")
    joblib.dump({"window":WINDOW},f"{out}/st39_meta.pkl")

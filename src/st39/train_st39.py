import sys
import numpy as np, joblib
from src.st39.preprocess_st39 import load_st39
from src.st39.model_st39 import build_st39_model
from src.common.window import create_windows

def train_st39(xls,out):
    df=load_st39(xls)
    values=df["Production"].astype(float).values
    WINDOW=6
    X,y=create_windows(values,WINDOW)
    X=X.reshape((-1,WINDOW,1))
    model=build_st39_model(WINDOW)
    model.fit(X,y,epochs=40,batch_size=16)
    model.save(f"{out}/st39_model.keras")
    joblib.dump({"window":WINDOW},f"{out}/st39_meta.pkl")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_st39.py <input_xls> <output_dir>")
        sys.exit(1)
    train_st39(sys.argv[1], sys.argv[2])

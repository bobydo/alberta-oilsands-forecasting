import sys
import numpy as np, joblib
from src.st53.preprocess_st53 import load_st53
from src.st53.model_st53 import build_st53_model
from src.common.window import create_windows

def train_st53(xls,out):
    df=load_st53(xls)
    values=df["Bitumen"].astype(float).values
    WINDOW=6
    X,y=create_windows(values,WINDOW)
    X=X.reshape((-1,WINDOW,1))
    model=build_st53_model(WINDOW)
    model.fit(X,y,epochs=40,batch_size=16)
    model.save(f"{out}/st53_model.keras")
    joblib.dump({"window":WINDOW},f"{out}/st53_meta.pkl")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_st53.py <input_xls> <output_dir>")
        sys.exit(1)
    train_st53(sys.argv[1], sys.argv[2])

import numpy as np
def create_windows(values, window):
    X=[]; y=[]
    for i in range(len(values)-window):
        X.append(values[i:i+window])
        y.append(values[i+window])
    return np.array(X), np.array(y)

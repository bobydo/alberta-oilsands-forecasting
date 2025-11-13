def create_windows(values, window_size):
    X=[];y=[]
    for i in range(len(values)-window_size):
        X.append(values[i:i+window_size]); y.append(values[i+window_size])
    return X,y
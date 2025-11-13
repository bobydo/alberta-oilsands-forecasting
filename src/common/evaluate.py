import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def evaluate_forecast(y_true, y_pred, title="Forecast Evaluation"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    plt.plot(y_true,label="Actual"); plt.plot(y_pred,label="Predicted")
    plt.legend(); plt.grid(); plt.show()
    return mae, rmse

import numpy as np

def prediction(Y_actual,Y_pre):
    res=np.subtract(Y_actual,Y_pre)
    return res
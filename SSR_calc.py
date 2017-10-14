import numpy as np

def SSR_Calc(Y_pre,Y_actual_,row):
    one_matrix = np.ones((row, row))
    SSR_cal_temp1 = np.matmul(Y_pre, np.transpose(Y_pre))
    SSR_cal_temp2 = np.matmul(np.transpose(Y_actual_), one_matrix)
    SSR_cal_temp2 = np.matmul(SSR_cal_temp2, Y_actual_)
    SSR_cal_temp2 = np.divide(SSR_cal_temp2, float(row))
    SSR = SSR_cal_temp1 - SSR_cal_temp2
    return SSR

import numpy as np
import math

def SLR(Y,res):
    #print(n)
    row=len(Y)
    temp=np.ones(row)
    #print(np.shape(temp))
    #print(temp)
    res=np.abs(res)
    #print(res)
    res=np.vstack([temp,res])
    #print(np.shape(res))
    res_transpose=np.transpose(res)
    #print(np.shape(res_transpose))
    temp=np.matmul(res,res_transpose)
    #print(np.shape(temp))
    temp=np.linalg.inv(temp)
    temp=np.matmul(temp,res)
    LSE=np.matmul(temp,Y)
    fitted_Y=np.matmul(res_transpose,LSE)
    #print(fitted_Y)
    return fitted_Y

def weight_cal(Y,res):

    fitted_Y=SLR(Y,res)
    row = len(fitted_Y)
    weighted_value = np.identity(row)
    for i in range(0,row):
        for j in range(0,row):
            if(j==i):
                weighted_value[i][j]=(1/(fitted_Y[i]))
    return weighted_value
import math
import os
import pandas as pd
import numpy as np

def ANOVA_CALC(SSE_list, SST_list, SSR_list, R_sqaured_list, MSE_list,RMSE_list, MSR_list, F_list, Adjusted_R_squared_list,K_FOLD_SIZE):
    SSE_list_Sum = 0
    SSR_list_Sum = 0
    SST_list_Sum = 0
    MSE_list_Sum = 0
    MSR_list_Sum = 0
    R_sqaured_list_Sum = 0
    Adjusted_R_squared_list_Sum = 0
    F_list_Sum = 0
    print(R_sqaured_list)
    print(RMSE_list)
    #LSE_list,SSE_list, SST_list, SSR_list, R_sqaured_list, MSE_list, MSR_list, F_list, Adjusted_R_squared_list = MLR(z)
    # print(len(SSE_list))
    #print(len(SSE_list))
    for i in range(0, K_FOLD_SIZE):
        SSE_list_Sum = SSE_list_Sum + SSE_list[i]
        SSR_list_Sum = SSR_list_Sum + SSR_list[i]
        SST_list_Sum = SST_list_Sum + SST_list[i]
        MSE_list_Sum = MSE_list_Sum + MSE_list[i]
        MSR_list_Sum = MSR_list_Sum + MSR_list[i]
        R_sqaured_list_Sum = R_sqaured_list_Sum + R_sqaured_list[i]
        F_list_Sum = F_list_Sum + F_list[i]
        Adjusted_R_squared_list_Sum = Adjusted_R_squared_list_Sum + Adjusted_R_squared_list[i]
    SSE = SSE_list_Sum / K_FOLD_SIZE
    SSR = SSR_list_Sum / K_FOLD_SIZE
    SST = SST_list_Sum / K_FOLD_SIZE
    print("SSE =",SSE, "SSR=",SSR, "SST=",SST)
    R_sqaured = R_sqaured_list_Sum / K_FOLD_SIZE
    print("R_Squared=",R_sqaured)
    #
    MSR = MSR_list_Sum / float(K_FOLD_SIZE)
    print("MSR=",MSR)
    MSE = MSE_list_Sum / float(K_FOLD_SIZE)
    print("MSE=",MSE)
    print('RMSE=', math.sqrt(MSE))
    F = F_list_Sum / float(K_FOLD_SIZE)
    print("F-value=",F)
    Adjusted_R_squared = Adjusted_R_squared_list_Sum / float(K_FOLD_SIZE)
    print("Adusted R-Squared=",Adjusted_R_squared )
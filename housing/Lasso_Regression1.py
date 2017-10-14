from sklearn.linear_model import Lasso
from sklearn import metrics
import pandas as pd
from K_CROSS_VALIDATION import Ten_fold
import numpy as np
import math
from scipy.stats import f
import itertools
def Lassos(test_data_list,train_data_list,K_fold_size):
    rmse_list=[]
    r_squared_train_list=[]
    r_squared_list=[]
    coeff_list=[]
    F_value=[]
    p=[]
    #res_list=[]
    pre=[]
    act=[]
    res=[]
    res_list=[]
    mse1=[]
    m=[]
    k_list=[]
    for i in range(0,K_fold_size):

        test_data=test_data_list[i]
        train_data=train_data_list[i]
        y_test=test_data["Commercial-rate"]
        y_train=train_data["Commercial-rate"]
        y_train=y_train.values
        y_test=y_test.values
        test_data=test_data.drop(["Commercial-rate","Intercept"],axis=1)
        test=test_data.values
        k_list.append(test)
        #print(test_data.shape)
        train_data=train_data.drop(["Commercial-rate","Intercept"],axis=1)
        train=train_data.values
        r,c=test_data.shape
        reg=Lasso(alpha=10)
        reg=reg.fit(train,y_train)


        y_train_fitted=reg.predict(train)
        r_squared_train=reg.score(train,y_train)
        y_fitted=reg.predict(test)
        r_squared=reg.score(test,y_test)
        mse=metrics.mean_squared_error(y_test,y_fitted)
        mse1.append(mse)
        rmse=math.sqrt(mse)
        mse_train=metrics.mean_squared_error(y_train,y_train_fitted)
        rmse_train=math.sqrt(mse_train)
        r_squared_list.append(r_squared)
        rmse_list.append(rmse)
        means=np.mean(y_test)
        sum=0
        for i in range(0,len(y_test)):
            res_list.append(y_test[i]-y_fitted[i])
        act.append(y_test)
        pre.append(y_fitted)
        res.append(res_list)

        for i in range (0,len(y_test)):
            sum+=(y_test[i]-means)**2
        MSR=sum/c
        F=MSR/mse
        F_value.append(F)
        p.append(f.pdf(F,c,r-c))
        k=reg.coef_

        l=(reg.intercept_)
        m.append(l)
        coeff_list.append(k)



        #r_s=metrics.r2_score(y_test,y_fitted)
        #print(r_squared,mse,rmse,r_squared_train,rmse_train)
        #print(k)
        #print(k1)
    return (m,coeff_list,rmse_list,r_squared_list,F_value,p,mse1,res,pre,act,k_list)


#Lassos(test_data,train_data,K_fold_size)
#def predict(data):

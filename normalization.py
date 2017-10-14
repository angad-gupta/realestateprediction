import pandas as pd
import numpy as np

def Normalization(df):
    list1=[]
    #print(df.shape)
    for col in df.columns:
        min=df[col].min()
        max=df[col].max()
        if col != 'Commercial-price ' and col != 'Intercept' and col != 'Commercial-rate' and col != 'log_commercial-price':

            Standard_Deviation=df[col].std()
            mean=df[col].mean()
            df[col]=((df[col]-min)/(max-min))
            temp=[max-min,min]
            list1.append(temp)
    #print(len(list1))
    return list1,df
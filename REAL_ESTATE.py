#print("asdasd")
import os
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as mp
from Cleaning_data import Outliers
from normalization import Normalization
from denormalization import denormalizaton
from MLR_gr import MLR_gr
from PLR import PLR
from regre import MLR
from K_CROSS_VALIDATION  import Ten_fold
from WLR import WLR
from django.conf import settings

ORDER=3

def Logarithmic_Transfromation(df):
    row, col = df.shape

    for col in df.columns:
        if col == 'Year' or col == 'Road_width' or col == 'Location_Access' or col=='Government-rate':
            for i in range(0,row):
                if (df.loc[i,col]==0):
                    #print
                    df.loc[i,col]=1

            df.loc[:,col] = np.log(df[col])
        # print(df)
        # df=df.drop(df[col])
    df.loc[:,'log_commercial_rate'] = np.log(df['Commercial-rate'])
    return df


def explore_data(y):
    max_price = np.max(y)
    min_price = np.min(y)
    mean_price = np.mean(y)
    median_price = np.median(y)
    standard_deviation = np.std(y)
    print("max Commercial-rate-th:", max_price)
    print("min Commercial-rate-th:", min_price)
    print("mean Commercial-rate-th:", mean_price)
    print("median Commercial-rate-th:", median_price)
    print("standard deviation for Commercial-rate-th:", standard_deviation)


def Ask_Data(Year,Road_width,Location_Access,Governmentrate,road_type,Land_type,PLR_Check):
    data=[]
    data.append(Year)
    data.append(Road_width)
    data.append(float(Location_Access)/100)
    data.append(float(Governmentrate)/100000)
    #data.append(input("Commercial-rate"))
    #road_type=input("Road-Type:")
    if(road_type=='Earthen'or road_type=='earthen'):
        data.append(1)
        data.append(0)
        data.append(0)
        data.append(0)
        data.append(0)
    elif(road_type=='Goreto'):
        data.append(0)
        data.append(1)
        data.append(0)
        data.append(0)
        data.append(0)
    elif (road_type == 'Pitch'):
        data.append(0)
        data.append(0)
        data.append(1)
        data.append(0)
        data.append(0)
    elif (road_type == 'Gravelled'):
        data.append(0)
        data.append(0)
        data.append(0)
        data.append(1)
        data.append(0)
    elif (road_type == 'Paved'):
        data.append(0)
        data.append(0)
        data.append(0)
        data.append(0)
        data.append(1)
    else:
        data.append(0)
        data.append(0)
        data.append(0)
        data.append(0)
        data.append(0)
    #Land_type=input("Land-Type:")
    if(Land_type=='Commercial'):
        data.append(1)
        data.append(0)
    elif (Land_type == 'Residential:'):
        data.append(0)
        data.append(1)
    else:
        data.append(0)
        data.append(0)
    return(data)
def predicting_value(data,LSE):
    sum=LSE[0]
    for i in range(0,len(data)):
        sum+=float(data[0])*LSE[i+1]
    return sum
def LSE_cal(LSE_list,column,K_FOLD_SIZE):
    LSE_avg_list = []
    for i in range(0, column):
        temp = 0
        for j in range(0, K_FOLD_SIZE):
            temp += (LSE_list[j][i])
        LSE_avg_list.append(temp / K_FOLD_SIZE)
    return LSE_avg_list

def multiply(a, b, k):
    r, c = np.shape(a)
    # print(r,c)
    temp = np.zeros((r, c))
    # print(temp)
    for i in range(0, r):
        # print(i)
        # tot+=1
        for j in range(0, c):
            temp[i][j] = a[i][j] * b[j][i]
    k.append(temp)
    # print(k[0][0])
    # print(df.head())
    return k, temp


def interaction_term(x):
    x = x.drop(
        ['Commercial-rate', 'Intercept', 'Earthen','Goreto', 'Pitch', 'Gravelled', 'paved', 'Commercial',
         'Residential'], axis=1)
    k = []

    # print(x.head)
    r, c = x.shape
    # list1=[]
    df1 = x
    df1 = df1.drop(df1.columns[c - 1], axis=1)
    temp1_matrix = df1.values
    ro, co = np.shape(temp1_matrix)
    for i in range(0, c - 1):
        x = x.drop(x.columns[0], axis=1)
        # print(x.head())
        temp2_matrix = x.values
        list1, temp3 = multiply(np.transpose(temp1_matrix), temp2_matrix, k)
        ro, co = np.shape(temp3)
        # print(tot)
        # print(ro,co)
        # for j in range(0,ro):
        # list1.append(temp3[j-1])
        # print(list1)
        temp1_matrix = temp3
        temp1_matrix = np.delete(temp1_matrix, [c - (i + 2)], axis=0)
        temp1_matrix = np.transpose(temp1_matrix)
    # print(i)
    # k+=1
    return list1


def interaction(list1, df):
    for i in range(0, len(list1)):
        for j in range(0, len(list1[i])):
            col = 'col'
            col = col + str(i) + str(j)
            print(col)
            df2 = pd.DataFrame(list1[i][j], columns=[col])
            df = pd.concat([df, df2], axis=1)
    #return df

def powering(df1):
    for col in df1.columns:
        # print(col)
        if col != 'Commercial-price' and col != 'Commercial-rate'and col != 'Intercept'  and col != 'Earthen' and col != 'Pitch' and col !='Goreto'and col != 'Gravelled' and col != 'paved' and col != 'Commercial' and col!='Residential':
            for K in range(2, ORDER):
                co = col + str(K)
                df1.loc[:,co] = df1.loc[:,col] ** K
    return df1

def realstate(Year,Road_width,Location_Access,Governmentrate,road_type,Land_type):
    location ="static/assets/dataq.csv" 
    file_path = os.path.join(settings.STATIC_ROOT, 'dataq.csv')
    df = pd.read_csv(file_path)
    df = df[['Year', 'Road_width', 'Location_Access', 'Commercial-price',  'Governmentrate','Commercial-rate',
              'Earthen', 'Goreto','Pitch', 'Gravelled', 'paved', 'Commercial', 'Residential','Places']]
    # print(row,col)
    row, col = df.shape
    df=df[df.Places != 'Nuwakot']
    df=df.reset_index(drop='TRUE')
    #del df['index']

    df=df.drop(['Places'],axis=1)
    row,column=df.shape
    data = np.ones(row)
    df = df.drop(['Commercial-price'], axis=1)
    df_one = pd.DataFrame({'Intercept': data})

    # print(df.isnull().any())

    # print(df.shape)

    # while (df.isnull().any().any()):
    for col in df.columns:

        if (df[col].isnull().any() and df[col].dtype == np.float64):
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].mean(), inplace=True)
        #print(df[col])
    df = Outliers(df)
    #df = var_transfromation(df)
    df = pd.concat([df_one, df], axis=1)
    df2=df
    #print(df)
    list1 = interaction_term(df)
    #df['log_commercial-price'] = np.log(df['Commercial-rate'])
    # print(df.corr())

    train_data, test_data ,K_FOLD_SIZE= K_CROSS_VALIDATION.Ten_fold(df2, row, col)

    df1 = powering(df)
    interaction(list1, df)
    #print(df.corr())
    row1,col1=df1.shape
    train_data_p,test_data_P,K_FOLD_SIZE=K_CROSS_VALIDATION.Ten_fold(df1,row1,col1)
    #print(train_data_p[1].head())
    # print(train_data[9].head())
    check=0
    print("For WLR")
    LSE_list6, column6 = WLR.WLR(train_data, test_data, K_FOLD_SIZE, check)
    print()
    print()
    print("For MLR")
    #LSE_list,column=MLR.MLR(train_data,test_data,K_FOLD_SIZE,check)
    print()
    print()
    print("For MLR using Gradient Descent")
    #LSE_list1,column1=MLR_gr.MLR_gr(train_data,test_data,K_FOLD_SIZE,check)
    print()
    print()
    print("For PLR")
   # LSE_list2,column2=PLR.PLR(train_data_p,test_data_P,K_FOLD_SIZE,check)

    #LSE=LSE_cal(LSE_list,column,K_FOLD_SIZE)
    #LSE1=LSE_cal(LSE_list1,column1,K_FOLD_SIZE)
    #LSE2=LSE_cal(LSE_list2,column2,K_FOLD_SIZE)
    print()
    print()

    LSE6=LSE_cal(LSE_list6,column6,K_FOLD_SIZE)
    print(LSE6)
    #print(LSE5)
    data=Ask_Data(Year,Road_width,Location_Access,Governmentrate,road_type,Land_type)
    return predicting_value(data,LSE6)


    #ANOVA_CALC(test_data)


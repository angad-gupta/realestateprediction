#print("asdasd")
import os
import pandas as pd
import math
import numpy as np
#from . import views
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
from bokeh.plotting import figure, output_file, show
from bokeh.embed import components
from bokeh.models import HoverTool

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
def multiply1(list):
    temp=[]
    for i in range(0,len(list)):
        for j in range(i+1,len(list)):
            temp.append(list[i]*list[j])
    return temp

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
    temp=data
    temp1=[data[0],data[1],data[2]]

    temp2=[data[1],data[2],data[3]]
    #print(temp1,temp2)
    temp3=[]

    if (PLR_Check==1):
        for i in range(0,4):
            for i in range(2,ORDER):

                temp.append(data[i]**ORDER)
        for i in range(0,3):
            for j in range(i,3):
                temp3.append(float(temp1[i])*float(temp2[j]))
                temp.append(float(temp1[i])*float(temp2[j]))
        temp1=[temp3[0],temp3[1],temp[3]]

        temp.append(float(temp3[0])*float(data[2]))
        temp.append(float(temp3[1]) * float(data[3]))
        temp.append(float(temp3[1]) * float(data[3]))
        temp.append(float(temp3[0]) * float(data[2])* float(data[3]))




        data=temp
    return(data)


def predicting_value(data,LSE):
    sum=LSE[0]
    for i in range(0,len(data)):
        sum+=float(data[i])*LSE[i+1]
        #print(sum,data)
    return sum
# def LSE_cal(LSE_list,column,K_FOLD_SIZE):
#     LSE_avg_list = []
#     for i in range(0, column):
#         temp = 0
#         for j in range(0, K_FOLD_SIZE):
#             temp += (LSE_list[j][i])
#         LSE_avg_list.append(temp / K_FOLD_SIZE)
#     return LSE_avg_list


def multiply(a, b, lists,l):
    r, c = np.shape(a)
    #print(r,c)
    temp = np.zeros((l, c))
    # print(temp)
    r1,c1=np.shape(b)
    l=0
    for i in range(0, r):
        # print(i)
        for j in range(i,c1):
            print(i,j)

            for k in range(0,c):
                temp[l][k] = a[i][k] * b[k][j]
            l += 1

    lists.append(temp)
    #print(temp)
    # print(df.head())
    return lists, temp


def interaction_term(x):
    x = x.drop(
        ['Commercial-rate', 'Intercept', 'Earthen','Goreto', 'Pitch', 'Gravelled', 'paved', 'Commercial',
         'Residential'], axis=1)
    k = []

    # print(x.head)
    r, c = x.shape
    # list1=[]
   # print(x.head)
    df1 = x.drop(x.columns[c - 1], axis=1)
   # print(x.head)
    #print((np.transpose(df1)).head())
    temp1_matrix = df1.values
    ro, co = np.shape(temp1_matrix)
    #print(ro,co)
    l=6
    for i in range(0, c - 1):
        x= x.drop(x.columns[0], axis=1)
       # print(x.head())

        temp2_matrix = x.values
        list1, temp3 = multiply(np.transpose(temp1_matrix), temp2_matrix, k,l)
        ro, co = np.shape(temp3)
        # print(tot)
        # print(ro,co)
        # for j in range(0,ro):
        # list1.append(temp3[j-1])
        # print(list1)
        temp1_matrix = temp3

        #print(i)
        if(i==0):
            temp1_matrix = np.delete(temp1_matrix, [2,4,5], axis=0)
            l=3
            #print(temp1_matrix)
        if(i==1):
            temp1_matrix= np.delete(temp1_matrix, [1,2], axis=0)
            l=1

        #print(temp1_matrix)
        temp1_matrix = np.transpose(temp1_matrix)

    # print(i)
    # k+=1
    return list1


def interaction(list1, df):
    for i in range(0, len(list1)):
        for j in range(0, len(list1[i])):
            col = 'col'
            col = col + str(i) + str(j)
            #print(col)
            df2 = pd.DataFrame(list1[i][j], columns=[col])
            df = pd.concat([df, df2], axis=1)
    #print(df.head())
    return df


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
    df = df[['Year', 'Road_width', 'Location_Access',  'Governmentrate','Commercial-rate',
              'Earthen', 'Goreto','Pitch', 'Gravelled', 'paved', 'Commercial', 'Residential','Places']]
    # print(row,col)
    # row, col = df.shape
    # df=df[df.Places != 'Nuwakot']
    # df=df.reset_index(drop='TRUE')
    # #del df['index']

    df=df.drop(['Places'],axis=1)
    row,column=df.shape
    data = np.ones(row)
    #df = df.drop(['Commercial-price'], axis=1)
    df_one = pd.DataFrame({'Intercept': data})

    # print(df.isnull().any())

    # print(df.shape)
    # df=df.dropna()
    #
    # df=df.reset_index(drop='True')
    #
    # # print(df.shape)
    # for col in df.columns:
    #     df[col] = pd.to_numeric(df[col], errors='coerce')

    while (df.isnull().any().any()):
        for col in df.columns:

            if (df[col].isnull().any() and df[col].dtype == np.float64):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].mean(), inplace=True)
                # print(df[col])
    #df = Outliers(df)
    #df = var_transfromation(df)
    df = pd.concat([df_one, df], axis=1)
    df2=df
    #print(df)
    list1 = interaction_term(df)
    #df['log_commercial-price'] = np.log(df['Commercial-rate'])
    # print(df.corr())
    PLR_check = 0

    train_data, test_data ,K_FOLD_SIZE= Ten_fold(df2, row, col)

    df1 = powering(df)
    df1=interaction(list1, df1)


    #print(df.corr())
    row1,col1=df1.shape
    print(col1)
    train_data_p,test_data_P,K_FOLD_SIZE=Ten_fold(df1,row1,col1)
    #print(train_data_p[1].head())
    # print(train_data[9].head())
    if (float(Road_width)>80):
        Road_width="80"
    if(float(Location_Access)>2000 and float(Governmentrate)<60000):
        Location_Access="2000"
    if (float(Road_width)<=1 and float(Governmentrate)<60000 ):
        Road_width=1.5
        if(float(Governmentrate)<=28000):
            Governmentrate=28000
    inputted_data = Ask_Data(Year, Road_width, Location_Access, Governmentrate, road_type, Land_type, 0)
    check=0
    # print("For WLR")
    # LSE_list6, column6 ,k_train6,k_test6,R_sq,ar_sq,testda1,res_list_test1,res_list_train1,predicted_test1,predicted_train1,y_actual_test1,y_actual_train1,mse2= WLR(train_data, test_data, K_FOLD_SIZE, check)
    # # print()
    # print()
    # title = 'Residual vs Predicted price plot'
    # hover1 = HoverTool(tooltips=[
    #     ("(x,y)", "($x, $y)"),])
    #
    # hover2 = HoverTool(tooltips=[
    #     ("(x,y)", "($x, $y)"),
    # ])
    # plot1 = figure(title=title,
    #                x_axis_label="Predicted value",
    #                y_axis_label="residual",
    #                tools=[hover1, "pan,wheel_zoom,box_zoom,reset,save"],
    #                plot_width=500,
    #                plot_height=350,
    #                responsive=False,
    #                toolbar_location='below',
    #                logo=None)
    #
    # plot1.circle(predicted_test1[1], res_list_test1[1], line_width=1.5)
    # plot2 = figure(title="Predicted vs Actual price",
    #                x_axis_label="Actual Price",
    #                y_axis_label="Predicted value",
    #                tools=[hover2, "pan,wheel_zoom,box_zoom,reset,save"],
    #                plot_width=500,
    #                plot_height=350,
    #                responsive=False,
    #                toolbar_location='below',
    #                logo=None)
    # # show(plot)
    # plot2.circle(y_actual_test1[1], predicted_test1[1], line_width=1.5)
    # show((plot1))
    # show(plot2)
    # print("durbin-waston result:", durbin_watson(res_list_test1[2]))
    # print("tolerance:", 1 - ar_sq[2])
    # print("VIF:", 1 / (1 - ar_sq[2]))

    print("For MLR")
    LSE_list2,column,k1,k2,R_sq1,ar_sq1,testda,res_list_test,res_list_train,predicted_test,predicted_train,y_actual_test,y_actual_train,mse1=MLR(train_data,test_data,K_FOLD_SIZE,check)
    # print(),testda,res_list_test,res_list_train,predicted_test,predicted_train,y_actual_test,y_actual_train,mse1
    # print()
    # test=testda[0]
    # t_test=[]
    # # print("durbin-waston result:",durbin_watson(res_list_test[2]))
    # # print("tolerance:",1-ar_sq1[2])
    # # print("VIF:",1/(1-ar_sq1[2]))
    # # predicted_PLR=predicting_value(inputted_data,LSE_list2[2],0)
    # # print(predicted_PLR)
    # #
    # var_LSE6=mse1[0]*(np.linalg.inv(np.dot(test.T,test)).diagonal())
    # sd_b = np.sqrt(var_LSE6)
    # # for i in range(0,len(predicted_test[0])):
    # #     print(predicted_test[0][i])
    # # print()
    # # print()
    # # for i in range(0,len(y_actual_test[0])):
    # #     print(y_actual_test[0][i])
    # # print()
    # # print()
    # # for i in range(0,len(res_list_test[0])):
    # #     print(res_list_test[0][i])
    # #print(np.shape(sd_b))
    #
    # for i in range(0,len(LSE_list2[1])):
    #         t_test.append(LSE_list2[0][i] / sd_b[i])
    #
    # p_values = [2 * (1 - t.cdf(np.abs(i), (len(test) - 1))) for i in t_test]
    #
    # #print(l[1])
    # #print ("Features            coefficient_value           Standard error              t_test value                         p-values"   )
    # for i in range(0, len(LSE_list2[1])):
    #
    #     print(LSE_list2[1][i])
    #
    # print()
    # print()
    #
    # for i in range(0,len(LSE_list2[1])):
    #     print(sd_b[i])
    #
    # print()
    # print()
    #
    # for i in range(0, len(LSE_list2[1])):
    #     print(t_test[i])
    # print()
    # print()
    # #
    # for i in range(0, len(LSE_list2[1])):
    #     print(i,    p_values[i])

    # print("For MLR using Gradient Descent")
    # LSE_list1,column1,k_train1,k_test1=MLR_gr(train_data,test_data,K_FOLD_SIZE,check)
    # print()
    # print()#
    #
    #
    t_test=[]
    #print("For PLR")
    #LSE_list2,column2,k1,k2,testda,res_list_test,res_list_train,predicted_test,predicted_train,y_actual_test,y_actual_train,mse1=PLR(train_data_p,test_data_P,K_FOLD_SIZE,check)
    #graph()
    #PLR_Check=1
    inputted_data=Ask_Data(Year, Road_width, Location_Access, Governmentrate, road_type, Land_type,0)
    print(len(inputted_data))
    if (float(Governmentrate)>=1000000):
        LSE=LSE_list2[1]
    else:
        LSE=LSE_list2[0]
    print(len(LSE_list2[0]))

    # test=testda[0]
    # #predicted_PLR=predicting_value(inputted_data,LSE_list2[2],0)
    # #print(predicted_PLR)
    #
    # var_LSE6=mse1[0]*(np.linalg.inv(np.dot(test.T,test)).diagonal())
    # sd_b = np.sqrt(var_LSE6)
    #
    # #print(np.shape(sd_b))
    #
    # for i in range(0,len(LSE_list2[1])):
    #         t_test.append(LSE_list2[0][i] / sd_b[i])
    #
    # p_values = [2 * (1 - t.cdf(np.abs(i), (len(test) - 1))) for i in t_test]
    #
    # #print(l[1])
    # #print ("Features            coefficient_value           Standard error              t_test value                         p-values"   )
    # for i in range(0, len(LSE_list2[1])):
    #
    #     print(LSE_list2[0][i])
    #
    # print()
    # print()
    #
    # for i in range(0,len(LSE_list2[1])):
    #     print(sd_b[i])
    #
    # print()
    # print()
    #
    # for i in range(0, len(LSE_list2[1])):
    #     print(t_test[i])
    # print()
    # print()
    #
    # for i in range(0, len(LSE_list2[1])):
    #     print(i,    p_values[i])
    # # LSE=LSE_cal(LSE_list,column,K_FOLD_SIZE)
    # # LSE1=LSE_cal(LSE_list1,column1,K_FOLD_SIZE)
    # # LSE2=LSE_cal(LSE_list2,column2,K_FOLD_SIZE)
    # # print()
    # # print()
    #
    # #print(LSE_list5)
    # #LSE3 = LSE_cal(LSE_list3, column3, K_FOLD_SIZE)
    # #LSE4 = LSE_cal(LSE_list4, column1, K_FOLD_SIZE)
    # #LSE5 = LSE_cal(LSE_list5, column5, K_FOLD_SIZE)
    # #LSE6=LSE_cal(LSE_list6,column6,K_FOLD_SIZE)
    # #print(LSE6)
    # #print(len(LSE5))
    # #print(LSE5)
    # #data = Ask_Data(Year, Road_width, Location_Access, Governmentrate, road_type, Land_type, PLR_Check)

    return int(predicting_value(inputted_data,LSE)*100)*1000,k2,k1,res_list_test,res_list_train,predicted_test,predicted_train,y_actual_test,y_actual_train
  #ANOVA_CALC(test_data)

#realstate(5.6,10,100,100000,'Earthen','Commercial')
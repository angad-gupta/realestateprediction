from sklearn.linear_model import Lasso
from sklearn import metrics
import pandas as pd
from K_CROSS_VALIDATION import Ten_fold
import numpy as np
import math


location = "/home/DarKing/Desktop/Real_Estate_Project/rep/housing/static/dataq.csv"
#file_path = os.path.join(settings.STATIC_ROOT, 'dataq.csv')
df = pd.read_csv(location)
df = df[['Year', 'Road_width', 'Location_Access', 'Commercial-price', 'Governmentrate', 'Commercial-rate',
         'Earthen', 'Goreto', 'Pitch', 'Gravelled', 'paved', 'Commercial', 'Residential', 'Places']]
# print(row,col)
row, col = df.shape
df = df[df.Places != 'Nuwakot']
df = df.reset_index(drop='TRUE')
# del df['index']

df = df.drop(['Places'], axis=1)

#data = np.ones(row)
df = df.drop(['Commercial-price'], axis=1)
#df_one = pd.DataFrame({'Intercept': data})
row, column = df.shape
# print(df.isnull().any())

# print(df.shape)

# while (df.isnull().any().any()):
for col in df.columns:

    if (df[col].isnull().any() and df[col].dtype == np.float64):
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].mean(), inplace=True)
train_data,test_data,K_fold_size=Ten_fold(df,row,col)
#def model_selection():
#def model_selection(rmse,r_squared):


def Lassos(test_data_list,train_data_list,K_fold_size):
    rmse_list=[]
    r_squared_train_list=[]
    r_squared_list=[]
    for i in range(0,K_fold_size):

        test_data=test_data_list[i]
        train_data=train_data_list[i]
        y_test=test_data["Commercial-rate"]
        y_train=train_data["Commercial-rate"]
        y_train=y_train.values
        y_test=y_test.values
        test_data=test_data.drop(["Commercial-rate"],axis=1)
        train_data=train_data.drop(["Commercial-rate"],axis=1)
        reg=Lasso(alpha=0.5)
        reg=reg.fit(train_data,y_train)


        y_train_fitted=reg.predict(train_data)
        r_squared_train=reg.score(train_data,y_train)
        y_fitted=reg.predict(test_data)
        r_squared=reg.score(test_data,y_test)
        mse=metrics.mean_squared_error(y_test,y_fitted)
        rmse=math.sqrt(mse)
        mse_train=metrics.mean_squared_error(y_train,y_train_fitted)
        rmse_train=math.sqrt(mse_train)
        r_squared_list.append(r_squared)
        rmse_list.append(rmse)
        k=reg.coef_
        k1=reg.intercept_
        #r_s=metrics.r2_score(y_test,y_fitted)
        print(r_squared,mse,rmse,r_squared_train,rmse_train)
        print(k)
        print(k1)


Lassos(test_data,train_data,K_fold_size)
#def predict(data):

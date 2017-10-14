import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
#from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from K_CROSS_VALIDATION import Ten_fold
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

data = np.ones(row)
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
train_data,test_data=Ten_fold(df,row,col,0)
#def predict(data):

def Knn(test_data,train_data):

    y_test=test_data["Commercial-rate"]
    test_data=test_data.drop(["Commercial-rate"],axis=1)
    test_data=test_data.values
    y_test=y_test.values
    Y_train=train_data["Commercial-rate"]
    train_data=train_data.drop(["Commercial-rate"],axis=1)
    Y_train=Y_train.values
    train_data=train_data.values
    reg=KNeighborsRegressor(5,'distance').fit(train_data,Y_train)
        #print(reg)
    Y_pre=reg.predict(test_data)
        # print(df[col])
    mse=mean_squared_error(y_test,Y_pre)
    RMSE=math.sqrt(mse)
    R_s=reg.score(test_data,y_test)
    print(mse,R_s,RMSE)
#df = Outliers(df)
Knn(test_data,train_data)
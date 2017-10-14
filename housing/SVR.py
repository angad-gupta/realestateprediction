from sklearn.svm import SVR
from sklearn import metrics
import pandas as pd
from K_CROSS_VALIDATION import Ten_fold
import numpy as np
import math

def SVRs(test_data_l,train_data_l,k):
    for i in range(0,k):
        test_data = test_data_l[0]
        train_data = train_data_l[0]
        y_test = test_data["Commercial-rate"]
        test_data = test_data.drop(["Commercial-rate"], axis=1)
        test_data = test_data.values
        y_test = y_test.values
        Y_train = train_data["Commercial-rate"]
        train_data = train_data.drop(["Commercial-rate"], axis=1)
        Y_train = Y_train.values
        train_data = train_data.values
        clf =SVR(C=0.1,epsilon=0.2)
        clf.fit(train_data,Y_train)
        Y_pre=clf.predict(test_data)
        R_sq=clf.score(test_data,y_test)
        print(R_sq)


location = "static/dataq.csv"
#file_path = os.path.join(settings.STATIC_ROOT, 'dataq.csv')
df = pd.read_csv(location)
df = df[['Year', 'Road_width', 'Location_Access', 'Inflation_rate', 'Governmentrate', 'Commercial-rate',
         'Earthen', 'Goreto', 'Pitch', 'Gravelled', 'paved', 'Commercial', 'Residential', 'Kathmandu', 'Lalitpur']]
# print(row,col)
row, col = df.shape
# df = df[df.Places != 'Nuwakot']
# df = df.reset_index(drop='TRUE')
# # del df['index']
#
# df = df.drop(['Places'], axis=1)

data = np.ones(row)
#df = df.drop(['Commercial-price'], axis=1)
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
train_data,test_data,k=Ten_fold(df,row,col,1)

SVRs(test_data,train_data,k)

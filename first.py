####real state prediction
import os
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import metrics
import scipy.stats as stats


#import seaborn as sns


#location = r'/home/darknight/Desktop/real estate/train.csv'
#df = pd.read_csv(location) #df for dataframe
#df=df[['Id','MSSubClass','MSZoning','LotArea','Street','LandContour','OverallCond','OverallQual','YearBuilt','MoSold','YrSold','SalePrice',]]
#print (df.head())
#print (df.tail())
#iris=sns.load_dataset("df")
#sns.pairplot(iris)
#data_frame = pd.read_csv(r'/home/darknight/Desktop/real estate/test.csv')
#data_frame=data_frame[['Id','MSSubClass','MSZoning','LotArea','Street','LandContour','OverallCond','OverallQual','YearBuilt','MoSold','YrSold',]]*/
def handle_categorical_data(df):
	columns=df.columns.values
	for column in columns:
		text_digit_vals = {}
		def convert_to_int(val):
			return text_digit_vals[val]

		if df[column].dtype !=np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements =set(column_contents)
			x=0
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique]=x
					x+=1
			df[column]=list(map(convert_to_int,df[column]))
	return df
def explore_data(y):
	max_price = np.max(y)
	min_price=np.min(y)
	mean_price=np.mean(y)
	median_price=np.median(y)
	standard_deviation=np.std(y)
	print ("max SalePrice:",max_price)
	print ("min SalePrice:",min_price)
	print ("mean SalePrice:",mean_price)
	print ("median SalePrice:",median_price)
	print ("standard deviation for SalePrice:",standard_deviation)

def main():
	style.use('ggplot')
	location = r'/home/DarKing/Desktop/mp/dataq.csv'
	df = pd.read_csv(location) #df for dataframe
	#df=df[['Id','MSSubClass','MSZoning','LotArea','Street','LandContour','OverallCond','OverallQual','YearBuilt','MoSold','YrSold','SalePrice',]]
	#row,col=df.shape
	#print("No. of data:",row)
	#print("No. of Features:",col)
	#df= handle_categorical_data(df)
	#print (df.head())
	#print (df.tail())
	#iris=sns.load_dataset("df")
	#sns.pairplot(iris)
	#data_frame = pd.read_csv(r'/home/darknight/Desktop/real estate/test.csv')
	#data_frame=data_frame[['Id','MSSubClass','MSZoning','LotArea','Street','LandContour','OverallCond','OverallQual','YearBuilt','MoSold','YrSold',]]

	df = df[['Year', 'Road_width', 'Location_Access', 'Commercial-price', 'Latitude', 'Longitude', 'Commercial-rate',
              'Earthen', 'Pitch', 'Gravelled', 'paved', 'Commercial', 'Residential']]
	# print(row,col)
	row, col = df.shape
	data = np.ones(row)
	#for col in df.columns:
		#df.boxplot(column=col)
		#mp.show()

	# df = df.drop(['Commercial-rate'], axis=1)
	# #data=np.transpose(data)
	# #print(data)
	# #print(df.corr())
	# df['log_commercial-price']=np.log(df['Commercial-price '])
	# print(df['Commercial-rate'])
	#df_one = pd.DataFrame({'Intercept': data})

	# print(df.describe())
	# print(df.isnull().any())
	#df = pd.concat([df_one, df], axis=1)

	# while (df.isnull().any().any()):
	for col in df.columns:
		if (df[col].isnull().any() and df[col].dtype == np.float64):
			df[col].fillna(df[col].mean(), inplace=True)
		# print(col)
		else:
			# print(1)
			df[col] = pd.to_numeric(df[col], errors='coerce')
			df[col].fillna(df[col].mean(), inplace=True)
	#print(df.head())
	#data_frame = handle_categorical_data(data_frame)
	y=df['Commercial-price']
	#print (y)
	Y=np.array(y)
	#explore_data(Y)
	x=df.drop(['Commercial-price'],axis=1)
	#print (x)
	X=preprocessing.scale(x)
	#X=x

	x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.30, train_size=0.70)

	#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
	linear_reg = LinearRegression(n_jobs=10)
	linear_reg.fit(x_train,y_train)
	linear_coeff=linear_reg.coef_
	#print(linear_coeff)
	linear_intercept=linear_reg.intercept_
	#print(linear_coeff)
	#print(linear_coeff)
	r1,c1=x_test.shape
	r2=y_test.shape
	#print(r1,c1)
	#print(r2)

		#X=list(zip(x_train,linear_coeff))
	coefficients = pd.concat([pd.DataFrame(x.columns),pd.DataFrame(np.transpose(linear_coeff))],axis = 1)
	#data=pd.DataFrame(X,columns=['Features','estimated_coefficients'])
	#print(data)
	#print(coefficients)
	#X_explanatory_test_data=pd.concat([pd.DataFrame(x.columns),pd.DataFrame(x_test)],axis=0)
	#print(X_explanatory_test_data)
	#print(x_test.shape(),y_test.shape())
	accuracy=linear_reg.score(x_test,y_test)
	#print(x_test)
	prediction = linear_reg.predict(x_test)
	err_test=abs(prediction-y_test)
	k=len(prediction)
	predict_train=linear_reg.predict(x_train)
	err=abs(predict_train-y_train)
	#print(err)
	tot_err=np.dot(err,err)
	RMSE_train= math.sqrt((tot_err/len(y_train)))
	#print(RMSE_train)
	explained_var_sc=metrics.explained_variance_score(y_test,prediction)
	mean_abs_error=metrics.mean_absolute_error(y_test,prediction)
	mean_squared_Error=metrics.mean_squared_error(y_test,prediction)
	median_abs_error=metrics.median_absolute_error(y_test,prediction)
	r2_sc=metrics.r2_score(y_test,prediction)
	k=np.transpose(prediction)
	#print(k)
	print(explained_var_sc, mean_abs_error,mean_squared_Error,median_abs_error,r2_sc)
	print (accuracy)
	x_ex_data=pd.DataFrame(x_test)
	#print(x_ex_data)
	RMSE_test=math.sqrt(mean_squared_Error/len(y_test))
	print(RMSE_test)
	#print(k)
	#r1,c1=(x_ex_data.iloc[:,3]).shape
	#print(r1,c1)
	#print(len(x_ex_data.iloc[:,:3]),len(prediction),len(y_test))
	print(prediction)
	#print(np.transpose(x_test[3]),len(np.transpose(x_test[3])))
	#plt.scatter(x_ex_data.iloc[:,:3],k,color='blue')
	#plt.scatter(np.array(x_ex_data.iloc[:,3]),prediction,color='red')
	##plt.xlabel('LotArea')
	#plt.ylabel('predicted')
	#plt.legend()
	#df['predict'].plot()
	#plt.legend(loc=4)
	#plt.xlabel('LotArea')
	#plt.ylabel('prediction')
	#plt.figure()
	#plt.scatter(np.array(x_ex_data.iloc[:,8]),prediction,color='red')
	#plt.figure()
	#plt.scatter(np.array(x_ex_data.iloc[:,10]),prediction,color='red')
	#plt.figure()
	#plt.scatter(np.array(x_ex_data.iloc[:,9]),prediction,color='red')
	#plt.figure()
	#plt.boxplot(err_test,0,'gd')
	#plt.show()
	##plt.scatter(df.LotArea,df.SalePrice)
	#plt.xlabel('LotArea')
	#plt.ylabel('SalePrice')
	#plt.show()
	#plt.scatter()

	
if __name__ == "__main__":
    main()





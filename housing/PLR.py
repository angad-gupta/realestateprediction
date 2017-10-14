import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as mp
from normalization import Normalization
from denormalization import denormalizaton
from Anova_calc  import ANOVA_CALC
from SSR_calc  import SSR_Calc
from LSE_CALC  import LSE_Calculation
from  Prediction import prediction

def PLR(training_data,test_data,K_FOLD_SIZE,check):
	SSR_list = []
	SSE_list = []
	SST_list = []
	# ro=0
	MSR_list = []
	MSE_list = []
	R_sqaured_list = []
	F_list = []
	Adjusted_R_squared_list = []
	SSR_list1 = []
	SSE_list1 = []
	SST_list1 = []
	# ro=0
	MSR_list1 = []
	MSE_list1 = []
	R_sqaured_list1 = []
	F_list1 = []
	Adjusted_R_squared_list1 = []
	LSE_list = []

	for i in range(0,K_FOLD_SIZE):
		LSE, Y_actual_, data = LSE_Calculation(training_data[i],check)
		LSE_list.append(LSE)
		row, column = np.shape(data)
		Y_pre = np.matmul(data, LSE)

		if (check==1):
			Y_pre = np.power(Y_pre, 10)
		res = prediction(Y_pre, Y_actual_)
		# print(Y_pre)
		# print(Y_pre)
		# print(Y_pre)
		# print(res)
		# print(predict(res, Y_actual_))
		# mp.scatter(Y_pre, res, marker='o', c='b')
		# mp.title('Predicted VS residual')
		# mp.xlabel('Y_pre')
		# mp.ylabel('res')
		# mp.show()
		# mp.scatter(Y_pre, Y_actual, marker='o', c='b')
		# mp.title('Predicted VS actual')
		# mp.xlabel('Y_pre')
		# mp.ylabel('Y_actual')
		# mp.show()

		SSE = np.matmul(res, np.transpose(res))
		# print(SSE)#sum of squares of residual
		SSE_list1.append(SSE)
		# print(SSE)
		SSR = SSR_Calc(Y_actual_, Y_pre, row)

		SSR_list1.append(SSR)

		# temp_SSM=np.subtract(Y_pre,Y_avg)
		# SSM=abs(temp_SSM)*abs(temp_SSM)
		# print(SSM)
		SST = SSR + SSE
		SST_list1.append(SST)

		# ro=ro+row
		R_sqaured = SSR / SST
		# Adjusted_R_squared_list.append(Adjusted_R_squared)
		R_sqaured_list1.append(R_sqaured)

		# print(R_sqaured)
		MSE = SSE / ((row - column - 1))
		MSE_list1.append(MSE)

		MSR = SSR / (column)
		MSR_list1.append(MSR)
		MST = SST / float(row - 1)
		Adjusted_R_squared = 1 - MSE / MST
		Adjusted_R_squared_list1.append(Adjusted_R_squared)
		# MST_list.append(MST)
		F = MSR / MSE
		F_list1.append(F)
		# print("For Training Data")


		# data = teat_data[i].drop(['log_commercial-price'], axis=1)
		# print(data)
		data = test_data[i].drop(['Commercial-rate'], axis=1)
		if check == 1:
			data = data.drop(['log_commercial_rate'], axis=1)
		row, column = data.shape
		# print(data)
		# print(row, column)
		Y_actual_ = test_data[i]['Commercial-rate']
		# print(Y_actual_)
		Y_actual_ = Y_actual_.values
		weight = np.identity(row)
		# Y_actual = x[i]['log_commercial-price']
		# Y_actual = Y_actual.values
		# print(Y_actual)
		# norm, data = Normalization(data)
		# print(Y_actual)
		# print(data)
		Y_pre = np.matmul(data, LSE)

		if (check == 1):
			Y_pre = np.power(Y_pre, 10)
		res = prediction(Y_pre, Y_actual_)
		# Y_pre = np.power(Y_pre, 10)
		# print(Y_pre)
		# print(Y_pre)
		# print(Y_pre)
		# print(res)
		# print(predict(res, Y_actual_))
		# mp.scatter(Y_pre, res, marker='o', c='b')
		# mp.title('Predicted VS residual')
		# mp.xlabel('Y_pre')
		# mp.ylabel('res')
		# mp.show()
		# mp.scatter(Y_pre, Y_actual, marker='o', c='b')
		# mp.title('Predicted VS actual')
		# mp.xlabel('Y_pre')
		# mp.ylabel('Y_actual')
		# mp.show()

		SSE = np.matmul(res, np.transpose(res))
		# print(SSE)#sum of squares of residual
		SSE_list.append(SSE)
		# print(SSE)
		SSR = SSR_Calc(Y_actual_, Y_pre, row)

		SSR_list.append(SSR)

		# temp_SSM=np.subtract(Y_pre,Y_avg)
		# SSM=abs(temp_SSM)*abs(temp_SSM)
		# print(SSM)
		SST = SSR + SSE
		SST_list.append(SST)

		# ro=ro+row
		R_sqaured = SSR / SST
		# Adjusted_R_squared_list.append(Adjusted_R_squared)
		R_sqaured_list.append(R_sqaured)

		# print(R_sqaured)
		MSE = SSE / ((row - column - 1))
		MSE_list.append(MSE)

		MSR = SSR / (column)
		MSR_list.append(MSR)
		MST = SST / float(row - 1)
		Adjusted_R_squared = 1 - MSE / MST
		Adjusted_R_squared_list.append(Adjusted_R_squared)
		# MST_list.append(MST)
		F = MSR / MSE
		F_list.append(F)

	print("For Training Data")
	ANOVA_CALC(SSE_list1, SST_list1, SSR_list1, R_sqaured_list1, MSE_list1, MSR_list1, F_list1,
			   Adjusted_R_squared_list1,K_FOLD_SIZE)

	print("For Test Data")
	ANOVA_CALC(SSE_list, SST_list, SSR_list, R_sqaured_list, MSE_list, MSR_list, F_list, Adjusted_R_squared_list,K_FOLD_SIZE)
	return LSE_list, column





import pandas as pd
import math
import numpy as np
from  normalization  import Normalization
from denormalization import denormalizaton



def gradient_descent(data1,norm,Y_actual_,check):
	#norm, data = Normalization(data)
	data = data1.values
	row, col = data.shape
	#print(col)
	#print(len(norm))
	Learning_Rate = 0.0000000000000001
	b = np.random.random(col)
	#print (b)
	#print(row,col)
	data = denormalizaton(data, norm)
	err=np.zeros(col)
	#b= np.ones(col)
	#print(err)
	err=0.01
	k=0
	Y_pre=np.matmul(data,b)
	if(check==1):
		Y_pre = np.power(Y_pre, 10)
		Learning_Rate=0.0000000000000000000000000001
	res=np.subtract(Y_actual_,Y_pre)
	# print(np.shape(res))
	SSE = np.transpose(res) * res
	#print(SSE)
	while (True):
		norm,data=Normalization(data1)
		data=data.values
		b_new=np.matmul(np.transpose(res),data)
		b_new=np.multiply(b_new,Learning_Rate/row)
		#print(np.shape(b_new))
		b_new=np.subtract(b,b_new)
		#print(b_new)
		b=b_new
		data = denormalizaton(data, norm)
		Y_pre = np.matmul(data, b_new)
		if (check == 1):
			Y_pre = np.power(Y_pre, 10)

		res = np.subtract(Y_actual_, Y_pre)
		J_theta=np.transpose(res)*res
		if(abs(J_theta-SSE).all()<=err):
			return b_new,data
		SSE=J_theta


		#print(data)
		#print(np.shape(SSE))
		k+=1
		if(k>=10000):
			print(b_new)
			return b_new,data
			#print(b_new)


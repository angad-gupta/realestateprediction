import numpy as np
from gradient_Decent  import gradient_descent
from  normalization  import Normalization

def LSE_Calculation(x,check):
    # data = x[i].drop(['log_commercial-price'], axis=1)
    # print(data)
	if (check == 1):
		data = x.drop(['log_commercial_rate'], axis=1)
		data = data.drop(['Commercial-rate'], axis=1)
		Y_actual = x['log_commercial_rate']
		Y_actual_ = x['Commercial-rate']
		row, column = data.shape
		Y_actual_ = Y_actual_.values
		Y_actual = Y_actual.values
		norm, data1 = Normalization(data)
		b, data = gradient_descent(data, norm,Y_actual_,check)
		return b, Y_actual_, data1, norm

	data1 = x.drop(['Commercial-rate'], axis=1)
	row, column = data1.shape
	column = column
	# print(data)
	Y_actual_=x['Commercial-rate']
	#print(row, column)
	Y_actual_ = Y_actual_.values
	weight = np.identity(row)
	# Y_actual = x[i]['log_commercial-price']
	# Y_actual=Y_actual.values
	norm, data1 = Normalization(data1)
	#print(len(norm))
	b, data = gradient_descent(data1, norm, Y_actual_,check)
	return b, Y_actual_, data1,norm



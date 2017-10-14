import numpy as np

def LSE_Calculation(x,check):
    # data = x[i].drop(['log_commercial-price'], axis=1)
    # print(data)
    # if (check==1):
    #     data=x.drop(['log_commercial_rate'],axis=1)
    #     data=data.drop(['Commercial-rate'],axis=1)
    #     Y_actual=x['log_commercial_rate']
    #     Y_actual_=x['Commercial-rate']
    #     row,column=data.shape
    #     #print(data.head())
    #
    #     #print(np.shape(weight))
    #     Y_actual_=Y_actual_.values
    #     Y_actual=Y_actual.values
    #     data=data.values
    #     data_transpose = np.transpose(data)
    #     #data_transpose = np.matmul(data_transpose, weight)
    #     #print(np.shape(data_transpose))
    #     mat_product_temp1 = np.matmul(data_transpose, data)
    #     # print(mat_product_temp1)
    #     mat_product_temp1_inv = np.linalg.inv(mat_product_temp1)
    #     # print(mat_product_temp1_inv)
    #     mat_product_temp2 = np.matmul(mat_product_temp1_inv, data_transpose)
    #     LSE = np.matmul(mat_product_temp2, Y_actual_)
    #     # data = denormalizaton(data, norm)
    #     return LSE, Y_actual_, data


    data = x.drop(['Commercial-rate'], axis=1)
    row, column = data.shape
    #column = column - 1
    # print(data)
    # print(row, column)
    Y_actual_ = x['Commercial-rate']
    # print(Y_actual_)
    Y_actual_ = Y_actual_.values
    weight = np.identity(row)
    # Y_actual = x[i]['log_commercial-price']
    # Y_actual = Y_actual.values
    # print(Y_actual)
    # norm, data = Normalization(data)
    # print(Y_actual)
    # print(data)
    data = data.values

    data_transpose = np.transpose(data)
    mat_product_temp1 = np.matmul(data_transpose, data)
    # print(mat_product_temp1)
    mat_product_temp1_inv = np.linalg.inv(mat_product_temp1)
    # print(mat_product_temp1_inv)
    mat_product_temp2 = np.matmul(mat_product_temp1_inv, data_transpose)
    LSE = np.matmul(mat_product_temp2, Y_actual_)
    # data = denormalizaton(data, norm)
    # print((LSE))

    return LSE,Y_actual_,data
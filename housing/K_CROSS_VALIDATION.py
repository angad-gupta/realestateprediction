import numpy as np
import math
K_FOLD_SIZE=10
def Ten_fold(data, row, col,k):
    data_size = math.ceil(row /K_FOLD_SIZE)
    # print(data_size)
    K = 0
    temp_data_size = data_size
    train_datalist = []
    test_datalist = []
    # data splitting
    if k==0:
        data_size = math.ceil(row*0.8)
        df = data.iloc[K:temp_data_size, :]
        # print (df)
        test_data=data.iloc[K:temp_data_size, :]
        train_data=(data.drop(data.index[K:temp_data_size]))
        #K = temp_data_size
        #temp_data_size = data_size + K
        return train_data,test_data

    for i in range(0, K_FOLD_SIZE):
        df = data.iloc[K:temp_data_size, :]
        # print (df)
        test_datalist.append(data.iloc[K:temp_data_size, :])
        train_datalist.append(data.drop(data.index[K:temp_data_size]))
        K = temp_data_size
        temp_data_size = data_size + K
    return train_datalist, test_datalist,K_FOLD_SIZE
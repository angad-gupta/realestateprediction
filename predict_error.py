import numpy as np

def predict(res, Y):
    sum = 0
    #print(res)
    #print(Y)
    k = np.divide(res, Y)
    # print(k)
    for i in range(0, len(k)):
        sum = sum + abs(k[i])
    #print(sum)
    error = sum / len(k)
    # print(error)
    return (error * 100)
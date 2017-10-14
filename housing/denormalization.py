import pandas as pd
import numpy as np


def denormalizaton(x,norm):
    #mean=np.mean(x)
    #max=np.max(x)
    #min=np.std(x)
    #print(len(norm))
    row,col=np.shape(x)
    #print(len(norm))
    #print(row,col)
    #print(norm[1][1])
    #print(x)
    #print(x[3][0])
    #
    for i in range (0,col-1):
        #print(i)
        for j in range(0,row):
            #print(j)
            #print(x[i][j])
            x[j][i+1]=x[j][i+1]*norm[i][0]+norm[i][1]
            #print(x[i][j])
        #print(x[i])
   # print(x)
   # print(row, col)
    return x



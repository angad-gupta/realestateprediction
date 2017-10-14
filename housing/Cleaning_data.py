import pandas as pd
import matplotlib.pyplot as mp
def Outliers(df):
    row,cols=df.shape
    #print(row,col)
    for col in df.columns:
        q1=df[col].quantile(.25)
        #print(col,q1)
        q3=df[col].quantile(.75)
        IQR=q3-q1
        max=(q3+1.5*IQR)

        min = (q1 - 1.5 * IQR)

        mean=df[col].mean()
        #print(max, min,mean)
        if col == 'Year' or col == 'Road_width' or col == 'Location_Access' or col == 'Latitude' or col == 'Longitude' or col == 'Area':
            for i in range(0, row):
                # print(df.loc[i,col])
                if (df.loc[i, col] > max or df.loc[i, col] < min):
                    #print(df.loc[i, col])
                    df.loc[i, col] = mean
                    #print(df.loc[i, col])


        #mp.boxplot(df[col])
    return (df)  #mp.show()






        #df=(df[col].all() <= (q1-1.5*IQR)) and (df[col].all()>=(q3-1.5*IQR))
        #print(df)
    #print(df)
      #print(df[col].all())





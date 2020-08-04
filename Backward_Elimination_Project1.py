
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('iq_size.csv')

features=df.iloc[:,1:4].values
labels = df.iloc[:,0:1].values

df.isnull().sum()

import statsmodels.regression.linear_model as sm
features= np.append(arr=np.ones((38,1)).astype(int),values=features,axis=1)
features_obj=features[:,[0,1,2,3]]

while True:
    labels=labels.astype(float)
    features_obj=features_obj.astype(float)
    ols=sm.OLS(endog = labels , exog=features_obj).fit()
    p_values=ols.pvalues
    if p_values.max()> 0.05:
        features_obj = np.delete(features_obj, p_values.argmax(), 1)
    else:
        break
    
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train, labels_test = train_test_split(features_obj,labels,test_size=0.20,random_state=0)    

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train,labels_train)


regressor.predict([[93]])





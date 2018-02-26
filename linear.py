# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:35:31 2018

@author: siddharth
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('newwomen.csv')
X = dataset.iloc[:,0].values
Y = dataset.iloc[:,-1].values

off = np.ones(X.size)

M = np.c_[off,X]
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(M,Y)

yPred = regressor.predict(M)
print(yPred,Y)
print(yPred)
k = open('womenPredict.csv','w')

for x,y in zip(yPred,X):
    k.write(str(y)+","+str(x) + "\n")
k.close()
print(M*regressor.coef_)

plt.scatter(yPred, X, color = 'red')
plt.title('race against time')
plt.xlabel('Year')
plt.ylabel('Time')
plt.show()
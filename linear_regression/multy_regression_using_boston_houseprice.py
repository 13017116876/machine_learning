"""
这里是利用sklearn中的波士顿房价做多元线性回归
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model

x,y = datasets.load_boston(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=666)
model = linear_model.LinearRegression()
model.fit(x,y)
print(model.coef_)
print(model.intercept_)

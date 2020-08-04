"""
利用sklearn进行lasso回归
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
data_size = 3000
x = np.random.normal(1.0,10.0,data_size)
y = 2.6*x+3.7+np.random.normal(0,1,data_size)
x = x.reshape(-1,1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
model = linear_model.Lasso()
model.fit(x_train,y_train)
print(model.coef_)
print(model.intercept_)

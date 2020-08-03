"""
这里是使用sklearn实现一元线性回归
1秒就可以输出w和b，之前手写里面因为每一步都计算损失函数，所以很慢
"""

from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
data_size = 3000
x= np.random.uniform(low=1.0,high=10.0,size=data_size)
y = 2.6*x+3.7+np.random.normal(0,1,3000)
x = x.reshape(-1,1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
model = linear_model.LinearRegression()
model.fit(x_train,y_train)
print(model.coef_)
print(model.intercept_)

import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

np.random.seed(272)
data_size = 100
x= np.random.uniform(low=1.0,high=10.0,size=data_size)
y = 2.6*x+3.7+np.random.normal(0,1,100)
print(x)
print(y)
plt.scatter(x,y)
plt.show()
model = LinearRegression(100)
model.train(x,y)
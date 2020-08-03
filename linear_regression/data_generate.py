import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

# np.random.seed(272)
data_size = 3000
x= np.random.uniform(low=1.0,high=10.0,size=data_size)
y = 2.6*x+3.7+np.random.normal(0,1,3000)
# print(x)
# print(y)
# plt.scatter(x,y)
# plt.show()
model = LinearRegression(5000)
w,b = model.train(x,y)
print(w)
print(b)
plt.scatter(x,y)
x1 = np.sort(x)
plt.plot(x1,w*x1+b)
plt.show()
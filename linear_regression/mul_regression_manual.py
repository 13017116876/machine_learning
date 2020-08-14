# 实现手写多元线性回归，刚开始效果不好，是因为迭代次数设置100太少了，后来50000发现效果不错
#之前LR设置过大，造成损失越来越大，后来调小了LR就好了

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data = np.genfromtxt(r"D:\machine_learning\linear_regression\Advertising.csv",delimiter=",")
x0 = np.random.normal(1,10,200).reshape(200,-1)
x1 =np.random.normal(1,10,200).reshape(200,-1)
y = 3.7*x0 + 2.6*x1 + 1.5 
# print(y)
y = y.reshape(200)+np.random.normal(0.0,1,200)
print(y)
# print(y)
x = np.concatenate((x0,x1),axis=1)
# print(x.shape)
# print(y.shape)
# x = data[1:,1:4]
# y = data[1:,4]
# print(x.shape)
# print(y.shape)
# exit()
ITER = 50000
LR = 0.000001


class MultiLinearRegression(object):
    def __init__(self):
        self.iter = ITER
        self.lr = LR

    def cal_grad(self):
        d_w = []
        d_b = []
        pre = np.sum(self.w*self.x,axis=1) + self.b
        error = self.y-pre
        # for i in range(self.w):
        d_w = (np.mean(error*self.x.T,axis=1))*-1
        d_b = (np.mean(error))*-1
        return d_w,d_b

    def step(self):
        d_w,d_b = self.cal_grad()
        # print(self.w)
        # print(self.b)
        self.w = self.w - self.lr*d_w
        self.b = self.b - self.lr*d_b


    def cal_loss(self):
        # print((self.y - np.sum(self.w*self.x,axis=1)+self.b)**2)
        # print(math.sqrt(np.sum((self.y - np.sum(self.w*self.x,axis=1)+self.b)**2)))
        loss = math.sqrt(np.sum((self.y - (np.sum(self.w*self.x,axis=1)+self.b))**2))/len(self.y)
        return loss

    def train(self,x,y):
        self.x = x
        self.y = y
        x_shape = x.shape[1]
        self.w = np.random.normal(0,1,x_shape)
        self.b = np.random.normal(0,1,1)
        for i in range(self.iter):
            self.step()
            loss = self.cal_loss()
            print(loss)
        return self.w,self.b
        
        

model = MultiLinearRegression()
w,b = model.train(x,y)
print(w)
print(b)

# 下面是画图
ax = plt.figure().add_subplot(111,projection='3d')
ax.scatter(x[:,0],x[:,1],y,c='r',marker='o',s=100)
x0 = x[:,0]
x1= x[:,1]
x0,x1 = np.meshgrid(x0,x1)
z = w[0] *x0 + w[1]*x1 + b
ax.plot_surface(x0,x1,z)
ax.set_xlabel('x0')
ax.set_xlabel('x1')
ax.set_xlabel('z')
plt.show()



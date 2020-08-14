
# 这个是手动实现的逻辑回归，LR这里设置0.1收敛较快，而前面的多元线性回归 设置大了就不行，回头要看看LR的选取规则
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
ITER = 1000
ALPHA = 0.3
LR = 0.1
class LogisticRegression(object):
    def __init__(self):
        self.w = np.random.normal(1,0.1,2)
        self.b = np.random.normal(0,1,1)
        self.iter = ITER
        self.alpha = ALPHA
        self.lr = LR

    def train(self,x,y):
        self.x = x
        self.y = y
        self.num = x.shape[1]
        for i in range(self.iter):
            self.step()
            self.loss()

    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    def cal_grad(self):
        # print(np.exp((np.dot(self.w,self.x)+self.b)))
        d_w = 1.0/self.num*(np.dot(self.x,(self.sigmoid(np.dot(self.w,self.x)+self.b)-self.y)))
        d_b = 1.0/self.num*(np.sum(self.sigmoid(np.dot(self.w,self.x)+self.b)-self.y))
        return d_w,d_b
        
                          
    def step(self):
        d_w,d_b = self.cal_grad()
        self.w = self.w - self.lr*d_w
        self.b = self.b - self.lr*d_b

    def loss(self):
        h = self.sigmoid(np.dot(self.w,self.x)+self.b)
        # print(h)
        loss = -1/self.num*((np.dot(self.y.T,np.log(h)) + np.dot(1-self.y.T,np.log(1-h))))
        print(loss)


def load_data(file):
    data = pd.read_table(file,encoding="utf-8",header=None)
    X = data.iloc[:,0:2]
    Y = data.iloc[:,2]
    return X.T,Y

def plot_graph(x,y,w,b):
    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(x[i][0],x[i][1], color="red",s=50)
        else:
            plt.scatter(x[i][0],x[i][1],color="green",s=50)
    # print(x.iloc[0,:].min())
    # hSpots = np.linspace(x.iloc[0,:].min(), x.iloc[0,:].max(), 100)
    # vSpots = -(b[0] + w[0]*hSpots)/w[1]
    # plt.plot(hSpots,vSpots,color="red")
    plt.show()

x,y = load_data(r"D:\machine_learning\logicstic_regression\data\non_linear.txt")
model = LogisticRegression()
model.train(x,y)
plot_graph(x,y,model.w,model.b)
print(model.w)
print(model.b)



        
    

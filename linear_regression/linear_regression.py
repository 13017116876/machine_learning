"""
这个是手写一元线性回归的
流程：
    1.准备数据、参数、迭代次数
    2.迭代，将数据输入到模型
    3.计算该点的损失函数梯度
    4.调整参数
    5.计算损失
    6.进行下次迭代
"""
import numpy as np
import math
class LinearRegression(object)

    def init(self,iter):        
        self.w = np.random.normal(1,0.1)
        self.b = np.random.normal(1,0.1) # 创建b
        self.iter = iter
        self.lr = 0.01
        self.loss_arr = []

    def train(self,x,y):
        """
        self.train_data = train_data #[1,2,3]
        self.train_lable = train_label#[4,5,6]
        """
        if len(x) != len(y):
            raise "data length is not equal with label length"
        self.x=x # x的特征数据
        self.y=y # y标签
        for i in self.iter: # 遍历每一次迭代
            self.step() # 迭代
            self.loss_arr.append(self.loss()) # 添加损失函数值

    def f(self,x):
        return x *self.w+self.b

    def predict(self):
        y_pred = self.f(self.x)
        return y_pred
        
    def mean_reduce(self,pre,label):
        if len(pre) != len(label):
            raise "pre length is not equal label length"
        sum = 0
        for i in range(len(pre)):
            sum+=math.sqrt((pre[i] - label[i])^2)
        return sum
                 
    def loss(self):
        loss_value = mean_reduce(self.predict(),self.y)
        return loss

    def cal_gradient(self):
        d_w = np.mean((self.w*self.x+self.b-self.y)) * self.x) #这里是用全部元素求损失函数
        d_b = np.mean((self.w*self.x+self.b-self.y))
        return d_w,d_b

    def step(self):
        d_w,d_b = self.cal_gradient()
        self.w = self.w - self.lr*d_w # 调整w
        self.b = self.b - self.lr*d_b # 调整b
        return self.w, self.b


    

    

    
        
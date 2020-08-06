import matplotlib.pyplot as plt
import numpy as np
ITER = 100
ALPHA = 0.3
LR = 0.03
class LogisticRegression(object):
    def __init__(self,iter):
        self.w = np.random.normal(1,0.1,2)
        self.b = np.random.normal(1.0.1)
        self.iter = iter
        self.alpha = ALPHA
        self.lr = LR

    def train(self,x,y,label):
        self.x = x
        self.y = y
        for i in range(self.iter):
            self.step()

    def cal_grad():
        
                          
    def step(x,y)
        d_w,d_b = self.cal_grad()
        self.w = self.w - self.lr*d_w
        self.b = self.b - self.lr*d_b


def load_data(file):
    x = []
    y = []
    label = []
    with open(file) as f:
        for each in f:
            x.append(float(each.split()[0]))
            y.append(float(each.split()[1]))
            label.append(each.split()[2])
    return np.array(x),np.array(y),np.array(label)

def plot_graph(x,y,label):
    for i in range(len(label)):
        if label[i] == "1":
            plt.scatter(x[i],y[i], color="red",s=50)
        else:
            plt.scatter(x[i],y[i],color="green",s=50)
    plt.show()

x,y,label = load_data(r"D:\machine_learning\logicstic_regression\data\linear.txt")
plot_graph(x,y,label)


# plt.scatter(x[:20],y[:20], c=label[:20],s=50,cmap='viridis')
# plt.xlabel("x")
# plt.ylabel("y")

        
    

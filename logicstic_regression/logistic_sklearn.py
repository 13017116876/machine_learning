from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def load_data(file):
    data = np.loadtxt(file,delimiter=",")
    X = data[:,:2]
    Y = data[:,2]
    return X,Y

def plot_decision_boundary(model,axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 20)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 20)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False
X,Y = load_data(r"D:\machine_learning\logicstic_regression\sklearn_data\data1")
x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=666)
model = LogisticRegression()# 这里面有默认的训练次数和学习率
model.fit(x_train,y_train)
plot_decision_boundary(model, axis=[0, 100, 0, 100])
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], color='red')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], color='blue')
plt.xlabel('成绩1')
plt.ylabel('成绩2')
plt.title('两门课程成绩与是否录取的关系')
plt.show()
print(model.score(x_train,y_train))
print(model.score(x_test,y_test))
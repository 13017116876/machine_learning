import pandas as pd
import math

def data_load(file):
    data = pd.read_table(file, encoding="utf-8",delimiter=" ")
    X_data = data.iloc[:,0:4]
    Y_data = data.iloc[:,4]
    return X_data,Y_data

def cal_shanong(y):
    entropy = 0
    cato = list(set(y))
    # print(cato)
    for i in range(len(cato)):
        va = len(y[y==cato[i]])/len(y)
        entropy += -(va) * math.log(va)
    return entropy

def cal_condition_entropy(x,y):
    condition_entropy = 0
    cato = list(set(x))
    for i in range(len(cato)):
        va = len(x[x==cato[i]])/len(x)
        y_condition = y[x[x==cato[i]].index.tolist()]
        condition_entropy += va * cal_shanong(y_condition)
    return condition_entropy

def cal_info_gain(x,y):
    dic = {}
    init_entropy = cal_shanong(y)
    for i in range(x.shape[1]):
        condition_x = x.iloc[:,i]
        condition_entropy = cal_condition_entropy(condition_x,y)
        dic[condition_x.name] = condition_entropy - init_entropy
    print(dic)


x, y = data_load("D:\machine_learning\decision_tree\play_tennis")
info_gain_dic = cal_info_gain(x,y)
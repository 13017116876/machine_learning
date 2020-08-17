# 手写决策树
import pandas as pd
import math
dic_result = {}
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
    # 计算熵
    init_entropy = cal_shanong(y)
    # 这里判断是否已经全部是一种结果
    if init_entropy == 0:
        return "empty"
    # 遍历x的每一列
    for i in range(x.shape[1]):
        condition_x = x.iloc[:,i]
        # 计算每一列的条件熵
        condition_entropy = cal_condition_entropy(condition_x,y)
        # 得到每一列信息增益
        dic[condition_x.name] = init_entropy - condition_entropy
    # print(dic)
    new_dic = dict(sorted(dic.items(), key=lambda x:x[1], reverse=True))
    # 得到信息熵最大的那一列名
    max_info = list(new_dic.keys())[0]
    return max_info

def create_dic1(x,y):
    dic = {}

    max_info = cal_info_gain(x,y)
    dic[max_info] = {}
    # 获取当前列有几个分类
    li = list(set(x[max_info]))
    # print(dic_result)
    # 遍历每一个分类
    for i in range(len(li)):
        # 获取当前的最大增益，因为此时当前的分类为0，所以不可能是当前的分类
        max1 = cal_info_gain(x[x[max_info]==li[i]], y[x[max_info][x[max_info]==li[i]].index.tolist()])
        if max1 == "empty":
            dic[max_info][li[i]] = list(set(y[x[max_info][x[max_info]==li[i]].index.tolist()]))[0]
        else:
            dic[max_info][li[i]] = create_dic1(x[x[max_info]==li[i]],y[x[max_info][x[max_info]==li[i]].index.tolist()])
        # print(y[x[max_info][x[max_info]==li[i]].index.tolist()])
    return dic

def create_dic(x, y):
    # 获取当前列的最大信息增益列
    max_info = cal_info_gain(x,y)
    # 获取当前列有几个分类
    li = list(set(x[max_info]))
    dic_result[max_info] = {}
    print(dic_result)
    # 遍历每一个分类
    for i in range(len(li)):
        # 获取当前的最大增益，因为此时当前的分类为0，所以不可能是当前的分类
        max1 = cal_info_gain(x[x[max_info]==li[i]], y[x[max_info][x[max_info]==li[i]].index.tolist()])
        if max1 == "empty":
            dic_result[max_info][li[i]] = list(set(y[x[max_info][x[max_info]==li[i]].index.tolist()]))[0]
        else:
            dic_result[max_info][li[i]] = create_dic1(x[x[max_info]==li[i]],y[x[max_info][x[max_info]==li[i]].index.tolist()])
        # print(y[x[max_info][x[max_info]==li[i]].index.tolist()])
    return dic_result

def tree_plot(di):
    pass

x, y = data_load("D:\machine_learning\decision_tree\play_tennis")
# max_info = cal_info_gain(x,y)
dic = create_dic(x,y) # 得到决策树字典，接下来就是画图，先放下了



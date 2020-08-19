# 手写决策树  也写了预测  没有画图,只做了全部分类之后才停止,ID3本身算法就没有剪枝操作
import pandas as pd
import math
from collections import Counter
dic_result = {}
def data_load(file):
    data = pd.read_table(file, encoding="utf-8",delimiter=" ")
    X_data = data.iloc[:,0:1]
    Y_data = data.iloc[:,1]
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
    # 初始化一个字典
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
        if x[x[max_info]==li[i]].shape[1] == 1:
            a = Counter(list(y[x[max_info][x[max_info]==li[i]].index.tolist()]))
            dic[max_info][li[i]] = a.most_common()[0][0]
        else:
            # 获取当前的最大增益，因为此时当前的分类为0，所以不可能是当前的分类
            max1 = cal_info_gain(x[x[max_info]==li[i]].drop(max_info,axis=1), y[x[max_info][x[max_info]==li[i]].index.tolist()])
            if max1 == "empty":
                dic[max_info][li[i]] = list(set(y[x[max_info][x[max_info]==li[i]].index.tolist()]))[0]
                # a = Counter(list(y[x[max_info][x[max_info]==li[i]].index.tolist()]))
                # dic[max_info][li[i]] = a.most_common()[0][0] # 这里要改，改成数量最多的
            else:
                dic[max_info][li[i]] = create_dic1(x[x[max_info]==li[i]].drop(max_info,axis=1),y[x[max_info][x[max_info]==li[i]].index.tolist()])
                # dic[max_info][li[i]] = create_dic1(x[x[max_info]==li[i]].drop(max_info,axis=1),y[x[max_info][x[max_info]==li[i]].index.tolist()])
            # print(y[x[max_info][x[max_info]==li[i]].index.tolist()])
    return dic

def create_dic(x, y):
    # TODO 如果一开始就是全部分好的，max_info会是empty，数据中找不到这一列
    # 获取当前列的最大信息增益列
    max_info = cal_info_gain(x,y)
    # 获取当前列有几个分类
    li = list(set(x[max_info]))
    dic_result[max_info] = {}
    # print(dic_result)
    # 遍历每一个分类
    for i in range(len(li)):
        if x[x[max_info]==li[i]].shape[1] == 1:
            a = Counter(list(y[x[max_info][x[max_info]==li[i]].index.tolist()]))
            dic_result[max_info][li[i]] = a.most_common()[0][0]
        else:
            max1 = cal_info_gain(x[x[max_info]==li[i]].drop(max_info,axis=1), y[x[max_info][x[max_info]==li[i]].index.tolist()])
            if max1 == "empty":
                dic_result[max_info][li[i]] = list(set(y[x[max_info][x[max_info]==li[i]].index.tolist()]))[0]
                # a = Counter(list(y[x[max_info][x[max_info]==li[i]].index.tolist()]))
                # dic_result[max_info][li[i]] = a.most_common()[0][0] # 这里要改，改成数量最多的
            else:            
                dic_result[max_info][li[i]] = create_dic1(x[x[max_info]==li[i]].drop(max_info,axis=1),y[x[max_info][x[max_info]==li[i]].index.tolist()])
        # print(y[x[max_info][x[max_info]==li[i]].index.tolist()])
    return dic_result

def get_result(data,decision_tree):
    key = list(decision_tree.keys())[0]
        # print(isinstance(decision_tree[key][data[key].loc[i]],dict))
    if isinstance(decision_tree[key][data[key]],dict):
        result = get_result(data,decision_tree[key][data[key]])
        return result
    else:
        return decision_tree[key][data[key]]
        
        # print(data[key].loc[i])
    # print(key)

def pred(data,dic):
    result_li = []
    for i in range(data.shape[0]):
        data_single = data.iloc[i]
        # print(data_single)
        result = get_result(data_single,dic)
        result_li.append(result)
        # print(result)
    return result_li


def tree_plot(di):
    pass

x, y = data_load("D:\machine_learning\decision_tree\play_tennis")
dic = create_dic(x,y) # 得到决策树字典，接下来就是画图，先放下了
print(dic)
data = pd.read_table(r"D:\machine_learning\decision_tree\test", encoding="utf-8",delimiter=" ")
predict = pred(data,dic)





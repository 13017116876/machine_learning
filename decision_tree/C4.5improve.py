# 通常C4.5选取特征是采取这种方式，先得到信息增益大于平均值的，然后再其中求信息增益率最大的
import pandas as pd
import math
from collections import Counter
THRESHOLD = 0.005

dic_result = {}
def data_load(file,column):
    """
    read dataset
    file：the data file
    cloumn:the num of data column
    return: feature data and label data
    """
    if column >= 1:
        data = pd.read_table(file, encoding="utf-8",delimiter=" ")
        X_data = data.iloc[:,0:column-1]
        Y_data = data.iloc[:,column-1]
    else:
        raise Exception("column must >0")
    return X_data,Y_data

def cal_y_cato(y):
    a = Counter(list(y))
    catogory = a.most_common()[0][0]
    return catogory

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
    max_info_gain_ratio = 0
    # 计算熵
    init_entropy = cal_shanong(y)
    # 这里判断是否已经全部是一种结果
    if init_entropy == 0:
        return "empty"
    # 遍历x的每一列
    for i in range(x.shape[1]):
        condition_x = x.iloc[:,i]
        punish = cal_shanong(condition_x)
        # 计算每一列的条件熵
        condition_entropy = cal_condition_entropy(condition_x,y)
        # 得到每一列信息增益
        dic[condition_x.name] = init_entropy - condition_entropy # 防止punish为0
    avg = sum(dic.values())/len(dic)
    for j in range(x.shape[1]):
        condition_x = x.iloc[:,j]
        punish = cal_shanong(condition_x)
        # 计算每一列的条件熵
        condition_entropy = cal_condition_entropy(condition_x,y)
        if init_entropy - condition_entropy >= avg:

        # 得到每一列信息增益
            if (init_entropy - condition_entropy)/(punish+0.01) > max_info_gain_ratio:
                max_info_gain_ratio = (init_entropy - condition_entropy)/(punish+0.01)
                max_info = condition_x.name
        else:
            continue
    if max_info_gain_ratio < THRESHOLD:
        return "cancel"
    return max_info

def create_dic(x,y):
    dic = {}
    max_info = cal_info_gain(x,y)
    dic[max_info] = {}
    # 获取当前列有几个分类
    li = list(set(x[max_info]))
    # print(dic_result)
    # 遍历每一个分类
    for i in range(len(li)):
        if x.shape[1] == 1: # 如果此时只有一列了，则此时根据该种类中哪个分类多归为哪个标签
            dic[max_info][li[i]] = cal_y_cato(y[x[max_info][x[max_info]==li[i]].index.tolist()])
        else:
            # 获取去除当前列情况下的继续求最大增益
            max1 = cal_info_gain(x[x[max_info]==li[i]].drop(max_info,axis=1), y[x[max_info][x[max_info]==li[i]].index.tolist()])
            # 如果max1等于empty就代表此时该小数据已经分隔开
            if max1 == "empty":
                dic[max_info][li[i]] = list(set(y[x[max_info][x[max_info]==li[i]].index.tolist()]))[0]
            elif max1 == "cancel":
                dic[max_info][li[i]] = cal_y_cato(y[x[max_info][x[max_info]==li[i]].index.tolist()])
            # 否则就再这个基础上继续求最大信息增益
            else:
                dic[max_info][li[i]] = create_dic(x[x[max_info]==li[i]].drop(max_info,axis=1),y[x[max_info][x[max_info]==li[i]].index.tolist()])
    return dic

def classfier(x, y):
    if x.empty:# 如果传入的数据没有x，则直接返回y值较多的那个
        dic_result["result"] = cal_y_cato(y)
        return dic_result
    # 获取当前列的最大信息增益列
    max_info = cal_info_gain(x,y)
    # 如果一开始就是全部分好的，result的值就直接为一个值
    if max_info == "empty":
        dic_result["result"] = list(set(y))[0]
    elif max_info == "cancel": # 如果该特征的信息增益率仍小于预先设定的一个阈值v ，就说明属性集中所有的属性均不能作为比较良好的分类标准，不如不分类
        dic_result["result"] = cal_y_cato(y)
        return dic_result
    else:
        # 否则使用创建字典
        dic_result["result"] = create_dic(x,y)
    return dic_result


def get_result(data,decision_tree):
    key = list(decision_tree.keys())[0] # 获取字典第一个值
    if isinstance(decision_tree[key][data[key]],dict):# 如果当前列的该数值是字典，则递归这个字典
        result = get_result(data,decision_tree[key][data[key]])
        return result
    else:
        return decision_tree[key][data[key]] # 如果当前列该数值不是字典，返回该结果

def pred(data,dic):
    result_li = [] # 结果集列表
    if isinstance(dic["result"],dict): # 判断result的值是否是字典
        for i in range(data.shape[0]): #遍历数据每一行
            data_single = data.iloc[i] # 获取当前行
            result = get_result(data_single,dic["result"]) #获取结果
            result_li.append(result) #结果添加到列表
    else: # 如果不是字典就说明直接分出了结果
        result_li.append(dic["result"])
    return result_li

def tree_plot(di):
    pass

x, y = data_load(r"D:\machine_learning\decision_tree\play_tennis",5)
dic = classfier(x,y) # 得到决策树字典，接下来就是画图，先放下了
print(dic)
data = pd.read_table(r"D:\machine_learning\decision_tree\test", encoding="utf-8",delimiter=" ")
predict = pred(data,dic)
print(predict)








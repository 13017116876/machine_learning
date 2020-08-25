# 由于ID3采用的信息增益的方法会对属性值较多的特征更有优势，会得到更大的值
#ID3 和C4.5主要的不同在于计获取分裂特征的依据是一个依据信息增益，一个依据信息增益率
# 通常采用悲观剪枝方法，后剪枝，通过递归的方式从下往上针对每一个叶子结点，评估用一个叶子结点来代替这颗子树是否有益，通常使用验证集，如果剪枝后错误率下降了，则这颗子树就会被替换掉
# TODO 还差处理缺失值和剪枝，还有连续值
#C4.5的缺点：
    # 1.C4.5用的是多叉树，用二叉树效率更高
    # 2.只能用于分类
    # 3.使用熵模型有大量耗时的对数操作，连续值还有排序运算
    # 4.构造树时，对数值属性需要按照其大小顺序，从中选择一个分割点，所以只适合于能够驻留于内存的数据集，当训练集大得无法在内存容纳时，程序无法运行。
"""
1.首先对集合D和属性集a进行判断。若D中所有实例均属于同一类，则决策树T为单节点树，该类即为该结点的类。若属性集a为空集，则T也为单节点树，将样本集D中实例数最多的类作为该结点的类。
2.如不为以上两种情况，则计算属性集a中每一个属性对D的信息增益率。找到其中信息增益率最大的那个特征ai
3.如果该特征的信息增益率仍小于预先设定的一个阈值v ，就说明属性集中所有的属性均不能作为比较良好的分类标准，不如不分类。则T为单节点树，将样本集D中实例数最多的类作为该结点的类。
4.否则，对该属性的每一个可能值，将该属性取对应值的所有样本分出来构建子结点。并从属性集中剔除这个已经用过的属性。
5.对每个子结点，递归调用上述算法，直至所有子结点均为叶结点为止。
"""
# # 为了解决ID3对属性值较多的有优势而提出的C4.5算法显得对属性值较少的更有优势，为了平衡这两个算法，我们通常先取信息增益大于平均值的，然后在计算他们的信息增益率，这个代码写在C4.5improve中
# 缺失值处理， 主要是在于计算信息增益时候的改变 https://blog.csdn.net/leaf_zizi/article/details/83503167这篇文章讲的清晰，有例题
# 缺失值目前可以跑通，内部权重细节有待于进一步修改

import numpy as np
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

def cal_shanong_missing(y,y_weight):
    entropy = 0
    cato = list(set(y))
    # print(cato)
    for i in range(len(cato)):
        va = np.dot(np.ones(len(y[y==cato[i]])),y_weight[y[y==cato[i]].index.tolist()].T)/np.dot(np.ones(len(y)),y_weight[y.index.tolist()].T)
        entropy += -(va) * math.log(va) 
    return entropy

def cal_shanong(y):
    entropy = 0
    cato = list(set(y))
    # print(cato)
    for i in range(len(cato)):
        va = len(y[y==cato[i]])/len(y)
        entropy += -(va) * math.log(va)
    return entropy

def cal_condition_entropy_missing(x,y,y_weight):
    condition_entropy = 0
    cato = list(set(x))
    for i in range(len(cato)):
        va = np.dot(np.ones(len(x[x==cato[i]])),y_weight[x[x==cato[i]].index.tolist()].T)/np.dot(np.ones(len(x)),y_weight[x.index.tolist()].T)
        y_condition = y[x[x==cato[i]].index.tolist()]
        condition_entropy += va * cal_shanong_missing(y_condition,y_weight)
    return condition_entropy

def cal_condition_entropy(x,y):
    condition_entropy = 0
    cato = list(set(x))
    for i in range(len(cato)):
        va = len(x[x==cato[i]])/len(x)
        y_condition = y[x[x==cato[i]].index.tolist()]
        condition_entropy += va * cal_shanong(y_condition)
    return condition_entropy

def cal_info_gain_missing(x,y,y_weight):
    max_info_gini_rato = 0
    init_entropy1 = cal_shanong(y) # 这里判断是否全为一个类别
    # 这里判断是否已经全部是一种结果
    if init_entropy1 == 0:
        return "empty"
    for i in range(x.shape[1]): # 遍历每一列 
        x_correct = x.iloc[:,i].notnull() # 获取没缺失值的x数据
        y_correct = y[x_correct] # 获取非空的y
        init_entropy = cal_shanong_missing(y_correct,y_weight) # 初始熵
        p = np.dot(np.ones(len(y_correct)),y_weight[y_correct.index.tolist()].T) / np.dot(np.ones(len(y)), y_weight[y.index.tolist()].T)
        # p = len(y_correct)/len(y) # 非缺失值比例
        condition_entropy = cal_condition_entropy_missing(x.iloc[:,i][x_correct],y_correct,y_weight)
        punish = cal_shanong_missing(x.iloc[:,i][x_correct],y_weight)
        info_gain = p*(init_entropy - condition_entropy)/(punish+0.01) 
        if info_gain > max_info_gini_rato: # 如果当前的信息增益率大于之前的信息增益率，则更新max_info_gini_rato和max_info
            max_info_gini_rato = info_gain
            max_info = x_correct.name
    if max_info_gini_rato < THRESHOLD:
        return "cancel"
    return max_info


def create_dic_missing(x,y,y_weight,y_num):
    dic = {} 
    max_info = cal_info_gain_missing(x,y,y_weight) # 得到最大的信息增益
    dic[max_info] = {}
    # 获取当前列有几个分类
    li = x[max_info].unique() #list(set(x[max_info]))
    # print(dic_result)
    # 遍历每一个分类
    for i in range(len(li)): 
        print(li[i])
        print(li[i] == None)
        if li[i] == None:
            continue
        y_weight1 = pd.Series(np.ones(y_num))
        y_weight1[x[max_info][x[max_info].isnull()].index.tolist()] = len(y[x[max_info]==li[i]])/len(y[x[max_info].notnull()]) # 这里是因为x[max_info].isnull()是六维的，y_weight是14维
        # print(y_weight)
        if x.shape[1] == 1:  # 如果此时只有一列了，则此时根据该种类中哪个分类多归为哪个标签
            dic[max_info][li[i]] = cal_y_cato(y[x[max_info][x[max_info]==li[i]].index.tolist()])
        else:
            # 获取去除当前列情况下的继续求最大增益
            max1 = cal_info_gain_missing(x[(x[max_info]==li[i]) | (x[max_info].isnull())].drop(max_info,axis=1), y[x[(x[max_info]==li[i]) | (x[max_info].isnull())].index.tolist()],y_weight1[x[(x[max_info]==li[i]) | (x[max_info].isnull())].index.tolist()])
            # 如果max1等于empty就代表此时该小数据已经分隔开
            if max1 == "empty":
                dic[max_info][li[i]] = list(set(y[x[max_info][x[max_info]==li[i]].index.tolist()]))[0]
            elif max1 == "cancel":
                dic[max_info][li[i]] = cal_y_cato(y[x[max_info][x[max_info]==li[i]].index.tolist()])
            # 否则就再这个基础上继续求最大信息增益
            else:
                dic[max_info][li[i]] = create_dic_missing(x[(x[max_info]==li[i]) | (x[max_info].isnull())].drop(max_info,axis=1),y[x[(x[max_info]==li[i]) | (x[max_info].isnull())].index.tolist()],y_weight1[x[(x[max_info]==li[i]) | (x[max_info].isnull())].index.tolist()],y_num)
    return dic

def classfier(x, y,y_weight):
    if x.empty:# 如果传入的数据没有x，则直接返回y值较多的那个
        dic_result["result"] = cal_y_cato(y)
        return dic_result
    # 获取当前列的最大信息增益列
    max_info = cal_info_gain_missing(x,y,y_weight)
    # 如果一开始就是全部分好的，result的值就直接为一个值
    if max_info == "empty":
        dic_result["result"] = list(set(y))[0]
    elif max_info == "cancel": # 如果该特征的信息增益率仍小于预先设定的一个阈值v ，就说明属性集中所有的属性均不能作为比较良好的分类标准，不如不分类
        dic_result["result"] = cal_y_cato(y)
        return dic_result
    else:
        # 否则使用创建字典
        y_num = len(y)
        dic_result["result"] = create_dic_missing(x,y,y_weight,y_num)
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
x.iloc[1,2] = None # 为了处理缺失值问题，设置的一个缺失值
x.iloc[2,0] = None 
y_weight = pd.Series(np.ones(len(y)))
dic = classfier(x,y,y_weight) # 得到决策树字典，接下来就是画图，先放下了
print(dic)
# data = pd.read_table(r"D:\machine_learning\decision_tree\test", encoding="utf-8",delimiter=" ")
# predict = pred(data,dic)
# print(predict)






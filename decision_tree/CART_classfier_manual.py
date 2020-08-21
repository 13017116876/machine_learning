# CART建立的是二叉树，非多叉树,还差处理缺失值和剪枝
# 流程：
    # 算法从根节点开始，用训练集递归建立CART分类树
    # 1.对于当前的节点数据集D，如果样本个数小于阈值或没有特征，则返回决策树，当前节点停止递归
    # 2.计算样本D的基尼系数，如果基尼系数小于阈值，则返回决策树子树，当前节点停止递归
    # 3.计算当前节点现有的各个特征的各个特征值对数据集D的基尼系数，要处理缺失值和连续值
    # 4.在计算出来的各个特征的各个特征值对数据集D的基尼系数中，选择基尼系数最小的特征A和对应的特征值a。根据这个最优特征和最优特征值，把数据集划分成两部分D1和D2，同时建立当前节点的左右节点，做节点的数据集D为D1，右节点的数据集D为D2
    # 5.对左右子树递归调用1-4步骤
import pandas as pd
from collections import Counter
SAMPLETHRESHOLD = 3 #样本个数的阈值
GINITHRESHOLD = 0.03
dic_result = {}
def data_load(file,column):
    """
    read dataset
    file：the data file
    cloumn:the num of data column
    return: feature data and label data
    """
    data = pd.read_table(file, encoding="utf-8",delimiter=" ")
    if column >= 1:
        X_data = data.iloc[:,0:column-1]
        Y_data = data.iloc[:,column-1]
    else:
        raise Exception("column must >0")
    X_data = X_data.astype("str")
    Y_data = Y_data.astype("str")
    return X_data,Y_data

def cal_y_cato(y):
    a = Counter(list(y))
    catogory = a.most_common()[0][0]
    return catogory

def cal_gini(y):
    gini = 0
    cato = list(set(y))
    for i in range(len(cato)):
        va = len(y[y==cato[i]])/len(y)
        gini += va**2
    return 1-gini

def cal_min_gini(x,y):
    init = 100
    dic = {}
    for i in range(x.shape[1]): # 遍历x的每一列
        condition_x = x.iloc[:,i] # 得到x的列
        cato1 = list(set(condition_x)) # 得到x列有多少类
        dic[condition_x.name] = {}
        for j in range(len(cato1)): # 遍历这一列的每一类，为了求出最小分类
            va = len(condition_x[condition_x==cato1[j]])/len(x) # 计算这一类的数量
            # print(y)
            # print(condition_x[condition_x==cato1[j]].index.tolist())
            y_condition = y[condition_x[condition_x==cato1[j]].index.tolist()] # 获取这一类的y的数据
            # print(y_condition)
            gini = cal_gini(y_condition) # 计算这一类y数据的gini系数
            va1 = len(condition_x[condition_x!=cato1[j]])/len(x) # 获取不是这一列的数量
            y_condition_rev = y[condition_x[condition_x!=cato1[j]].index.tolist()] # 获取不是这一列的y的数据
            gini_rev = cal_gini(y_condition_rev) # 计算不是这一列的y的gini系数
            feature_gini = va * gini + va1 * gini_rev # 得到这一类的gini系数
            if feature_gini < init:
                init = feature_gini
                min_feature = condition_x.name
                min_split_value = cato1[j]
    if init == 0:
        return "empty","empty"
    
    if init < GINITHRESHOLD:
        return "cancel","cancel"
            # dic[condition_x.name][cato1[j]] = featurn_gini
    return min_feature, min_split_value

# def cal_info_gain(x,y):
#     init_gini_gain = cal_gini(y)

def cal_dic(x,y):
    dic = {} # 初始化字典
    min_feature, min_split_value= cal_min_gini(x,y) # 获取基尼系数最小的特征
    print(min_feature)
    dic[min_feature] = {} 
    if x.shape[1] == 1: #如果x只有一个特征
        dic[min_feature][min_split_value] = cal_y_cato(y[x[min_feature][x[min_feature]==min_split_value].index.tolist()]) # 以该数据min_split_value特征分裂
        dic[min_feature]["not"+min_split_value] = cal_y_cato(y[x[min_feature][x[min_feature]!=min_split_value].index.tolist()]) # 分割开的数据集按标签概率分类
    else:
        # 先判断样本大小是否小于阈值
        if len(y[x[min_feature][x[min_feature]==min_split_value].index.tolist()]) < SAMPLETHRESHOLD:
            dic[min_feature][min_split_value] = cal_y_cato(y[x[min_feature][x[min_feature]==min_split_value].index.tolist()])
        else:
            # 对分裂后的子列求最小基尼系数
            min_feature_new,min_split_value_new = cal_min_gini(x[x[min_feature]==min_split_value].drop(min_feature,axis=1), y[x[min_feature][x[min_feature]==min_split_value].index.tolist()])
            if min_feature_new == "empty": #如果分裂后的y值一样，直接分类
                dic[min_feature][min_split_value] = list(set(y[x[min_feature][x[min_feature]==min_split_value].index.tolist()]))[0] 
                # dic[min_feature]["not"+min_split_value] = list(set(y[x[min_feature][x[min_feature]!=min_split_value].index.tolist()]))[0]
            elif min_feature_new  == "cancel": # 如果分裂后gini系数小于gini阈值，也直接分类
                dic[min_feature][min_split_value] = cal_y_cato(y[x[min_feature][x[min_feature]==min_split_value].index.tolist()])
            else: # 否则递归          
                dic[min_feature][min_split_value] = cal_dic(x[x[min_feature]==min_split_value].drop(min_feature,axis=1),y[x[min_feature][x[min_feature]==min_split_value].index.tolist()])
        
        if len(y[x[min_feature][x[min_feature]!=min_split_value].index.tolist()]) < SAMPLETHRESHOLD:
            dic[min_feature]["not"+min_split_value] = cal_y_cato(y[x[min_feature][x[min_feature]!=min_split_value].index.tolist()])
        else:
            min_feature_new,min_split_value_new = cal_min_gini(x[x[min_feature]!=min_split_value].drop(min_feature,axis=1), y[x[min_feature][x[min_feature]!=min_split_value].index.tolist()])
            if min_feature_new == "empty":
                dic[min_feature]["not"+min_split_value] = list(set(y[x[min_feature][x[min_feature]!=min_split_value].index.tolist()]))[0]
            elif min_feature_new  == "cancel": # 如果分裂后gini系数小于gini阈值，也直接分类
                dic[min_feature]["not"+min_split_value] = cal_y_cato(y[x[min_feature][x[min_feature]!=min_split_value].index.tolist()])
            else: # 否则递归          
                dic[min_feature][min_split_value] = cal_dic(x[x[min_feature]!=min_split_value].drop(min_feature,axis=1),y[x[min_feature][x[min_feature]!=min_split_value].index.tolist()])
    return dic

def classfier(x,y):
    if x.empty:# 如果传入的数据没有x，则直接返回y值较多的那个
        dic_result["result"] = cal_y_cato(y)
        return dic_result
    if len(y) < SAMPLETHRESHOLD: # 如果当前传入样本数量小于样本阈值，也直接返回y值较多的那个
        dic_result["result"] = cal_y_cato(y)
        return dic_result
    min_feature, min_split_value= cal_min_gini(x,y) # 获取基尼系数最小的特征
    if min_feature == "empty": # 如果此时全部为一个种类
        dic_result["result"] = list(set(y))[0]
        return dic_result
    if min_feature == "cancel": # 如果此时所有的gini系数都小于阈值
        dic_result["result"] = cal_y_cato(y)
        return dic_result
    dic_result["result"] = cal_dic(x,y)
    return dic_result


x, y = data_load(r"D:\machine_learning\decision_tree\play_tennis",5)  # read dataset
dic_result = classfier(x,y)
print(dic_result)

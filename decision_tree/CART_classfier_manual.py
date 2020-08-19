# CART建立的是二叉树，非多叉树,暂时初步完成，后面可能还有很多要改
import pandas as pd
from collections import Counter
result_dic = {}
def data_load(file):
    data = pd.read_table(file, encoding="utf-8",delimiter=" ")
    X_data = data.iloc[:,0:4].astype("str")
    Y_data = data.iloc[:,4]
    return X_data,Y_data

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
            y_condition = y[condition_x[condition_x==cato1[j]].index.tolist()] # 获取这一类的y的数据
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
            # dic[condition_x.name][cato1[j]] = featurn_gini
    return min_feature, min_split_value

# def cal_info_gain(x,y):
#     init_gini_gain = cal_gini(y)

def cal_dic1(x,y):
    dic = {}
    min_feature, min_split_value= cal_min_gini(x,y) # 获取基尼系数最小的特征
    dic[min_feature] = {}

    if x.shape[1] == 1:
        a = Counter(list(y[x[min_feature][x[min_feature]==min_split_value].index.tolist()]))
        dic[min_feature][min_split_value] = a.most_common()[0][0]
        b = Counter(list(y[x[min_feature][x[min_feature]==min_split_value].index.tolist()]))
        print(min_split_value)
        dic[min_feature]["not"+min_split_value] = b.most_common()[0][0]
    
    else:
        min_feature_new,min_split_value_new = cal_min_gini(x[x[min_feature]==min_split_value].drop(min_feature,axis=1), y[x[min_feature][x[min_feature]==min_split_value].index.tolist()])
        if min_feature_new == "empty":
            dic[min_feature][min_split_value] = list(set(y[x[min_feature][x[min_feature]==min_split_value].index.tolist()]))[0]
        else:            
            dic[min_feature][min_split_value] = cal_dic1(x[x[min_feature]==min_split_value].drop(min_feature,axis=1),y[x[min_feature][x[min_feature]==min_split_value].index.tolist()])

        min_feature_new,min_split_value_new = cal_min_gini(x[x[min_feature]!=min_split_value].drop(min_feature,axis=1), y[x[min_feature][x[min_feature]!=min_split_value].index.tolist()])
        if min_feature_new == "empty":
            dic[min_feature]["not"+min_split_value] = list(set(y[x[min_feature][x[min_feature]!=min_split_value].index.tolist()]))[0]
        else:            
            dic[min_feature]["not"+min_split_value] = cal_dic1(x[x[min_feature]!=min_split_value].drop(min_feature,axis=1),y[x[min_feature][x[min_feature]!=min_split_value].index.tolist()])
    return dic

def cal_dic(x,y):
    dic_result = {}
    min_feature, min_split_value= cal_min_gini(x,y) # 获取基尼系数最小的特征
    dic_result[min_feature] = {}

    if x.shape[1] == 1:
        a = Counter(list(y[x[min_feature][x[min_feature]==min_split_value].index.tolist()]))
        dic_result[min_feature][min_split_value] = a.most_common()[0][0]
        b = Counter(list(y[x[min_feature][x[min_feature]==min_split_value].index.tolist()]))
        dic_result[min_feature]["not"+min_split_value] = b.most_common()[0][0]
    
    else:
        min_feature_new,min_split_value_new = cal_min_gini(x[x[min_feature]==min_split_value].drop(min_feature,axis=1), y[x[min_feature][x[min_feature]==min_split_value].index.tolist()])
        if min_feature_new == "empty":
            dic_result[min_feature][min_split_value] = list(set(y[x[min_feature][x[min_feature]==min_split_value].index.tolist()]))[0]
        else:
            dic_result[min_feature][min_split_value] = cal_dic1(x[x[min_feature]==min_split_value].drop(min_feature,axis=1),y[x[min_feature][x[min_feature]==min_split_value].index.tolist()])

        min_feature_new,min_split_value_new = cal_min_gini(x[x[min_feature]!=min_split_value].drop(min_feature,axis=1), y[x[min_feature][x[min_feature]!=min_split_value].index.tolist()])
        if min_feature_new == "empty":
            dic_result[min_feature]["not"+min_split_value] = list(set(y[x[min_feature][x[min_feature]!=min_split_value].index.tolist()]))[0]
        else:
            dic_result[min_feature]["not"+min_split_value] = cal_dic1(x[x[min_feature]!=min_split_value].drop(min_feature,axis=1),y[x[min_feature][x[min_feature]!=min_split_value].index.tolist()])
    return dic_result


    # dic_result[min_feature][min_split_value] = cal_dic1(x,y)
    
    # return mini_gini
    # result_dic[mini_gini] = {}


x, y = data_load("D:\machine_learning\decision_tree\play_tennis")
dic_result = cal_dic(x,y)
print(dic_result)

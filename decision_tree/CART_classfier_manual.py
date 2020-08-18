# CART建立的是二叉树，非多叉树
import pandas as pd
def data_load(file):
    data = pd.read_table(file, encoding="utf-8",delimiter=" ")
    X_data = data.iloc[:,0:4]
    Y_data = data.iloc[:,4]
    return X_data,Y_data

x, y = data_load("D:\machine_learning\decision_tree\play_tennis")
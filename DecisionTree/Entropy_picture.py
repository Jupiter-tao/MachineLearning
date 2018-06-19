from sklearn.datasets import load_iris
import numpy as np
iris = load_iris()
train_data = np.concatenate((iris.data[0:40, :], iris.data[50:90, :], iris.data[100:140, :]), axis = 0)
train_target = np.concatenate((iris.target[0:40], iris.target[50:90], iris.target[100:140]), axis = 0)
test_data = np.concatenate((iris.data[40:50, :], iris.data[90:100, :], iris.data[140:150, :]), axis = 0)
test_target = np.concatenate((iris.target[40:50], iris.target[90:100], iris.target[140:150]), axis = 0)
 
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy')       #基于信息增益的决策树
clf.fit(train_data, train_target)                       # 训练决策树
predict_target = clf.predict(test_data)                 # 预测
sum(predict_target == test_target)                       # 预测成功的数量

from sklearn.tree import export_graphviz                 #可视化
with open("iris1.dot", 'w') as out:
    out = export_graphviz(clf, out_file=out)
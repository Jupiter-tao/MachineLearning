from sklearn.datasets import load_iris
import numpy as np
iris = load_iris()
features = iris['data']
target = iris['target']
feature_names = iris.feature_names
feature_names = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
class_names = iris.target_names
class_names = ['山鸢尾花', '变色鸢尾花', '维吉尼亚鸢尾花']

def calcEntropy(target):             #计算熵的函数
    label = np.unique(target)
    n = label.size
    count = np.zeros(n)
    p_i = np.zeros(n)
    for i in range(n):
        count[i] = target[target == label[i]].size
    p_i = np.divide(count, target.size)
    entropy = 0
    for i in range(n):
        entropy = entropy - p_i[i] * np.log2(p_i[i])
    return entropy

def calcConditionEntropy(feature, condition, target):  # 计算条件熵的函数
    true_condition = condition(feature)
    false_condition = true_condition == False
    target_true = target[true_condition]
    target_false = target[false_condition]
    p_true = target_true.size / target.size
    p_false = 1 - p_true
    entropy = p_true * calcEntropy(target_true) + p_false * calcEntropy(target_false)
    return entropy

H = calcEntropy(target)
# 加入鸢尾花花瓣宽度属性后，计算鸢尾花的条件熵
petal_width = features[:,3]
HC = calcConditionEntropy(petal_width, lambda feature: feature < 0.8, target)
print('鸢尾花默认的信息熵 ：', H)
print('带花瓣宽度的条件熵 ：', HC)

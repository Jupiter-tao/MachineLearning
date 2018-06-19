from sklearn.datasets import load_iris
import numpy as np
iris = load_iris()
features = iris['data']
target = iris['target']
feature_names = iris.feature_names
feature_names = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
class_names = iris.target_names
class_names = ['山鸢尾花', '变色鸢尾花', '维吉尼亚鸢尾花']

def calcEntropy(target):                         #计算熵的函数
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

def calcConditionEntropy(feature, condition, target):        # 计算条件熵的函数
    true_condition = condition(feature)
    false_condition = true_condition == False
    target_true = target[true_condition]
    target_false = target[false_condition]
    p_true = target_true.size / target.size
    p_false = 1 - p_true
    entropy = p_true * calcEntropy(target_true) + p_false * calcEntropy(target_false)
    return entropy

def generate_feature_points(feature, target):       # 生成特征的所有分界点，先对特征进行排序，然后将 target 有变动的地方作为分界点
    argsort = feature.argsort()
    f1 = feature[argsort]
    t1 = target[argsort]
    last_value = target[0]
    split_value = []
    for i in range(t1.size):
        if last_value != t1[i]:
            split_value.append((f1[i] + f1[i - 1]) / 2)
            last_value = t1[i]
    return np.array(split_value)

def calc_feature_entropy(feature, target):        #计算特征的所有分界点的条件熵，返回最小的那个条件熵
    min_entropy = float('inf')
    min_point = 0
    points = generate_feature_points(feature, target)
    for p in points:
        entropy = calcConditionEntropy(feature, lambda f: f < p, target)
        if entropy < min_entropy:
            min_entropy = entropy
            min_point = p
    if points.size == 0:
        min_entropy = 0
    return min_point, min_entropy

def select_feature(features, target):   #从所有特征中选择出条件熵最小的特征
    min_entropy = float('inf')
    min_point = 0
    num = features.shape[1]
    index = 0
    for i in range(num):
        point, entropy = calc_feature_entropy(features[:, i], target)
        if entropy <= min_entropy:
            index = i
            min_point = point
            min_entropy = entropy
    return index, min_point, min_entropy

class TreeNode:
    idn = 0
    feature_index = ''
    feature_point = 0
    feature_entropy = 0
    target_label = ''
    true_node = None
    false_node = None
    
    @staticmethod
    def decision(feature, point):
        return feature < point


def build_tree(features, target, idn):     #递归构建决策树
    node = TreeNode()
    index, point, entropy = select_feature(features, target)
    node.idn = idn
    node.feature_index = index
    node.feature_point = point
    node.feature_entropy = entropy
    node.target_label = target[np.argmax(np.bincount(target))]
    print('build tree node id %d, index %d, point %f, entropy %f, label %s ' %
          (idn, index, point, entropy, node.target_label))
    if entropy < 0.1:    #熵小于 0.1 时则结束创建子节点，防止过拟合
        print('too low entropy : ', entropy)
        return node

    f_copy = features.copy()
    t_copy = target.copy()
    f = f_copy[:, index]
    selector = node.decision(f, point)
                  #创建左右两个子节点
    idn = idn + 1
    node.true_node = build_tree(f_copy[selector, :], t_copy[selector], idn)
    idn = node.true_node.idn + 1
    node.false_node = build_tree(f_copy[selector == False], t_copy[selector == False], idn)
    return node

build_tree(features, target, 1)

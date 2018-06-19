from sklearn.datasets import load_iris
import numpy as np
iris = load_iris()
features = iris['data']
target = iris['target']
feature_names = iris.feature_names
feature_names = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
class_names = iris.target_names
class_names = ['山鸢尾花', '变色鸢尾花', '维吉尼亚鸢尾花']

iris_count = np.zeros(3)
iris_count[0] = target[target == 0].size
iris_count[1] = target[target == 1].size
iris_count[2] = target[target == 2].size
print("三种鸢尾花的数量分别为：", iris_count)
iris_probability = np.divide(iris_count, 150)
print("三种莺尾花的概率为：", iris_probability)
iris_h = -np.sum(iris_probability * np.log2(iris_probability))
print("鸢尾花的熵为：", iris_h)
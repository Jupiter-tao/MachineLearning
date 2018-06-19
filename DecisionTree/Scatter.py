from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()
features = iris['data']
target = iris['target']
feature_names = iris.feature_names
feature_names = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
class_names = iris.target_names
class_names = ['山鸢尾花', '变色鸢尾花', '维吉尼亚鸢尾花']

colors='rgby'
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.scatter(features[:,i], target, c=colors[i])
    plt.xlabel(feature_names[i],fontproperties='SimHei')
    plt.ylabel('花的种类',fontproperties='SimHei')
plt.suptitle("特征和鸢尾花种类散点图",fontproperties='SimHei')
plt.tight_layout(pad=3, w_pad=2, h_pad=2)
plt.show()
